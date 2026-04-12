import json
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import torch_functional as VF
from qwen_vl_utils import process_vision_info
from PIL import Image
from PIL.Image import Image as ImageObject
from io import BytesIO
import math
import numpy as np
from collections import defaultdict
import pandas as pd
from data.prompt.uitars_system_prompt import prepare_system_prompt
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return features

def collate_fn_dataproto(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}
class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

class PRMCriticDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        csv_file: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        max_prompt_length: int = 16384,
        max_response_length: int = 512,
        truncation: str = "left",
        max_pixels: int = None,
        min_pixels: int = None,
        fast_rollout: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.fast_rollout = fast_rollout
        
        self.dataset = pd.read_csv(csv_file)

    def __len__(self):
        return self.dataset.shape[0]

    def load_content(self, content):
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            return ''.join([self.load_content(c) for c in content])

        if isinstance(content, dict):
            if "text" in content:
                return content["text"]
            elif "image" in content:
                return "<|vision_start|><|image_pad|><|vision_end|>"
        
        raise ValueError(f"Unknown content type: {content}")
    
    def prepare_logits_input(self, messages, score):
        tokenizer = self.tokenizer
        processor = self.processor
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True)

        input_ids = []
        attention_mask = []

        image_count = 0
        pixel_values = []
        image_grid_thw = []
        # for prompt part:
        for turn_idx, msg in enumerate(messages[:-1]):
            role = msg['role']
            content = self.load_content(msg['content'])
            prompt = f'<|im_start|>{role}\n' + content + '<|im_end|>\n'

            cur_image_num = prompt.count("<|image_pad|>")
            if turn_idx == len(messages) - 2: # last user turn    
                prompt += '<|im_start|>assistant\n'
            if cur_image_num > 0:
                result = processor(image_inputs[image_count:image_count+cur_image_num], [prompt], add_special_tokens=False, return_tensors="pt")
                image_count += cur_image_num
            else:
                result = processor(None, [prompt], add_special_tokens=False, return_tensors="pt")
            
            cur_input_ids = result.pop('input_ids')[0]
            cur_attention_mask = result.pop('attention_mask')[0]
            if 'pixel_values' in result: # 10764, 1176
                pixel_values.append(result["pixel_values"])
            if 'image_grid_thw' in result:
                image_grid_thw.append(result["image_grid_thw"])

            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)  

        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
        image_grid_thw = torch.cat(image_grid_thw, dim=0) if len(image_grid_thw) > 0 else None

        position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        
        input_ids, attention_mask, position_ids= VF.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='right',
            )
        
        # for response part:
        role = messages[-1]['role']
        response = self.load_content(messages[-1]['content'])
        response = response + '<|im_end|>\n'
        enc_response = processor(None, [response], add_special_tokens=False, return_tensors="pt")
        response = pad_2d_list_to_length(enc_response['input_ids'], self.tokenizer.pad_token_id, max_length=self.max_response_length)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=self.tokenizer.eos_token_id, dtype=attention_mask.dtype
        )
        
        attention_mask = torch.cat((attention_mask, response_attention_mask[0]), dim=-1)
        input_ids = torch.cat((input_ids, response[0]), dim=-1)
        
        position_ids = get_rope_index(
            processor,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        
        input_ids, attention_mask, position_ids= VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length + self.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
            truncation='right',
        )
        position_ids = position_ids[:, : input_ids.size(0)]
        response_attention_mask = response_attention_mask[:, : self.max_response_length]
        response = response[:, : self.max_response_length][0]
            
        step_return_label = torch.tensor(score, dtype=torch.float32)
        masked_step_return_label = response_attention_mask[0].to(torch.float32) * step_return_label

        return input_ids, attention_mask, position_ids, response, masked_step_return_label, pixel_values, image_grid_thw
    
    def __getitem__(self, item):
        
        info = self.dataset.iloc[item]
        image_path = info['NAS_PATH']
        task = info['PROMPT']
        action = info['RESPONSE']
        if str(info['GPT_JUDGE']) == 'True':
            score = 1
        else:
            score = 0
            
        user_prompt = prepare_system_prompt(task)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Your are a helpful assistant."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": action,
                    }
                ]
            },
        ]
        input_ids, attention_mask, position_ids, responses, masked_step_return_label, pixel_values, image_grid_thw = self.prepare_logits_input(messages, score)
        data = {
            'raw_history_messages': messages,
            'input_ids': input_ids,
            'responses': responses,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'step_return_label': masked_step_return_label,
        }
        if pixel_values is not None:
            multi_modal_inputs = dict()
            multi_modal_inputs['pixel_values'] = pixel_values
            multi_modal_inputs['image_grid_thw'] = image_grid_thw
            data['multi_modal_inputs'] = multi_modal_inputs
        return data
