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

class GUIDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        messages,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        max_prompt_length: int = 1024,
        truncation: str = "left",
        max_pixels: int = None,
        min_pixels: int = None,
        fast_rollout: bool = False,
    ):
        self.messages = messages
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.fast_rollout = fast_rollout

    def __len__(self):
        return len(self.messages)
    
    
    def __getitem__(self, index):
        message = self.messages[index]

        tokenizer = self.tokenizer
        processor = self.processor
        
        prompt = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
                message, return_video_kwargs=True)

        row_dict = dict()
        row_dict["multi_modal_data"] = {"image": image_inputs} # [PIL.Image, ...]

        if not self.fast_rollout: 
            # Multi-turn conversation tokenization
            model_inputs = processor(image_inputs, [prompt], add_special_tokens=False, return_tensors="pt")
        
            input_ids = model_inputs.pop('input_ids')[0]
            attention_mask = model_inputs.pop('attention_mask')[0]

            row_dict["multi_modal_inputs"] = dict(model_inputs)
            
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            # to make sure dataproto can be created
            input_ids = torch.zeros((0,), dtype=torch.int64) 
            attention_mask = torch.zeros((0,), dtype=torch.int64)
            position_ids = torch.zeros((3, 0), dtype=torch.int64)
            row_dict['multi_modal_inputs'] = dict()

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        return row_dict

class AndroidlabTaskConfigDataset(Dataset):

    def __init__(
        self,
        data_path: str,
    ):
        self.data_path = data_path
          
        self.dataset = json.load(open(data_path, 'r'))

    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        domain = self.dataset[index]
        task_config = {}
        
        task_config['task_name'] = domain['task']
        task_config['task_id'] = domain['task_id']

        return task_config