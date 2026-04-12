# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs


def convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item

from typing import Optional

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin
from verl.models.transformers.qwen2_vl import get_rope_index
import verl.utils.torch_functional as VF
from qwen_vl_utils import process_vision_info

def get_tokenizer(model_path: str, **kwargs) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        # the EOS token in gemma2 & gemma3 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if tokenizer.pad_token_id is None:
        print("Pad token is None. Set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_processor(model_path: str, **kwargs) -> Optional[ProcessorMixin]:
    """Create a huggingface pretrained processor."""
    try:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    except Exception:
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor

import pandas as pd
from verl.utils.dataset.utils import USER_PROMPT_WThought

class PRMwoHistoryWThoughtSFTDataset(Dataset):
    def __init__(self, csv_file, config=None):
        config = config or {}
        self.config = config
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        assert self.truncation in ["error", "left", "right"]
        self.dataset = pd.read_csv(csv_file)
        self.tokenizer = get_tokenizer(
            config.model_path,
            trust_remote_code=False,
            use_fast=True,
        )
        self.processor = get_processor(
            config.model_path,
            trust_remote_code=False,
            use_fast=True,
        )
        self.min_pixels_value = 256
        self.max_pixels_value = 1270180
    
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

    def prepare_logits_input(self, messages):
        tokenizer = self.tokenizer
        processor = self.processor
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True)

        input_ids = []
        labels = []
        attention_mask = []

        image_count = 0
        pixel_values = []
        image_grid_thw = []
        for turn_idx, msg in enumerate(messages):
            role = msg['role']
            content = self.load_content(msg['content'])
            prompt = f'<|im_start|>{role}\n' + content + '<|im_end|>\n'

            cur_image_num = prompt.count("<|image_pad|>")                
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

            # if role in ["system", "user"]:
            #     labels.append(torch.full_like(cur_input_ids, -100))
            # else:
            #     labels.append(cur_input_ids)

            if role in ["system", "user"]:
                cur_labels = torch.full_like(cur_input_ids, -100)
            elif role == "assistant":
                cur_labels = cur_input_ids.clone()
                prompt_header = f'<|im_start|>{role}\n'
                header_ids = tokenizer.encode(prompt_header, add_special_tokens=False)
                header_len = len(header_ids)
                cur_labels[:header_len] = -100
            else:
                cur_labels = torch.full_like(cur_input_ids, -100)
            
            labels.append(cur_labels)

        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)  

        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
        image_grid_thw = torch.cat(image_grid_thw, dim=0) if len(image_grid_thw) > 0 else None
        data = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
        }
        return data

    def __getitem__(self, item):
        
        info = self.dataset.iloc[item]
        image_path = info['NAS_PATH']
        task = info['PROMPT']
        action = info['RESPONSE']
        thought = info['GPT_REASON']
        if str(info['GPT_JUDGE']) == 'True':
            score = 'True'
        else:
            score = 'False'
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
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT_WThought.format(instruction=task, agent_thought_and_action=action),
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f'<think>\n{thought}\n</think>\nScore:{score}',
                    }
                ]
            }
        ]
        inputs = self.prepare_logits_input(messages)
        position_ids = get_rope_index(
                self.processor,
                input_ids=inputs['input_ids'],
                image_grid_thw=inputs['image_grid_thw'],
                attention_mask=inputs['attention_mask'],
            )
        
        input_ids, attention_mask, position_ids, labels = VF.postprocess_data(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                position_ids=position_ids,
                max_length=self.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=False,
                truncation='right',
                labels=inputs['labels']
            )
        loss_mask = labels!=-100
        data = {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
        }
        return data
 
class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: str | list[str], tokenizer, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None
        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

    def __len__(self):
        return len(self.messages)

    def _process_message_tokens(
        self,
        messages: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        enable_thinking: Optional[bool] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process tokens for a single message or a group of messages.

        Args:
            messages: List of message dictionaries
            start_idx: Start index in messages list
            end_idx: End index in messages list
            is_assistant: Whether this is an assistant message
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (tokens, loss_mask, attention_mask)
        """
        if start_idx > 0:
            prev_applied_text = self.tokenizer.apply_chat_template(
                messages[:start_idx],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
                **self.apply_chat_template_kwargs,
            )
            if is_assistant:
                prev_applied_text_w_generation_prompt = self.tokenizer.apply_chat_template(
                    messages[:start_idx],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    tools=tools,
                    **self.apply_chat_template_kwargs,
                )

        else:
            prev_applied_text = ""

        cur_applied_text = self.tokenizer.apply_chat_template(
            messages[:end_idx],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            tools=tools,
            **self.apply_chat_template_kwargs,
        )
        # Get tokens for the current message only
        if is_assistant:
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text) :]
            generation_prompt_tokens = self.tokenizer.encode(
                generation_prompt_text,
                add_special_tokens=False,
            )
            _message_tokens = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text_w_generation_prompt) :],
                add_special_tokens=False,
            )
            message_tokens = generation_prompt_tokens + _message_tokens
            loss_mask = [0] * (len(generation_prompt_tokens)) + [1] * (
                len(message_tokens) - len(generation_prompt_tokens)
            )
        else:
            message_tokens = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text) :],
                add_special_tokens=False,
            )
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)

        return message_tokens, loss_mask, attention_mask

    def _validate_and_convert_tokens(
        self,
        full_tokens: torch.Tensor,
        concat_tokens: list[int],
        concat_loss_mask: list[int],
        concat_attention_mask: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate tokenization and convert to tensors.

        Args:
            full_tokens: Full conversation tokens
            concat_tokens: Concatenated tokens
            concat_loss_mask: Concatenated loss mask
            concat_attention_mask: Concatenated attention mask

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask) as tensors
        """
        full_tokens_list = full_tokens.tolist()

        if len(concat_tokens) != len(full_tokens_list) or not all(
            a == b for a, b in zip(concat_tokens, full_tokens_list, strict=True)
        ):
            logging.warning(
                f"Token mismatch detected! Full tokenization length: {len(full_tokens_list)}, Concatenated tokens "
                f"length: {len(concat_tokens)}. Using concatenated version."
                # f"full tokens text: {self.tokenizer.decode(full_tokens_list)}"
                # f"concat tokens text: {self.tokenizer.decode(concat_tokens)}"
            )
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.long),
                torch.tensor(concat_attention_mask, dtype=torch.long),
            )

        return (
            full_tokens,
            torch.tensor(concat_loss_mask, dtype=torch.long),
            torch.tensor(concat_attention_mask, dtype=torch.long),
        )

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None

        # First, get the full conversation tokens
        try:
            full_tokens = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                **self.apply_chat_template_kwargs,
            )
        except Exception as e:
            logging.error(
                f"Error applying chat template: {e}\nMessages: {messages}\nTools: {tools}\nEnable thinking: "
                f"{enable_thinking}"
            )
            raise

        # Track concatenated tokens for validation
        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []

        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                # Process assistant message
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, is_assistant=True, enable_thinking=enable_thinking, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            elif cur_messages["role"] == "tool":
                # Process consecutive tool messages
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, st, ed, enable_thinking=enable_thinking, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i = ed
            elif cur_messages["role"] in ["user", "system"]:
                # Process user or system message
                if cur_messages["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, enable_thinking=enable_thinking, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

        # Validate and convert tokens
        input_ids, loss_mask, attention_mask = self._validate_and_convert_tokens(
            full_tokens[0], concat_tokens, concat_loss_mask, concat_attention_mask
        )

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
