import torch
from qwen_vl_utils import process_vision_info

import base64
import io
from PIL import Image

def pil_image_to_base64_uri(pil_image: Image.Image) -> str:
    byte_io = io.BytesIO()
    img_format = pil_image.format if pil_image.format else 'PNG'
    pil_image.save(byte_io, format=img_format)
    image_bytes = byte_io.getvalue()
    encoded_bytes = base64.b64encode(image_bytes)
    encoded_string = encoded_bytes.decode('utf-8')
    data_uri = f"data:image/{img_format.lower()};base64,{encoded_string}"
    
    return data_uri

def find_subtensor_torch(main_tensor: torch.Tensor, sub_tensor: torch.Tensor) -> int:
    n, m = main_tensor.shape[0], sub_tensor.shape[0]

    if m == 0:
        return 0
    if n < m:
        return -1

    windows = main_tensor.unfold(0, m, 1)
    matches = (windows == sub_tensor)
    valid_matches = torch.all(matches, dim=1)
    indices = torch.where(valid_matches)[0]

    if indices.numel() > 0:
        return indices[0].item()
    else:
        return -1

def find_sublist_index(main_list, sublist):
    len_sub = len(sublist)
    for i in range(len(main_list) - len_sub + 1):
        if main_list[i:i+len_sub] == sublist:
            return i
    return -1

def load_content(content):
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        return ''.join([load_content(c) for c in content])

    if isinstance(content, dict):
        if "text" in content:
            return content["text"]
        elif "image" in content:
            return "<|vision_start|><|image_pad|><|vision_end|>"
    
    raise ValueError(f"Unknown content type: {content}")

def prepare_logits_input(messages, tokenizer, processor):
    
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
        content = load_content(msg['content'])
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
        if role in ["system", "user"]:
            labels.append(torch.full_like(cur_input_ids, -100))
        else:
            labels.append(cur_input_ids)

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


USER_PROMPT_WThought = """
**Role Definition:**
You are a meticulous evaluator for a Android GUI automation agent. Your primary mission is to analyze the agent's reasoning and proposed action in the context of a given task and the current user interface. You must determine if the agent's action is a correct and logical step towards completing the task, and judge whether the operation conforms to Android system specifications.

**Input Data:**
You will be provided with:
1. Instruction: The high-level goal.
2. Screenshot: A visual representation of the current GUI state.
3. Agent's Thought and Action: The reasoning process and the specific `Action` intended.

**Evaluation Criteria:**
Your output format should be `<think>...thought process...</think> Judgement:True or False`.
* **Return `True` if:** The proposed `Action` is logical, relevant, and productive based on a correct interpretation of the `Screenshot`.
* **Return `False` if:** The action is incorrect (illogical, misinterpretation of UI, redundant, or counter-productive).

---

**Example 1 (Correct Action):**
**Instruction:** Record an audio clip and save it with name "F3tb_presentation.m4a" using the Audio Recorder app.
**Screenshot:** [Home screen with "Audio Recorder" icon visible.]
**Agent's Thought and Action:**
Thought: I need to open the app. I see the "Audio Recorder" icon. I will tap it.
Action: `click(description='Audio Recorder icon')`
**Evaluation:**
<think>
Agent's Logic Analysis: The agent correctly identifies the first step and locates the icon.
Action Validation: The action is the most direct and logical step.
</think>
Judgement:True

---

**Example 2 (Incorrect Action):**
**Instruction:** Record an audio clip and save it with name "F3tb_presentation.m4a" using the Audio Recorder app.
**Screenshot:** [Inside Audio Recorder app. "Get Start" button visible.]
**Agent's Thought and Action:**
Thought: I don't see the app icon, so I must be in the wrong place. I need to go back.
Action: `swipe_up()`
**Evaluation:**
<think>
Agent's Logic Analysis: The agent flawed reasoning; it failed to recognize it is already in the app.
Action Validation: `swipe_up()` exits the app, which is counter-productive. Correct action is clicking get start.
</think>
Judgement:False

---

**Now, evaluate the following scenario:**

**Instruction:**
{instruction}

**Agent's Thought and Action:**
{agent_thought_and_action}

**Evaluation:**
"""
