import ast
import base64
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import copy
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from time import sleep
from typing import Any, Dict, List, Tuple

import adbutils
import cv2
import numpy as np
import openai
import ray
import requests
import torch
import uiautomator2 as u2
from dotenv import load_dotenv
from lxml.etree import XPathEvalError
from PIL import Image, ImageDraw, ImageFont
from uiautomator2.exceptions import ConnectError

from and_utils.packages import find_package
from and_utils.parser import compressed_xml_android
from data.prompt.uitars_system_prompt import prepare_system_prompt
from qwen_vl_utils import process_vision_info
from ui_tars.action_parser import parse_action_to_structure_output
import verl.utils.torch_functional as VF
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.dataset.utils import USER_PROMPT_WThought, pil_image_to_base64_uri
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.tokenizer import hf_processor, hf_tokenizer

@ray.remote(num_cpus=1)
class AndroidWorker:
    """
    A Ray worker that manages a single Android emulator via uiautomator2.
    """

    def __init__(
        self,
        worker_idx: int,
        ip_label: str,
        exp_name: str = "test_exp",
        max_steps: int = 15,
        outcome_ration: float = 5.0,
        config: Dict[str, Any] = None,
    ):
        avd_name = "Pixel_7_Pro_API_33"
        android_avd_home = '/app/.android/avd'
        emulator_path = "/app/.android/emulator/emulator"
        adb_path = "/app/.android/platform-tools/adb"
        temp_path = "./temp"
        no_window = True

        self.android_avd_home = os.path.expanduser(android_avd_home)
        self.emulator_path = os.path.expanduser(emulator_path)
        self.adb_path = os.path.expanduser(adb_path)
        self.worker_idx = worker_idx
        self.avd_name = avd_name
        self.exp_name = exp_name
        self.ip_label = ip_label
        self.outcome_ration = outcome_ration
        self.udid = f"emulator-{5554 + worker_idx * 2}"
        self.max_steps = max_steps
        self.temp_path = os.path.join(temp_path, f"env_{worker_idx}")
        os.makedirs(self.temp_path, exist_ok=True)
        self.no_window = no_window
        self.is_done = False
        self.d = None
        self.proc_emulator = None
        self.width = 1440
        self.height = 3120
        self.lambdax = 0.95
        self.marked_screenshots = []
        self.config = config
        self.max_pixels = config.data.max_pixels
        self.min_pixels = config.data.min_pixels
        self.tokenizer = hf_tokenizer(
            config.actor_rollout_ref.model.path,
            trust_remote_code=config.data.trust_remote_code,
            use_fast=True,
        )
        
        self.processor = hf_processor(
            config.actor_rollout_ref.model.path,
            trust_remote_code=config.data.trust_remote_code,
            use_fast=True,
        )
        
        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.process_reward = []
        self.use_prm = config.critic.use_prm
        self.outcome_reward = 0
        
        
    def _kill_running(self):
        # Kill any existing emulator with same udid
        try:
            out = subprocess.check_output([self.adb_path, "devices"]).decode()
            for line in out.splitlines():
                if line.startswith(self.udid):
                    subprocess.run([self.adb_path, "-s", self.udid, "emu", "kill"])
                    print(f'{self.udid} has been shut down.')
        except Exception:
            pass

    def _clone_avd(self):
        src_dir = os.path.join(self.android_avd_home, f"{self.avd_name}.avd")
        dst_name = f"{self.avd_name}_ray_{self.worker_idx}"
        dst_dir = os.path.join(self.android_avd_home, f"{dst_name}.avd")
        src_ini = os.path.join(self.android_avd_home, f"{self.avd_name}.ini")
        dst_ini = os.path.join(self.android_avd_home, f"{dst_name}.ini")
        
        print(f"Copying the AVD folder from {src_dir} to {dst_dir}")

        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir, ignore_errors=True)
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True) 
        with open(src_ini, "r") as f_src, open(dst_ini, "w") as f_dst:
            for line in f_src:
                f_dst.write(line.replace(self.avd_name, dst_name))
        
        # Update additional ini files in the cloned AVD directory
        for ini_name in ['config.ini', 'hardware-qemu.ini']:
            ini_path = os.path.join(dst_dir, ini_name)
            if os.path.exists(ini_path):
                with open(ini_path, 'r') as file:
                    lines = file.readlines()
                with open(ini_path, 'w') as file:
                    for line in lines:
                        # Update paths and AVD name/ID
                        new_line = line.replace(self.avd_name, dst_name)
                        file.write(new_line)

        # Update the snapshots' hardware.ini file if it exists
        snapshots_hw_ini = os.path.join(dst_dir, 'snapshots', 'default_boot', 'hardware.ini')
        if os.path.exists(snapshots_hw_ini):
            with open(snapshots_hw_ini, 'r') as file:
                lines = file.readlines()
            with open(snapshots_hw_ini, 'w') as file:
                for line in lines:
                    # Update AVD name/ID
                    new_line = line.replace(self.avd_name, dst_name)
                    file.write(new_line)

        self.clone_name = dst_name

    def _start_emulator(self):
        print(f"Starting the Emulator")
        port = int(self.udid.split("-")[-1])
        args = [
            self.emulator_path,
            "-avd", self.clone_name,
            "-no-audio", "-skip-adb-auth",
            "-no-boot-anim", "-gpu", "auto", "-no-snapshot-save",
            "-port", str(port)
        ]
        if self.no_window:
            args.append("-no-window")
            

        self.proc_emulator = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        
        # Wait for emulator to be ready
        for _ in range(20): # Wait up to 120 seconds
            try:
                result = subprocess.run([self.adb_path, "-s", self.udid, "shell", "getprop", "sys.boot_completed"], capture_output=True, text=True, timeout=10)
                if "1" in result.stdout.strip():
                    print(f"Emulator {self.udid} booted.")
                    sleep(15) # Extra wait time for stability
                    
                    return
            except subprocess.TimeoutExpired:
                print(f"Waiting for emulator {self.udid} to boot...")
            sleep(6)
        raise RuntimeError(f"Emulator {self.udid} failed to boot.")

    def _connect_device(self):
        for i in range(3):
            try:
                print(f"Connecting to device {self.udid} with uiautomator2...")
                self.d = u2.connect(self.udid)
                # ensure the device is responsive (healthcheck in newer uiautomator2)
                try:
                    self.d.healthcheck(timeout=120)
                except AttributeError:
                    # fallback: brief sleep if healthcheck not available
                    sleep(5)
                self.screen_size = self.d.window_size()
                print(f"Connected to {self.udid} successfully! Screen size: {self.screen_size}")
                return
            except Exception as e:
                print(f"Failed to connect to the device: {e}\n Retrying... ({i+1}/3)")
                if i == 2:
                    raise RuntimeError(f"Failed to connect to the device with uiautomator2, error: {e}")
                sleep(20)
        raise RuntimeError("Failed to connect to device")

    def _get_obs(self):
        """
        Returns the current UI hierarchy and element locations.
        """
        try:
            xml = self.d.dump_hierarchy(compressed=False)
            comp_xml, elem_loc = compressed_xml_android(xml, type="plain_text")
        except XPathEvalError as e:
            logging.warning(f"Invalid XPath predicate in UI dump: {e}. Skipping element locations.")
            comp_xml = xml
            elem_loc = {}
        except Exception as e:
            logging.warning(f"Uiautomator2 error, reconnecting device: {e}")
            # Reconnect to the device
            self._connect_device()
            xml = self.d.dump_hierarchy(compressed=False)
            comp_xml, elem_loc = compressed_xml_android(xml, type="plain_text")
        return comp_xml, elem_loc

    def _get_coords(self, box_str: str) -> Tuple[int, int]:
        """
        A helper function to parse a string representing absolute coordinates into an (x, y) point.
        - If the input is '[x, y]', it directly returns (x, y).
        - If the input is '[x1, y1, x2, y2]', it returns the center point (x, y).
        Args:
            box_str (str): A string in the format '[x, y]' or '[x1, y1, x2, y2]'.
        Returns:
            Tuple[int, int]: (x, y) absolute pixel coordinates.
        """
        coords = ast.literal_eval(box_str)
        
        if len(coords) == 2:
            return int(coords[0] * self.width), int(coords[1] * self.height)
        elif len(coords) == 4:
            x1, y1, x2, y2 = map(float, coords)
            center_x = int((x1 + x2) / 2 * self.width)
            center_y = int((y1 + y2) / 2 * self.height)
            return center_x, center_y
        else:
            raise ValueError(f"Invalid coordinate format in string: {box_str}")

    def _draw_circle_on_screenshot(self, x: int, y: int, radius: int = 30, color: Tuple[int, int, int] = (255, 0, 0)):
        """
        Draw a circle on the latest screenshot to mark the click position. Update history_messages and history_images. Not used in training.
        Args:
            x (int): The x-coordinate of the circle center.
            y (int): The y-coordinate of the circle center.
            radius (int): The radius of the circle, default is 30.
            color (Tuple[int, int, int]): The color of the circle, default is red (255, 0, 0).
        """
        
        if not self.history_images:
            print("No history images to draw on.")
            return
        
        screenshot = self.history_images[-1].copy()
        draw = ImageDraw.Draw(screenshot)
        
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], outline=color, width=5)
        
        print("--------draw circle on screenshot--------")
        self.marked_screenshots.append(screenshot)
    
    def _parse_action_to_command(self, action: Dict[str, Any]) -> bool:
        """
        Turn the structured action dict from the Agent into commands that uiautomator2 can execute.
        (Final version, adapted to your model's action space and coordinate logic. Here is a basic template for UI-Tars)

        Args:
            action (Dict[str, Any]): A dict in the format {"action_type": "...", "action_inputs": {...}}.
            
        Returns:
            bool: Indicates whether the operation was executed successfully.
        """
        
        action_type = action.get('action_type')
        inputs = action.get('action_inputs', {})

        try:
            if action_type == 'click':
                box_str = inputs.get('start_box')
                if not box_str: raise ValueError("'click' action requires 'start_box'")
                x, y = self._get_coords(box_str)
                self.d.click(x, y)
                # draw a circle on the screenshot for visualization
                self._draw_circle_on_screenshot(x, y)
                time.sleep(5)

            elif action_type == 'long_press':
                box_str = inputs.get('start_box')
                if not box_str: raise ValueError("'long_press' action requires 'start_box'")
                x, y = self._get_coords(box_str)
                self.d.long_click(x, y)
                # draw a circle on the screenshot for visualization
                self._draw_circle_on_screenshot(x, y)
                time.sleep(5)

            elif action_type == 'type':
                content = inputs.get('content')
                if content is None: raise ValueError("'type' action requires 'content'")
                self.d.send_keys(content, clear=False)
                time.sleep(1)

            elif action_type == 'scroll' or action_type == 'drag':
                start_box_str = inputs.get('start_box')
                end_box_str = inputs.get('end_box')
                if not start_box_str or not end_box_str:
                    raise ValueError(f"'{action_type}' action requires 'start_box' and 'end_box'")
                
                start_x, start_y = self._get_coords(start_box_str)
                end_x, end_y = self._get_coords(end_box_str)
                self.d.swipe(start_x, start_y, end_x, end_y)
                # draw a circle on the screenshot for visualization
                self._draw_circle_on_screenshot(start_x, start_y)
                self._draw_circle_on_screenshot(end_x, end_y)

                time.sleep(1)

            elif action_type == 'open_app':
                app_name = inputs.get('app_name').lower()
                if app_name == 'sms messenger':
                    app_name = 'simple sms'
                if not app_name: raise ValueError("'open_app' action requires 'app_name'")
                # package_name = self.APP_PACKAGE_MAP.get(app_name)
                package_name = find_package(app_name)
                
                if package_name:
                    # print(f"Found package name for '{app_name}': {package_name}")
                    self.d.app_start(package_name, stop=True)
                    current_app = self.d.app_current()
                else:
                    return False
                time.sleep(5)
            
            elif action_type == 'press_home':
                self.d.press("home")
                time.sleep(1)
            
            elif action_type == 'press_back':
                self.d.press("back")
                time.sleep(1)
            
            elif action_type == 'wait':
                time.sleep(2)
            
            elif action_type == 'finished':
                print("Finish action received. Task episode will end.")
                pass

            else:
                print(f"Unknown or unsupported action type: {action_type}")
                return False

            return True

        except Exception as e:
            print(f"Error executing action {action}: {e}")
            tb = traceback.extract_tb(e.__traceback__)
            print('Error info:', tb)
            return False

    def reset(self, task: dict, task_id: int):
        """
        Reset env: kill, clone, start emulator, connect driver, get initial obs.
        Returns dict with keys: env_idx, obs, is_done, format_reward.
        """
        self._kill_running()
        self._clone_avd()
        self._start_emulator()
        self._connect_device()

        self.d.press("back")
        obs, elem_loc = self._get_obs()
        self.elem_loc = elem_loc
        
        self.task = task
        
        self.task_id = task_id
        self.marked_screenshots = []
        
        self.process_reward = []
        self.outcome_reward = 0

        initial_screenshot = self.d.screenshot(format="pillow")

        user_prompt = prepare_system_prompt(task)
        
        init_messages = [
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
                        "image": initial_screenshot,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                ]
            }
        ]
        
        xml_init_messages = [
            {
                "role": "mobile",
                "content": obs
            },
        ]
        
        self.xml_history_messages = xml_init_messages

        self.history_images = [initial_screenshot]
        self.history_messages = init_messages
        self.is_done = False

        # return initial_screenshot, self.history_messages
        self.step_counter = 0
        return {
            'env_idx': self.worker_idx,
            'obs_messages': self.history_messages,
            'is_done': self.is_done,
            'format_reward': 0.0
        }

    def evaluate(self):
        xml_history_str = ""
        for history in self.xml_history_messages:
            if history['role'] == 'mobile':
                xml_history_str += f"<mobile>{history['content']}</mobile>\n"
            if history['role'] == 'uiagent':
                xml_history_str += f"<uiagent>{history['content']}</uiagent>\n"
        is_successful = outcome_judger(self.task, xml_history_str)
        
        if is_successful:
            self.outcome_reward = 1 * self.outcome_ration

        # PRM assign reward
        if self.use_prm:
            tasks_to_process = []
            for step_idx in range(self.step_counter):
                try:
                    action = self.history_messages[2*step_idx + 3: 2*step_idx + 4][0]['content'][0]['text']
                    image = self.history_messages[2*step_idx + 2: 2*step_idx + 3][0]['content'][0]['image']
                    tasks_to_process.append((self.task, action, image))
                except (IndexError, KeyError) as e:
                    print(f"Warning: Could not get action/image for step {step_idx}, skipping. Error: {e}")
                    continue

            def judge_step(args):
                task, action, image = args
                try:
                    base64_qwen = pil_image_to_base64_uri(image)
                    return process_judger(task, action, base64_qwen)
                except Exception as e:
                    print(f"Error during parallel process_judger call: {e}")
                    return 0 # or some other default error value

            max_workers = min(8, len(tasks_to_process))
            if max_workers > 0:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(judge_step, tasks_to_process))
                self.process_reward = results
            else:
                self.process_reward = []

        return is_successful

    def recover_step(self, history):
        self.history_messages = history
        original_image_width, original_image_height = self.width, self.height
        action_step = 0
        for message in history:
            if message['role'] == 'assistant':
                original_response = message['content'][0]['text']
                try:
                    action = parse_action_to_structure_output(
                        original_response,
                        factor=1000,
                        origin_resized_height=original_image_height,
                        origin_resized_width=original_image_width,
                        model_type="qwen25vl",
                        max_pixels=self.max_pixels,
                        min_pixels=self.min_pixels,
                    )[0]
                except Exception as e:
                    print("during recover, error: ", e)
                    tb = traceback.extract_tb(e.__traceback__)
                    print('Error info:', tb)
                    print('Error Response:', original_response)
                    continue
                
                print(f"Recovring history action step {action_step} : {action}")
                action_step += 1
                self._parse_action_to_command(action)
        
        if history[-1]['role'] == 'user' and history[-1]['content'][0]['type'] == 'image':
            if self.step_counter == self.max_steps:
                self.is_done = True
            return {
                'env_idx': self.worker_idx,
                'obs_messages': self.history_messages,
                'is_done': self.is_done,
                'format_reward': 0.0
            }
        
        try:
            new_screenshot = self.d.screenshot(format="pillow")
        except (ConnectError, adbutils.AdbError, requests.RequestException) as e:
            raise ConnectError()
        except Exception as e:
            print("evaluate failed due to ", e)
            tb = traceback.extract_tb(e.__traceback__)
            print('Error info:', tb)
            print('Error Response:', original_response)
            self.history_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "The simulator fails to capture a screenshot properly.",
                        }
                    ]
                })
            return {
                'env_idx': self.worker_idx,
                'obs_messages': self.history_messages,
                'is_done': self.is_done,
                'format_reward': 0.0
            }
        self.history_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": new_screenshot,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            ]
        })
        self.step_counter += 1
        if self.step_counter == self.max_steps:
            self.is_done = True
        return {
            'env_idx': self.worker_idx,
            'obs_messages': self.history_messages,
            'is_done': self.is_done,
            'format_reward': 0.0
        }
        
    def step(self, original_response):
        """
        Take a step in the environment by executing the action.
        Args:
            action (Dict[str, Any]): The action from the Agent.
        Returns:
            Tuple[Image.Image, bool, List[Dict]]:
                - new_screenshot (Image.Image): The screenshot after executing the action.
                - is_done (bool): Flag indicating if the task is done [cite: 8].
                - history (List[Dict]): The updated history trajectory (intermediate results).
        """
        
        original_image_width, original_image_height = self.width, self.height
        try:
            action = parse_action_to_structure_output(
                original_response,
                factor=1000,
                origin_resized_height=original_image_height,
                origin_resized_width=original_image_width,
                model_type="qwen25vl",
                max_pixels=self.max_pixels,
                min_pixels=self.min_pixels,
            )[0]
        except (ConnectError, adbutils.AdbError, requests.RequestException) as e:
            raise ConnectError()
        except Exception as e:
            print("step failed due to action parsing error: ", e)
            tb = traceback.extract_tb(e.__traceback__)
            print('Error info:', tb)
            print('Error Response:', original_response)
            self.history_messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": original_response,
                        }
                    ]
                })

            self.step_counter += 1
            # if self.step_counter == self.max_steps:
            #     self.is_done = True
            self.is_done = True
            return {
                    'env_idx': self.worker_idx,
                    'obs_messages': self.history_messages,
                    'is_done': self.is_done,
                    'format_reward': -1.0
                    }


        print(f"Executing step with action: {action}")

        self.history_messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": original_response,
                    }
                ]
            })
        
        self.xml_history_messages.append({
            "role": "uiagent",
            "content": original_response
        })
        
        # check if it is the finish step
        is_done = (action.get('action_type') == 'finished')
        format_reward = 0.0
        execution = self._parse_action_to_command(action)

        if not execution:
            self.is_done = True
            return {
                'env_idx': self.worker_idx,
                'obs_messages': self.history_messages,
                'is_done': self.is_done,
                'format_reward': format_reward
            }

        try:
            new_screenshot = self.d.screenshot(format="pillow")
        except Exception as e:
            print("step failed due to ", e)
            tb = traceback.extract_tb(e.__traceback__)
            print('Error info:', tb)
            print('Error Response:', original_response)

            self.is_done = True
            return {
                'env_idx': self.worker_idx,
                'obs_messages': self.history_messages,
                'is_done': self.is_done,
                'format_reward': format_reward
            }
        self.history_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": new_screenshot,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                ]
            })
        
        obs, elem_loc = self._get_obs()
        self.xml_history_messages.append({
            "role": "mobile",
            "content": obs
        })
        
        self.is_done = is_done
        self.step_counter += 1
        if self.step_counter == self.max_steps:
            self.is_done = True

        return {
            'env_idx': self.worker_idx,
            'obs_messages': self.history_messages,
            'is_done': self.is_done,
            'format_reward': format_reward
        }


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


    def prepare_critic_logits_input(self, messages, step_idx):
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
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='right',
            )
        
        # for response part:
        role = messages[-1]['role']
        response = self.load_content(messages[-1]['content'])
        response = response + '<|im_end|>\n'
        enc_response = processor(None, [response], add_special_tokens=False, return_tensors="pt")
        response = pad_2d_list_to_length(enc_response['input_ids'], self.tokenizer.pad_token_id, max_length=self.config.data.max_response_length)

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
            max_length=self.config.data.max_prompt_length + self.config.data.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
            truncation='right',
        )
        position_ids = position_ids[:, : input_ids.size(0)]
        response_attention_mask = response_attention_mask[:, : self.config.data.max_response_length]
        response = response[:, : self.config.data.max_response_length][0]

        if self.use_prm:
            return_raw = (self._future_returns[step_idx] + self.outcome_reward) / self.outcome_ration
            step_return_label = torch.tensor(return_raw, dtype=torch.float32)

        else:
            step_return_label = torch.tensor(self.outcome_reward / self.outcome_ration * pow(self.lambdax, self.step_counter - step_idx - 1), dtype=torch.float32)
        masked_step_return_label = response_attention_mask[0].to(torch.float32) * step_return_label

        return input_ids, attention_mask, position_ids, response, masked_step_return_label, pixel_values, image_grid_thw

    def get_critic_train_dict(self):
        datas = []

        # Precompute future-looking discounted returns for PRM:
        # G(T-1) = r_p[T-1], G(t) = r_p[t] + λ * G(t+1)
        if self.use_prm and self.process_reward:
            self._future_returns = [0.0] * self.step_counter
            self._future_returns[self.step_counter - 1] = self.process_reward[self.step_counter - 1]
            for t in range(self.step_counter - 2, -1, -1):
                self._future_returns[t] = self.process_reward[t] + self.lambdax * self._future_returns[t + 1]

        for step_idx in range(self.step_counter):
            step_message = self.history_messages[:2*step_idx + 4]

            input_ids, attention_mask, position_ids, responses, masked_step_return_label, pixel_values, image_grid_thw = self.prepare_critic_logits_input(step_message, step_idx)
            
            data = {
                'raw_history_messages': step_message,
                'input_ids': input_ids,
                'responses': responses,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'env_idx': self.worker_idx,
                'step_return_label': masked_step_return_label,
                'task_id': self.task_id,
                'uid': self.task_id,
            }
            if pixel_values is not None:
                multi_modal_inputs = dict()
                multi_modal_inputs['pixel_values'] = pixel_values
                multi_modal_inputs['image_grid_thw'] = image_grid_thw
                data['multi_modal_inputs'] = multi_modal_inputs
            datas.append(data)
        return datas


    def prepare_policy_logits_input(self, messages, step_idx):
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
        for turn_idx, msg in enumerate(messages):
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
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='right',
            )
        

        return input_ids, attention_mask, position_ids, pixel_values, image_grid_thw

    def get_policy_train_dict(self):
        datas = []
        
        for step_idx in range(self.step_counter):
            step_message = self.history_messages[:2*step_idx + 3]

            data = {
                'env_idx': self.worker_idx,
                'obs_messages': step_message,
                'is_done': False,
                'format_reward': 0
            }
            
            datas.append(data)
        return datas

    def get_history_messages(self):
        return self.history_messages

    def get_step_counter(self):
        return self.step_counter

    def get_worker_idx(self):
        return self.worker_idx
    
    def get_ip_label(self):
        return self.ip_label

def get_completion_openai(prompt, model_name="gpt-4o", api_key=None):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=16384,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in get_completion_openai: {e}")
        return ""

def outcome_judger(task: str, xml_history: str):
    """
    Judge whether the task is completed based on the interaction trajectory.
    """
    try:
        print(f"Task: {task}\nModel's interaction trajectory with androidmobile: {xml_history}\n")

        detect_prompt = (
            "You are an expert evaluator for determining the success of GUI tasks. You will be provided with the following information:\n"
            "1. The task description.\n"
            "2. Mobile and UI Agent Interaction History including the step-by-step page state in compressed XML format and the agent's action for each step.\n\n"
            "**Scoring Rule:**\n"
            "You need to judge if the UI Agent completed the task based on the interaction trajectory. You should return True or False according to your judgement.\n\n"
            "**Output Format:**\n"
            "<analysis> [Your analysis] </analysis>\n"
            "<ans> [Your judgment] </ans>\n\n"
            "**Current Task Information:**\n"
            f"Task Description: {task}\n"
            f"Mobile and UI Agent Interaction History: {xml_history}"
        )
        load_dotenv()
        for attempt in range(2):
            if attempt > 0:
                sleep(6)  # retry delay

            return_message = get_completion_openai(
                prompt=detect_prompt,
                model_name="gpt-4o",
                api_key=os.environ.get("OPENAI_API_KEY")
            )

            if not return_message:
                continue

            # Parse the <ans> tag for robust extraction
            ans_match = re.search(r'<ans>\s*(.*?)\s*</ans>', return_message, re.IGNORECASE)
            if ans_match:
                ans_text = ans_match.group(1).strip().lower()
                if 'true' in ans_text:
                    return True
                elif 'false' in ans_text:
                    return False

            # Fallback: check full text
            if "True" in return_message:
                return True
            elif "False" in return_message:
                return False

        return False
        
    except Exception as e:
        print(f"Error in outcome_judger: {e}")
        return False

def get_completion_prm(messages):
    from openai import OpenAI
    client = OpenAI(
        base_url="http://127.0.0.1:5001/v1", # Host of the local VLLM server
        api_key="vllm-will-ignore-this-key" 
    )
    MODEL_NAME_ON_SERVER = "PRM_MODEL_PATH"  # Replace with the actual model name on your VLLM server
    print(f"--- Sending request to local VLLM server (Model: {MODEL_NAME_ON_SERVER}) ---")

    response = client.chat.completions.create(
        model=MODEL_NAME_ON_SERVER,
        messages=messages
    )
    return response.choices[0].message.content

def process_judger(task: str, action: str, base64_qwen):
    from verl.utils.dataset.utils import USER_PROMPT_WThought
    try:
        messages = [{"role": "system", "content": "Your are a helpful assistant."},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_qwen
                    },
                },
                {"type": "text", "text": USER_PROMPT_WThought.format(instruction=task, agent_thought_and_action=action)},
            ]}]
        raw_pr_result = get_completion_prm(messages)
        pr_result = parse_prediction(raw_pr_result)
        if pr_result:
            return 1
        return 0

    except Exception as e:
        print(f"Error in detect_answer: {e}")
        return 0
    
def parse_prediction(generated_text):
    text = str(generated_text).strip()

    if 'score:' in text.lower():
        score_part = text.lower().split('score:')[1].strip()
        if score_part.startswith('true'):
            return True
        elif score_part.startswith('false'):
            return False
        # Strict check: only match standalone "true"/"false" or exact "1"/"0"
        first_word = score_part.split()[0] if score_part.split() else ''
        if first_word in ('true', '1'):
            return True
        elif first_word in ('false', '0'):
            return False

    return None
