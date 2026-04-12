# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

import uiautomator2 as u2
from .android import AndroidWorker
import re
from verl.utils.dataset.uiandroidlab_dataset import AndroidlabTaskConfigDataset, GUIDataset, collate_fn_dataproto, collate_fn
from verl.utils.dataset.uivalue_pretrain_dataset import PRMCriticDataset
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.ACLOO:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_values": data.batch["token_level_values"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data

import random

def expand_batch(process_results: list, multiple: int = 64) -> list:
    """
    Expand the process_results list to make its length a multiple of the specified value.

    Args:
        process_results (list): The list to be expanded.
        multiple (int): The multiple to which the list length should be expanded. Default is 4.

    Returns:
        list: The expanded list with length as a multiple of the specified value.
    """
    current_length = len(process_results)
    padding_size = (multiple - (current_length % multiple)) % multiple
    if padding_size > 0:
        padding_elements = random.choices(process_results, k=padding_size)
        process_results.extend(padding_elements)
    assert len(process_results) % multiple == 0, f"Length after padding is not a multiple of {multiple}: {len(process_results)}"

    return process_results

class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)
        
        self._create_dataloader()
        if not self.config.critic.pretrain_critic:
            self._create_envs()
            

    def _create_dataloader(self):
        """
        Creates the train and validation dataloaders.
        """
        from torch.utils.data import RandomSampler, SequentialSampler
        self.train_dataset = AndroidlabTaskConfigDataset(
            data_path=self.config.data.train_files,
        )

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

        self.val_dataset = AndroidlabTaskConfigDataset(
            data_path=self.config.data.val_files,
        )
        
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=min(self.config.env.num_envs, len(self.val_dataset)), # use the same number as envs
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1
        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")


        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        else:
            total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")
    
    def _create_pretrain_critic_dataloader(self):
        from torch.utils.data import RandomSampler, SequentialSampler
        self.pretrain_critic_dataset = PRMCriticDataset(
            csv_file = self.config.data.critic_pretrain_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            truncation="right",
            max_pixels=self.config.data.max_pixels,
            min_pixels=self.config.data.min_pixels,
        )
        
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.pretrain_critic_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.pretrain_critic_dataset)

        batch_size = self.config.data.pretrain_critic_batch_size
        if len(self.pretrain_critic_dataset) < self.config.data.pretrain_critic_batch_size:
            batch_size = 8
            print(f"example training")
        self.pretrain_critic_dataloader = StatefulDataLoader(
            dataset=self.pretrain_critic_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )
        print(f"Size of pretrain critic dataloader: {len(self.pretrain_critic_dataloader)}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch
    
    # def _get_gen_state_batch(self, batch: DataProto) -> DataProto:
    #     reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

    #     # pop those keys for generation
    #     batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    #     non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
    #     gen_batch = batch.pop(
    #         batch_keys=batch_keys_to_pop,
    #         non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
    #     )

    #     # For agent loop, we need reward model keys to compute score.
    #     if self.async_rollout_mode:
    #         gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

    #     return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )
        
        self._create_envs()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        
    def renew_env(self, worker):
        max_steps = self.config.env.max_steps
        ip_label = ray.get(worker.get_ip_label.remote())
        worker_id = ray.get(worker.get_worker_idx.remote())
        print(f"offline ip:port {ip_label}")
        print(f"renew env: {worker_id}")
        if len(self.ip_labels['available']) == 0:
            raise RuntimeError("Not enough available IPs!")
        new_ip_label = self.ip_labels['available'][0]
        self.ip_labels['used'].append(new_ip_label)
        w = AndroidWorker.options(
                resources={ new_ip_label: 1 },   # make sure the new ip is available
                name=f"env_worker_{worker_id}"
            ).remote(worker_idx=worker_id, ip_label=new_ip_label, max_steps=self.config.env.max_steps, config=self.config)
        print(f"ip pool info updated: {self.ip_labels}")
        return w
    
    def _create_envs(self) -> None:
        """
        Create env workers and data-processor workers, 
        and pin each EnvWorker to a different node (round-robin).
        """
        print('Start to create env_worker for Android Environment')
        max_steps = self.config.env.max_steps
        num_envs = self.config.env.num_envs

        config = {
            # "avd_name": "Pixel_7_Pro_API_33",
            "max_steps": max_steps,
        }

        # 1) pick custom IP resource labels from cluster_resources
        #   cluster_resources() also contains built-in resources like "CPU"/"GPU"/"memory", we need to filter them out
        all_res = ray.cluster_resources().keys()
        # ip_labels = [r for r in all_res if re.match(r"^\d+\.\d+\.\d+\.\d+$", r)]\
        self.ip_labels = {
            "available": [r for r in all_res if re.match(r"^docker:\d+\.\d+\.\d+\.\d+$", r)],
            "used": [],
            "offline": []
        }
        if not self.ip_labels["available"]:
            raise RuntimeError("No available IP resource labels found, please check the --resources parameter when starting ray")

        # 2) Pin each env worker to different nodes
        self.env_workers = []
        for i in range(num_envs):
            ip_label = self.ip_labels["available"][i % len(self.ip_labels["available"])]
            udid = f"emulator-{5554 + i * 2}"  # example udid
            w = AndroidWorker.options(
                    resources={ ip_label: 1 },   # make sure this actor is scheduled to a node with the ip_label resource
                    name=f"env_worker_{i}"
                ).remote(worker_idx=i, ip_label=ip_label, max_steps=self.config.env.max_steps, config=self.config)
            self.env_workers.append(w)

        print(f'Env_worker for Android Environment created!  total: {len(self.env_workers)}, all_res:{all_res}')
            
    def start_reset_envs(self, batch_dict):
        # rollout_task_repeat_n = self.config.data.rollout_n

        task_configs = [x for x in batch_dict] # interleave
        assert len(task_configs) == len(self.env_workers), f"Expected {len(self.env_workers)} env workers, but got {len(task_configs)} task configs."
        # reset_envs_object = [worker.reset.remote(task_config) for worker, task_config in zip(self.env_workers, task_configs)]
        
        reset_envs_object = []
        future_worker_map = {}
        worker_id_futures = [worker.get_worker_idx.remote() for worker in self.env_workers]
        worker_ids = ray.get(worker_id_futures)

        for worker, task_config, worker_id in zip(self.env_workers, task_configs, worker_ids):
            messages = worker.reset.remote(task=task_config['task_name'], task_id=task_config['task_id'])
            reset_envs_object.append(messages)
            future_worker_map[messages] = {
                'worker': worker,
                'worker_id': worker_id
            }
        return task_configs, reset_envs_object, future_worker_map

    def prepare_vllm_inputs_full(self, env_outputs, fast_rollout = True):
        # NOTE: processor will be very slow

        valid_obs_messages = [x['obs_messages'] for x in env_outputs if x['is_done'] is False]
        valid_env_idx = [x['env_idx'] for x in env_outputs if x['is_done'] is False]

        dataset = GUIDataset(
            valid_obs_messages,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            max_pixels=self.config.data.max_pixels,
            min_pixels=self.config.data.min_pixels,
            fast_rollout=fast_rollout,
        )

        def get_dataset_item(index):
            return dataset[index]

        with ThreadPoolExecutor(max_workers=64) as executor:
            batch_dict = list(executor.map(get_dataset_item, range(len(dataset))))

        batch_dict = collate_fn_dataproto(batch_dict)
        batch = DataProto.from_single_dict(batch_dict)
        
        return batch, valid_env_idx
    
    def prepare_training_batch(self, batch: DataProto) -> DataProto:
        def nearest_power_of_two_leq(n: int) -> int:
            import math
            if n <= 0:
                return 0
            return 2 ** int(math.floor(math.log2(n)))
        batch_size = len(batch)
        num_to_take = nearest_power_of_two_leq(batch_size)
        if num_to_take == 0:
            return batch
        pos_batch = batch[:num_to_take]

        return pos_batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            if self.config.critic.pretrain_critic:
                self._create_pretrain_critic_dataloader()
                for pretrain_critic_batch_dict in self.pretrain_critic_dataloader:
                    pretrain_critic_batch = collate_fn_dataproto(pretrain_critic_batch_dict) #process_results理论上是和prepare_critic_batch输出一致的list
                    pretrain_critic_batch = DataProto.from_single_dict(pretrain_critic_batch)

                    pretrain_critic_batch.batch["response_mask"] = pretrain_critic_batch.batch["responses"] != -100
                    pretrain_critic_batch.meta_info["global_token_num"] = torch.sum(pretrain_critic_batch.batch["attention_mask"], dim=-1).tolist()

                    values = self.critic_wg.compute_values(pretrain_critic_batch)
                    pretrain_critic_batch = pretrain_critic_batch.union(values)
                    critic_output = self.critic_wg.update_critic(pretrain_critic_batch)
            
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                
                with marked_timer("step", timing_raw):
                    
                    task_configs, reset_envs_object, reset_worker_map = self.start_reset_envs(batch_dict)
                    assert len(task_configs) == len(self.env_workers)
                    eval_results_objects = [None] * len(task_configs)
                    eval_worker_map = {}
                    print(f"task_num: {len(task_configs)}, env_num: {len(self.env_workers)}")
                    pending_reset_futures = reset_envs_object
                    with marked_timer("env_reset", timing_raw):
                        # reset_outputs = ray.get(reset_envs_object)
                        reset_outputs = []
                        workers_to_retry = []
                        while pending_reset_futures:
                            ready_reset_futures, pending_reset_futures = ray.wait(pending_reset_futures, num_returns=1, timeout=60)
                            if not ready_reset_futures:
                                print("reset waiting timeout, no task finished.")
                                continue
                            
                            ready_reset_future = ready_reset_futures[0]
                            try:
                                reset_result = ray.get(ready_reset_future)
                                reset_outputs.append(reset_result)
                            except:
                                print("Connection error during reset")
                                workers_to_retry.append(reset_worker_map[ready_reset_future])
                                print("workers_to_retry: ", workers_to_retry)
                        
                        if workers_to_retry:
                            for worker_info in workers_to_retry:
                                worker = worker_info['worker']
                                worker_id = worker_info['worker_id']
                                new_worker = self.renew_env(worker)
                                message = ray.get(new_worker.reset.remote(batch_dict[worker_id]))
                                self.env_workers[worker_id] = new_worker
                                reset_outputs.append(message)
                        
                    print(f"reset_time: {timing_raw['env_reset']}")

                    env_outputs = reset_outputs
                    for step_idx in range(self.config.env.max_steps):
                        is_done_stats = ray.get([worker.is_done.remote() for worker in self.env_workers])
                        print(f'step_idx: {step_idx}, finished: {sum(is_done_stats)}')

                        num_workers = len(self.actor_rollout_wg._workers)
                        with marked_timer("prepare_vllm_inputs", timing_raw):
                            vllm_batch, valid_env_idx = self.prepare_vllm_inputs_full(env_outputs)

                        print('prepare_vllm_inputs_time: ', timing_raw['prepare_vllm_inputs'])
                        vllm_batch_pad, pad_size = pad_dataproto_to_divisor(vllm_batch, num_workers)

                        gen_batch = vllm_batch_pad.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                        )

                        # predict actions
                        with marked_timer("actor_rollout_wg", timing_raw):
                            action_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        print('action_batch_output_time: ', timing_raw['actor_rollout_wg'])
                        action_batch_output = unpad_dataproto(action_batch_output, pad_size=pad_size)

                        response_texts = self.tokenizer.batch_decode(action_batch_output.batch['responses'], skip_special_tokens=True)

                        cur_valid_envs = [self.env_workers[i] for i in valid_env_idx]
                        print("valid_env_idx: ", valid_env_idx)
                        print("cur_valid_envs: ", cur_valid_envs)
                        env_outputs = []
                        with marked_timer("env_step", timing_raw):
                            # futures = [
                            #     worker.step.remote(action_text) for worker, action_text in zip(cur_valid_envs, response_texts)
                            # ]
                            futures = []
                            future_worker_map = {}
                            workers_to_retry = []
                            cur_valid_envs_index = 0
                            for worker, action_text in zip(cur_valid_envs, response_texts):
                                worker_history = ray.get(worker.get_history_messages.remote())
                                worker_id = ray.get(worker.get_worker_idx.remote())
                                future_obj = worker.step.remote(action_text)
                                futures.append(future_obj)
                                future_worker_map[future_obj] = {
                                    'worker': worker,
                                    'worker_id': worker_id,
                                    'worker_history': worker_history,
                                    'cur_valid_envs_index': cur_valid_envs_index,
                                    'action_text': action_text
                                }
                                cur_valid_envs_index += 1
                            pending_futures = futures
                            while pending_futures:
                                # wait for at least one future to complete, timeout can be set as needed
                                ready_futures, pending_futures = ray.wait(pending_futures, num_returns=1, timeout=10)
                                
                                if not ready_futures:
                                    print("step waiting timeout, no task finished.")
                                    continue
                                
                                ready_future = ready_futures[0]
                                try:
                                    result = ray.get(ready_future)
                                    env_outputs.append(result)
                                except u2.exceptions.ConnectError as e:
                                    print("Connection error during step function")
                                    workers_to_retry.append(future_worker_map[ready_future])
                                    print("workers_to_retry: ", workers_to_retry)
                            if workers_to_retry:
                                futures = []
                                for worker_info in workers_to_retry:
                                    worker = worker_info['worker']
                                    worker_id = worker_info['worker_id']
                                    worker_history = worker_info['worker_history']
                                    cur_valid_envs_index = worker_info['cur_valid_envs_index']
                                    action_text = worker_info['action_text']
                                    new_worker = self.renew_env(worker)
                                    _ = ray.get(new_worker.reset.remote(batch_dict[worker_id]))
                                    new_worker.step_counter = ray.get(worker.get_step_counter.remote())
                                    self.env_workers[worker_id] = new_worker
                                    cur_valid_envs[cur_valid_envs_index] = new_worker
                                    # recover history trajectory
                                    ray.get(cur_valid_envs[cur_valid_envs_index].recover_step.remote(worker_history))
                                    futures.append(cur_valid_envs[cur_valid_envs_index].step.remote(action_text))
                                env_outputs.extend(ray.get(futures))
                                    # print("env_outputs: ", env_outputs)
                                    # env_outputs = ray.get(futures)
                        print('env_step_time: ', timing_raw['env_step'])
                        # get format rewards
                        for single_output in env_outputs:
                            if single_output['is_done']:
                                cur_env_idx = single_output['env_idx']
                                # start evaluate, do not evaluate in the end together
                                # eval_results_objects[cur_env_idx] = self.env_workers[cur_env_idx].evaluate.remote()
                                eval_res_obj = self.env_workers[cur_env_idx].evaluate.remote()
                                eval_results_objects[cur_env_idx] = eval_res_obj
                                eval_worker_map[eval_res_obj] = {
                                    'worker': self.env_workers[cur_env_idx],
                                    'worker_id': ray.get(self.env_workers[cur_env_idx].get_worker_idx.remote()),
                                    'worker_history': ray.get(self.env_workers[cur_env_idx].get_history_messages.remote()),
                                }

                        is_last_step = self.global_steps >= self.total_training_steps
                        is_all_done = all([x['is_done'] for x in env_outputs])
                        if is_all_done:
                            break
                        
                    with marked_timer("evaluate_env", timing_raw):
                        # eval_results = ray.get(eval_results_objects)
                        # eval_results = eval_results_objects
                        eval_results = []
                        workers_to_retry = []
                        pending_eval_futures = eval_results_objects
                        while pending_eval_futures:
                            ready_eval_futures, pending_eval_futures = ray.wait(pending_eval_futures, num_returns=1, timeout=10)
                            if not ready_reset_futures:
                                print("eval waiting timeout, no task finished.")
                                continue
                            
                            ready_eval_future = ready_eval_futures[0]
                        
                            eval_result = ray.get(ready_eval_future)
                            eval_results.append(eval_result)

                    print('evaluate_env_time: ', timing_raw['evaluate_env'])
                    
                    with marked_timer("acloo training", timing_raw):
                        with marked_timer("prepare Q batch", timing_raw):
                            process_results_raw = ray.get([worker.get_critic_train_dict.remote() for worker in self.env_workers])
                            process_results = [item for sublist in process_results_raw for item in sublist]
                            process_results = expand_batch(process_results, multiple = 16 * self.config.data.rollout_batch_size)
                            batch = collate_fn_dataproto(process_results)
                            batch = DataProto.from_single_dict(batch)
                            
                            # batch.batch["responses"] = batch.batch["input_ids"]
                            batch.batch["response_mask"] = batch.batch["responses"] != -100
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        
                        
                        with marked_timer("critic training", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                        
                        if self.config.trainer.critic_warmup >= self.global_steps:
                            continue  # skip actor update until critic is warmed up online
                        
                        with marked_timer("supplement rollout", timing_raw):
                            process_results_raw = ray.get([worker.get_policy_train_dict.remote() for worker in self.env_workers])
                            process_results = [item for sublist in process_results_raw for item in sublist]
                            process_results = expand_batch(process_results, multiple = 4)
                            with marked_timer("prepare_vllm_inputs", timing_raw):
                                vllm_batch, valid_env_idx = self.prepare_vllm_inputs_full(process_results, fast_rollout=False)

                            print('prepare_vllm_inputs_time: ', timing_raw['prepare_vllm_inputs'])
                            batch, pad_size = pad_dataproto_to_divisor(vllm_batch, num_workers)
                            
                            batch.non_tensor_batch["uid"] = np.array(
                                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                            )

                            gen_batch = batch.pop(
                                batch_keys=["input_ids", "attention_mask", "position_ids"],
                                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                            )

                            gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            # repeat to align with repeated responses in rollout
                            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            batch = batch.union(gen_batch_output)
                            if "response_mask" not in batch.batch.keys():
                                batch.batch["response_mask"] = compute_response_mask(batch)
                                
                            # Balance the number of valid tokens across DP ranks.
                            # NOTE: This usually changes the order of data in the `batch`,
                            # which won't affect the advantage calculation (since it's based on uid),
                            # but might affect the loss calculation (due to the change of mini-batching).
                            # TODO: Decouple the DP balancing and mini-batching.
                            if self.config.trainer.balance_batch:
                                self._balance_batch(batch, metrics=metrics)
                                
                            # compute global_valid tokens
                            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        
                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                        # value assign: Q give values
                        # compute values
                        with marked_timer("calculate Q", timing_raw):
                            if self.use_critic:
                                with marked_timer("values", timing_raw, color="cyan"):
                                    values = self.critic_wg.compute_values(batch)
                                    batch = batch.union(values)
                        
                        with marked_timer("calculate adv", timing_raw):
                            batch.batch["token_level_values"] = batch.batch["values"]
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )
                        
                        with marked_timer("prepare training batch", timing_raw):
                            batch = self.prepare_training_batch(batch)
                            with marked_timer("update_actor", timing_raw, color="red"):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)
                        

                        # # Log rollout generations if enabled
                        # rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        # if rollout_data_dir:
                        #     with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                        #         inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                        #         outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                        #         scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                        #         sample_gts = [
                        #             item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                        #             for item in batch
                        #         ]

                        #         if "request_id" in batch.non_tensor_batch:
                        #             reward_extra_infos_dict.setdefault(
                        #                 "request_id",
                        #                 batch.non_tensor_batch["request_id"].tolist(),
                        #             )

                        #         self._dump_generations(
                        #             inputs=inputs,
                        #             outputs=outputs,
                        #             gts=sample_gts,
                        #             scores=scores,
                        #             reward_extra_infos_dict=reward_extra_infos_dict,
                        #             dump_path=rollout_data_dir,
                        #         )

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                # if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                #     self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
