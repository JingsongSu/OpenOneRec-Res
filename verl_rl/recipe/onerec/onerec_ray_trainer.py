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
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
import wandb
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean, postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger


WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


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
        node_available_resources = ray.state.available_resources_per_node()
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
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

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
    tokenizer = None,
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
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
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


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
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
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
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

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"} and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy in {"fsdp", "fsdp2"}:
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, ground_truths=None):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        if ground_truths and len(ground_truths) == n:
            base_data["ground_truth"] = ground_truths

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

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Debug: print dataset sizes before validation
        print(f"[_validate] Starting validation. train_dataset size: {len(self.train_dataset)}, val_dataset size: {len(self.val_dataset)}")
        print(f"[_validate] actor_rollout_wg world_size: {self.actor_rollout_wg.world_size}")

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []
        sample_ground_truths = []

        batch_idx = 0
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            print(f"[Validation Debug] Batch {batch_idx}: test_batch size = {len(test_batch)}")
            batch_idx += 1

            # Check if beam search or two-stage rollout is enabled for validation
            val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
            rollout_config = self.config.actor_rollout_ref.rollout
            use_beam_search_val = val_kwargs.get("use_beam_search", False)
            is_two_stage_rollout_val = rollout_config.get("name") == "two_stage"

            # Only repeat if NOT using beam search (beam search will expand outputs internally)
            # For two-stage rollout, we DO repeat (for different CoT samples), beam expansion happens in rollout
            if not use_beam_search_val:
                # repeat test batch for sampling-based generation
                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
                )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs (will be expanded later if beam search returns all beams)
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            # Note: sample_inputs will be extended after beam search expansion handling

            if "reward_model" in test_batch.non_tensor_batch:
                ground_truths = [item["ground_truth"] for item in test_batch.non_tensor_batch["reward_model"]]
                # Note: ground_truths will be extended after beam search expansion handling

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            # Validation configuration
            val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
            rollout_config = self.config.actor_rollout_ref.rollout
            meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            # Check for Two-Stage Rollout in Validation
            if rollout_config.get("enable_two_stage_rollout", False):
                meta_info["enable_two_stage_rollout"] = True
                meta_info["stage2_beam_size"] = rollout_config.get("stage2_beam_size", 32)
                meta_info["stage2_max_tokens"] = rollout_config.get("stage2_max_tokens", 16)
                
                # Stage 1 CoT config
                meta_info["max_tokens"] = self.config.data.get("max_response_length", 1024)
                # Disable standard beam search for Stage 1 (use sampling)
                meta_info["use_beam_search"] = False
                meta_info["n"] = val_kwargs.get("n", 1)
                
                print(f"[OneRecTrainer] Validation Two-Stage Enabled: {meta_info}")

            # Inject Beam Search parameters if enabled for validation (Single Stage)
            elif val_kwargs.get("use_beam_search", False):
                meta_info["use_beam_search"] = True
                meta_info["best_of"] = val_kwargs.get("best_of", 4)
                # Use max_response_length from config for validation as well
                meta_info["max_tokens"] = self.config.data.get("max_response_length", 16)
                meta_info["temperature"] = 0
                # n controls how many beams to return per prompt (will expand output)
                meta_info["n"] = val_kwargs.get("n", 1)
                # Signal rollout to return all beams (no repeat, expand internally)
                meta_info["return_all_beams"] = True

                print(f"[OneRecTrainer] Validation Beam Search Enabled (optimized, no repeat): {meta_info}")

            test_gen_batch.meta_info = meta_info
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

            # unpad - For beam search or two-stage rollout, output is expanded, so we need to unpad accordingly
            if use_beam_search_val or is_two_stage_rollout_val:
                # For two-stage rollout, expansion is val_kwargs.n * stage2_beam_size
                if is_two_stage_rollout_val:
                    stage2_beam_size = rollout_config.get("stage2_beam_size", 2)
                    n_beams = stage2_beam_size  # rollout already expands by beam_width
                    print(f"[Validation Debug] Two-stage unpad: original pad_size={pad_size}, stage2_beam_size={stage2_beam_size}, actual_pad_size={pad_size * n_beams}")
                else:
                    n_beams = val_kwargs.get("n", 1)
                    print(f"[Validation Debug] Beam search unpad: original pad_size={pad_size}, n_beams={n_beams}, actual_pad_size={pad_size * n_beams}")
                actual_pad_size = pad_size * n_beams
            else:
                actual_pad_size = pad_size
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=actual_pad_size)

            # Debug: Check keys returned from worker
            print(f"[Trainer Debug] test_output_gen_batch keys: {test_output_gen_batch.non_tensor_batch.keys()}")

            print("validation generation end")

            # Handle beam search or two-stage rollout expansion: output may be larger than input
            # When return_all_beams=True, rollout expands output to batch_size * beam_width
            output_len = len(test_output_gen_batch)
            input_len = len(test_batch)
            if output_len > input_len and (use_beam_search_val or is_two_stage_rollout_val):
                # Rollout guarantees output_len = input_len * expand_factor, so we can use simple repeat
                expand_factor = output_len // input_len
                print(f"[Validation Debug] Batch {batch_idx-1}: Beam/TwoStage expansion - input={input_len}, output={output_len}, factor={expand_factor}")
                test_batch = test_batch.repeat(repeat_times=expand_factor, interleave=True)
                input_texts = [t for t in input_texts for _ in range(expand_factor)]
                if "reward_model" in test_batch.non_tensor_batch:
                    ground_truths = [t for t in ground_truths for _ in range(expand_factor)]
                print(f"[Validation Debug] Batch {batch_idx-1}: After expansion - len(input_texts)={len(input_texts)}, len(test_batch)={len(test_batch)}")

            # Now extend sample_inputs and sample_ground_truths
            before_extend = len(sample_inputs)
            sample_inputs.extend(input_texts)
            print(f"[Validation Debug] Batch {batch_idx-1}: Extended sample_inputs from {before_extend} to {len(sample_inputs)} (+{len(input_texts)})")
            if "reward_model" in test_batch.non_tensor_batch:
                sample_ground_truths.extend(ground_truths)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # Collect response lengths for validation metrics
            response_lengths = [(ids != self.tokenizer.pad_token_id).sum().item() for ids in output_ids]
            reward_extra_infos_dict["response_length"].extend(response_lengths)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            
            # Debug: Check keys after union
            print(f"[Trainer Debug] test_batch keys after union: {test_batch.non_tensor_batch.keys()}")

            # Critical Step: Move generated_items into extra_info for NaiveRewardManager
            if "generated_items" in test_batch.non_tensor_batch:
                print("[Trainer Debug] Moving generated_items into extra_info...")
                generated_items_arr = test_batch.non_tensor_batch["generated_items"]
                batch_size = len(generated_items_arr)
                
                # Ensure extra_info exists
                if "extra_info" not in test_batch.non_tensor_batch:
                    test_batch.non_tensor_batch["extra_info"] = np.array([{} for _ in range(batch_size)], dtype=object)
                
                extra_info_arr = test_batch.non_tensor_batch["extra_info"]
                for i in range(batch_size):
                    if extra_info_arr[i] is None: extra_info_arr[i] = {}
                    # Update dict (reference modification)
                    extra_info_arr[i]["generated_items"] = generated_items_arr[i]

            # evaluate using reward_function
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

            # 获取 data_source 信息，用于按task分组统计（和training逻辑一致）
            reward_fn_key = self.config.data.get("reward_fn_key", "data_source")
            data_sources_batch = test_batch.non_tensor_batch.get(reward_fn_key, None)

            # 如果没有找到，尝试其他常见字段名
            if data_sources_batch is None:
                data_sources_batch = test_batch.non_tensor_batch.get("source", None)
            if data_sources_batch is None:
                data_sources_batch = test_batch.non_tensor_batch.get("data_source", None)

            # 如果还是找不到，使用默认值
            if data_sources_batch is None:
                data_sources_batch = ["unknown"] * reward_tensor.shape[0]

            data_source_lst.append(data_sources_batch)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                ground_truths=sample_ground_truths,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        # Debug: Check for duplicate prompts
        from collections import Counter
        prompt_counts = Counter(sample_inputs)
        duplicate_prompts = {p: c for p, c in prompt_counts.items() if c > 1}
        if duplicate_prompts:
            print(f"[Validation Debug] Found {len(duplicate_prompts)} duplicate prompts!")
            for p, c in list(duplicate_prompts.items())[:3]:  # Show first 3
                print(f"  Prompt (truncated): '{p[:100]}...' appears {c} times")
        else:
            print(f"[Validation Debug] No duplicate prompts found. Total unique prompts: {len(prompt_counts)}")
        print(f"[Validation Debug] Total samples: {len(sample_inputs)}, Total scores: {len(sample_scores)}")

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best", "pass"])
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

        # Add validation response_length statistics
        if "response_length" in reward_extra_infos_dict:
            response_lengths = reward_extra_infos_dict["response_length"]
            if len(response_lengths) > 0:
                import torch
                response_lengths_tensor = torch.tensor(response_lengths)
                metric_dict["val/response_length/mean"] = response_lengths_tensor.float().mean().item()
                metric_dict["val/response_length/max"] = response_lengths_tensor.max().item()
                metric_dict["val/response_length/min"] = response_lengths_tensor.min().item()

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
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
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
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
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
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

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
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

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

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps

                # Get original batch size for beam_idx calculation
                original_bs = len(gen_batch)

                # Check if beam search is enabled - if so, don't repeat (optimization)
                # Two-stage rollout: still repeat (for different CoT samples), but beam expansion happens in rollout
                rollout_config = self.config.actor_rollout_ref.rollout
                use_beam_search_train = rollout_config.get("use_beam_search", False)
                is_two_stage_rollout = rollout_config.get("name") == "two_stage"
                rollout_n = self.config.actor_rollout_ref.rollout.n

                if not use_beam_search_train:
                    # Standard sampling or two-stage rollout: repeat the batch for n_rollout different samples
                    gen_batch = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                    if "reward_model" in batch.non_tensor_batch:
                        # repeat reward_model to match gen_batch size
                        repeated_reward_model = np.repeat(
                            batch.non_tensor_batch["reward_model"],
                            rollout_n,
                            axis=0
                        )
                        gen_batch.non_tensor_batch["reward_model"] = repeated_reward_model
                else:
                    print(f"[OneRecTrainer] Beam search enabled, skipping repeat (optimized path)")

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        # Dynamically configure generation parameters based on config
                        rollout_config = self.config.actor_rollout_ref.rollout
                        
                        # Check if beam search is enabled in config
                        if rollout_config.get("use_beam_search", False):
                            gen_batch.meta_info["use_beam_search"] = True
                            gen_batch.meta_info["best_of"] = rollout_config.get("best_of", 4)
                            # Use max_response_length from data config if available, otherwise default
                            gen_batch.meta_info["max_tokens"] = self.config.data.get("max_response_length", 16)
                            gen_batch.meta_info["temperature"] = 0
                            n = rollout_config.get("n", 1)
                            gen_batch.meta_info["n"] = n
                            # Optimized: return all beams from rollout, no repeat needed
                            gen_batch.meta_info["return_all_beams"] = True

                            print(f"[OneRecTrainer] Beam Search Enabled (optimized, no repeat): {gen_batch.meta_info}")
                        
                        # Check if Two-Stage Rollout is enabled
                        if rollout_config.get("enable_two_stage_rollout", False):
                            gen_batch.meta_info["enable_two_stage_rollout"] = True
                            gen_batch.meta_info["stage2_beam_size"] = rollout_config.get("stage2_beam_size", 32)
                            gen_batch.meta_info["stage2_max_tokens"] = rollout_config.get("stage2_max_tokens", 16)
                            # For Stage 1 (CoT), we use sampling params
                            gen_batch.meta_info["max_tokens"] = self.config.data.get("max_response_length", 1024) # CoT length
                            gen_batch.meta_info["temperature"] = rollout_config.get("temperature", 1.0)
                            gen_batch.meta_info["top_p"] = rollout_config.get("top_p", 1.0)
                            # Disable use_beam_search flag to prevent conflict in standard flow if both are set
                            gen_batch.meta_info["use_beam_search"] = False 
                            
                            print(f"[OneRecTrainer] Two-Stage Rollout Enabled: {gen_batch.meta_info}")

                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                        # Handle beam search/two-stage rollout expansion: output may be larger than input
                        # When return_all_beams=True, rollout expands output to batch_size * beam_width
                        if use_beam_search_train or is_two_stage_rollout:
                            output_len = len(gen_batch_output)
                            input_len = len(batch)
                            print(f"[OneRecTrainer] Beam/TwoStage: gen_batch_output size={output_len}, batch size={input_len}, n={rollout_n}")

                            # CRITICAL FIX: Generate UIDs BEFORE expansion so that beams from
                            # the same prompt share the same UID for correct GRPO grouping
                            # This must happen regardless of whether expansion is needed
                            batch.non_tensor_batch["uid"] = np.array(
                                [str(uuid.uuid4()) for _ in range(input_len)], dtype=object
                            )
                            print(f"[OneRecTrainer] Generated UIDs before expansion: {len(batch.non_tensor_batch['uid'])} unique UIDs")

                            if output_len > input_len:
                                # Rollout guarantees output_len = input_len * expand_factor
                                assert output_len % input_len == 0, \
                                    f"Output size {output_len} must be a multiple of input size {input_len}"
                                expand_factor = output_len // input_len
                                print(f"[OneRecTrainer] Expanding batch using repeat: factor={expand_factor}")

                                batch = batch.repeat(repeat_times=expand_factor, interleave=True)
                                print(f"[OneRecTrainer] After expansion: batch size={len(batch)}, UIDs will be repeated {expand_factor}x")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # Generate UIDs and repeat batch for standard sampling path
                    # Skip if beam search or two-stage rollout already handled this above
                    if not use_beam_search_train and not is_two_stage_rollout:
                        # Use original_bs (stored at line 1786) for consistency
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(original_bs)], dtype=object
                        )
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    
                    # FORCE INJECTION: Bypass union and inject directly into extra_info
                    if "generated_items" in gen_batch_output.non_tensor_batch:
                        print(f"[Trainer Fit Debug] Force injecting generated_items into extra_info...")
                        gen_items = gen_batch_output.non_tensor_batch["generated_items"]
                        
                        # Ensure extra_info exists in batch
                        if "extra_info" not in batch.non_tensor_batch:
                            batch.non_tensor_batch["extra_info"] = np.array([{} for _ in range(len(batch))], dtype=object)
                        
                        extra_infos = batch.non_tensor_batch["extra_info"]
                        
                        if len(gen_items) == len(extra_infos):
                            for i in range(len(gen_items)):
                                if extra_infos[i] is None: extra_infos[i] = {}
                                extra_infos[i]["generated_items"] = gen_items[i]
                        else:
                            print(f"[Trainer Fit Error] Batch size mismatch during injection: {len(gen_items)} vs {len(extra_infos)}")

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

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}

                        # per-position entropy plot
                        # masked_entropys = entropys * response_masks
                        # sum_entropy_per_position = torch.sum(masked_entropys, dim=0)
                        # num_tokens_per_position = torch.sum(response_masks, dim=0)
                        # mean_entropy_per_position = sum_entropy_per_position / torch.clamp(
                        #     num_tokens_per_position, min=1
                        # )
                        # try:
                        #     entropy_list = mean_entropy_per_position.cpu().tolist()
                        #     table_data = [[i, ent] for i, ent in enumerate(entropy_list)]
                        #     table = wandb.Table(data=table_data, columns=["position", "entropy"])
                        #     old_log_prob_metrics["actor/per_position_entropy_plot"] = wandb.plot.line(
                        #         table, "position", "entropy", title="Per-Position Entropy"
                        #     )
                        # except Exception as e:
                        #     print(f"Warning: Could not create wandb per-position entropy plot. Error: {e}")

                        # token-type entropy
                        try:
                            responses = batch.batch["responses"]
                            # mask for token type 1 (id >= 151669)
                            type1_mask = (responses >= 151669) * response_masks
                            # mask for token type 2 (id < 151669)
                            type2_mask = (responses < 151669) * response_masks

                            count_type1 = type1_mask.sum().item()
                            count_type2 = type2_mask.sum().item()

                            if count_type1 > 0:
                                entropy_type1 = masked_mean(entropys, mask=type1_mask, axis=None).item()
                                old_log_prob_metrics["actor/entropy_itemic_token"] = entropy_type1

                            if count_type2 > 0:
                                entropy_type2 = masked_mean(entropys, mask=type2_mask, axis=None).item()
                                old_log_prob_metrics["actor/entropy_lang_token"] = entropy_type2

                            old_log_prob_metrics["actor/token_count_itemic_token"] = count_type1
                            old_log_prob_metrics["actor/token_count_lang_token"] = count_type2
                        except Exception as e:
                            print(f"Warning: Could not compute token-type entropy metrics. Error: {e}")

                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # 获取 data_source 信息，用于按task分组统计
                            # 尝试多个可能的字段名：source, data_source, data_source_key
                            reward_fn_key = self.config.data.get("reward_fn_key", "data_source")
                            data_sources = batch.non_tensor_batch.get(reward_fn_key, None)

                            # 如果没有找到，尝试其他常见字段名
                            if data_sources is None:
                                data_sources = batch.non_tensor_batch.get("source", None)
                            if data_sources is None:
                                data_sources = batch.non_tensor_batch.get("data_source", None)

                            # 调试信息：打印可用的字段
                            if self.global_steps <= 2:  # 只在前几步打印
                                print(f"[DEBUG] Batch size: {len(batch)}")
                                print(f"[DEBUG] Available non_tensor_batch keys: {list(batch.non_tensor_batch.keys())}")
                                print(f"[DEBUG] reward_fn_key from config: {reward_fn_key}")
                                print(f"[DEBUG] data_sources found: {data_sources is not None}")
                                if data_sources is not None:
                                    print(f"[DEBUG] data_sources type: {type(data_sources)}, shape: {getattr(data_sources, 'shape', len(data_sources))}")
                                    print(f"[DEBUG] first 10 sources: {data_sources[:10] if len(data_sources) > 0 else []}")
                                    print(f"[DEBUG] unique sources: {np.unique(data_sources)}")

                            if data_sources is not None:
                                # 按 data_source 分组统计不同task的得分
                                unique_sources = np.unique(data_sources)
                                print(f"[Task Statistics] Found {len(unique_sources)} unique tasks: {unique_sources}")

                                for source in unique_sources:
                                    source_mask = data_sources == source
                                    num_samples = int(np.sum(source_mask))

                                    for key, values in reward_extra_infos_dict.items():
                                        if values and len(values) > 0:
                                            values_array = np.array(values)
                                            # 只记录数值类型的指标
                                            if np.issubdtype(values_array.dtype, np.number):
                                                source_values = values_array[source_mask]
                                                if len(source_values) > 0:
                                                    metrics[f"reward/{source}/{key}/mean"] = float(np.mean(source_values))
                                                    metrics[f"reward/{source}/{key}/max"] = float(np.max(source_values))
                                                    metrics[f"reward/{source}/{key}/min"] = float(np.min(source_values))
                                                    metrics[f"reward/{source}/{key}/count"] = num_samples
                            else:
                                print(f"[WARNING] data_sources not found in batch.non_tensor_batch. Available keys: {list(batch.non_tensor_batch.keys())}")

                            # 全局统计（所有task合并）
                            for key, values in reward_extra_infos_dict.items():
                                if values and len(values) > 0:
                                    values_array = np.array(values)
                                    # 只记录数值类型的指标
                                    if np.issubdtype(values_array.dtype, np.number):
                                        metrics[f"reward/all/{key}/mean"] = float(np.mean(values_array))
                                        metrics[f"reward/all/{key}/max"] = float(np.max(values_array))
                                        metrics[f"reward/all/{key}/min"] = float(np.min(values_array))

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # 🎯 添加基于GT PPL的think quality reward
                        enable_sid_ppl_reward = self.config.get("enable_sid_ppl_reward", False)
                        if enable_sid_ppl_reward:

                            
                            gt_ppl_reward_weight = self.config.get("sid_ppl_reward_weight", 0.1)
                            
                            # 1. 构造 probe data
                            probe_batch_dict, probe_non_tensor_dict, probe_mapping = construct_gt_probe_data(
                                batch=batch,
                                tokenizer=self.tokenizer
                            )
                            
                            if probe_batch_dict:
                                # 2. 构造 DataProto 并计算 log_prob
                                # 注意：我们需要将 probe data 放到 device 上
                                device = batch.batch["input_ids"].device
                                for k, v in probe_batch_dict.items():
                                    probe_batch_dict[k] = v.to(device)

                                # 将 dict 转换为 TensorDict
                                probe_batch_size = probe_batch_dict["input_ids"].shape[0]
                                probe_tensor_dict = TensorDict(probe_batch_dict, batch_size=probe_batch_size)

                                probe_batch = DataProto(
                                    batch=probe_tensor_dict,
                                    non_tensor_batch=probe_non_tensor_dict
                                )
                                
                                # 计算 log_prob
                                # compute_log_prob 返回的是 DataProto，其中 batch["old_log_probs"] 是 log_prob
                                probe_output = self.actor_rollout_wg.compute_log_prob(probe_batch)
                                probe_log_probs = probe_output.batch["old_log_probs"] # (num_probes, seq_len)
                                
                                # 3. 提取 GT tokens 的 log_prob 并计算 reward
                                # 我们需要聚合每个 original_idx 的最大 reward
                                original_idx_to_rewards = defaultdict(list)
                                original_idx_to_think_end = {}
                                
                                for i, mapping in enumerate(probe_mapping):
                                    original_idx = mapping["original_idx"]
                                    gt_len = mapping["gt_len"]
                                    think_end_idx = mapping["think_end_idx"]
                                    
                                    original_idx_to_think_end[original_idx] = think_end_idx
                                    
                                    # 提取最后 gt_len 个 token 的 log_prob
                                    # 注意：old_log_probs 对应的是 input_ids 的 log_prob
                                    # input_ids = [prompt, thought, </think>, GT]
                                    # 我们只关心 GT 部分
                                    gt_log_probs = probe_log_probs[i, -gt_len:]
                                    
                                    # 计算平均 log_prob (即 -PPL score)
                                    reward = gt_log_probs.mean().item()
                                    
                                    original_idx_to_rewards[original_idx].append(reward)
                                
                                # 4. 回填 reward
                                reward_added_count = 0
                                reward_sum = 0.0
                                max_reward_val = -float('inf')
                                min_reward_val = float('inf')
                                
                                for i, rewards in original_idx_to_rewards.items():
                                    # 取 max reward (最匹配的 GT)
                                    max_reward = max(rewards)
                                    
                                    think_end_idx = original_idx_to_think_end[i]
                                    
                                    # 确保索引不越界
                                    if think_end_idx < batch.batch["token_level_rewards"].shape[1]:
                                        # 加上权重
                                        weighted_reward = max_reward * gt_ppl_reward_weight
                                        batch.batch["token_level_rewards"][i, think_end_idx] += weighted_reward
                                        
                                        reward_added_count += 1
                                        reward_sum += max_reward
                                        max_reward_val = max(max_reward_val, max_reward)
                                        min_reward_val = min(min_reward_val, max_reward)
                                
                                # 记录 metrics
                                if reward_added_count > 0:
                                    metrics["gt_ppl_reward/mean"] = reward_sum / reward_added_count
                                    metrics["gt_ppl_reward/max"] = max_reward_val
                                    metrics["gt_ppl_reward/min"] = min_reward_val
                                    metrics["gt_ppl_reward/count"] = reward_added_count
                                    print(f"[Step {self.global_steps}] GT PPL Reward added to {reward_added_count} samples. Mean raw reward: {reward_sum / reward_added_count:.4f}")

                        # compute advantages, executed on the driver process

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
                            tokenizer=self.tokenizer,
                        )
                        
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
                            hit_rewards = batch.non_tensor_batch["score"]
                            if isinstance(hit_rewards, np.ndarray):
                                hit_rewards_tensor = torch.tensor(hit_rewards, dtype=torch.float32)
                            else:
                                hit_rewards_tensor = torch.tensor(list(hit_rewards), dtype=torch.float32)

                            # 根据uid分组
                            uids = batch.non_tensor_batch["uid"]
                            unique_uids = np.unique(uids)

                            zero_hit_reward_group_ratios = []
                            all_group_zero_count = 0  # 统计完全为0的group数量

                            for uid in unique_uids:
                                # 找到属于当前uid的所有样本
                                uid_mask = (uids == uid)
                                uid_hit_rewards = hit_rewards_tensor[uid_mask]

                                # 统计hit_reward为0的样本数量
                                zero_count = (uid_hit_rewards == 0).sum().item()
                                total_count = len(uid_hit_rewards)

                                # 计算当前group中hit_reward为0的比例
                                zero_ratio = zero_count / total_count if total_count > 0 else 0
                                zero_hit_reward_group_ratios.append(zero_ratio)

                                # 如果整个group的hit_reward都是0，计数加1
                                if zero_count == total_count:
                                    all_group_zero_count += 1

                            # 计算统计指标
                            if len(zero_hit_reward_group_ratios) > 0:
                                # 每个group中hit_reward为0的样本的平均比例
                                mean_zero_hit_reward_ratio_in_group = np.mean(zero_hit_reward_group_ratios)
                                # hit_reward完全为0的group占总group数的比例
                                all_zero_group_ratio = all_group_zero_count / len(unique_uids)

                                metrics["training/grpo_zero_hit_reward_ratio_in_group_mean"] = mean_zero_hit_reward_ratio_in_group
                                metrics["training/grpo_all_zero_hit_reward_group_ratio"] = all_zero_group_ratio
                                metrics["training/grpo_all_zero_hit_reward_group_count"] = all_group_zero_count
                                metrics["training/grpo_total_group_count"] = len(unique_uids)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            ground_truths = None
                            if "reward_model" in batch.non_tensor_batch:
                                ground_truths = [item["ground_truth"] for item in batch.non_tensor_batch["reward_model"]]
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                                ground_truths=ground_truths,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

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
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                train_data_metrics = compute_data_metrics(batch=batch, use_critic=self.use_critic)
                # Add train/ prefix to response_length metrics
                train_data_metrics_prefixed = {}
                for key, value in train_data_metrics.items():
                    if key.startswith("response_length/") or key.startswith("prompt_length/"):
                        train_data_metrics_prefixed[f"train/{key}"] = value
                    else:
                        train_data_metrics_prefixed[key] = value
                metrics.update(train_data_metrics_prefixed)
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
