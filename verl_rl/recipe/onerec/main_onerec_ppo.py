# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
OneRec custom main entry point for PPO training using custom onerec_ray_trainer.
"""

import os
import sys

# Add project root to path to ensure imports work correctly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hydra
import ray
from omegaconf import OmegaConf

# Import the custom trainer from onerec_ray_trainer.py
from recipe.onerec.onerec_ray_trainer import RayPPOTrainer

# Import other necessary components from verl
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import TaskRunner as BaseTaskRunner, create_rl_dataset, create_rl_sampler
from verl.utils.device import is_cuda_available


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for OneRec PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


def run_ppo(config) -> None:
    """Run PPO training process with OneRec custom trainer.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class
    if (
        is_cuda_available
        and config.trainer.get("profile_steps") is not None
        and len(config.trainer.get("profile_steps", [])) > 0
    ):
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = OneRecTaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = OneRecTaskRunner.remote()
    ray.get(runner.run.remote(config))

    # Optional: get the path of the timeline trace file from the configuration
    timeline_json_file = config.trainer.get("ray_timeline_filename", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class OneRecTaskRunner:
    """Ray remote class for executing distributed OneRec PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    Uses the custom onerec_ray_trainer.RayPPOTrainer instead of the default trainer.
    """

    def run(self, config):
        """Execute the main PPO training workflow with OneRec custom trainer.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        import socket
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.trainer.ppo.reward import load_reward_manager
        from verl.utils.fs import copy_to_local
        from verl.utils.import_utils import load_extern_type
        
        # Import Role and ResourcePoolManager from the custom onerec_ray_trainer
        # to ensure we use the same Role enum
        from recipe.onerec.onerec_ray_trainer import ResourcePoolManager, Role

        print(f"OneRecTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        print("=" * 80)
        print("Using Custom OneRec RayPPOTrainer from recipe/onerec/onerec_ray_trainer.py")
        print("=" * 80)
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Define worker classes based on the actor strategy
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            # Use custom OneRecActorRolloutRefWorker instead of standard ActorRolloutRefWorker
            from recipe.onerec.onerec_fsdp_workers import OneRecActorRolloutRefWorker as ActorRolloutRefWorker
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker
                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError(f"Unknown strategy: {config.actor_rollout_ref.actor.strategy}")

        # Load reward model worker if enabled
        if config.reward_model.get("enable", False):
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError(f"Unknown reward model strategy: {config.reward_model.strategy}")
        else:
            RewardModelWorker = None

        # Setup resource pool configuration
        n_gpus_per_node = config.trainer.n_gpus_per_node
        nnodes = config.trainer.nnodes
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [n_gpus_per_node] * nnodes}

        # Map roles to workers
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        if config.critic.get("enable", True):
            role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
            mapping[Role.Critic] = global_pool_id

        if config.reward_model.get("enable", False):
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            mapping[Role.RefPolicy] = global_pool_id

        # Load reward managers
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # ========================================================================
        # KEY CHANGE: Use the custom OneRec RayPPOTrainer instead of default
        # ========================================================================
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )

        # Initialize the workers of the trainer
        trainer.init_workers()
        # Start the training process
        trainer.fit()


if __name__ == "__main__":
    main()

