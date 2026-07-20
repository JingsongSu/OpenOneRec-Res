
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from recipe.onerec.onerec_vllm_rollout import OneRecvLLMRollout
from verl.utils.fs import copy_to_local
from torch.distributed.device_mesh import init_device_mesh
from verl.utils.device import get_device_name
import logging
import torch

logger = logging.getLogger(__name__)

class OneRecActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    Custom ActorRolloutRefWorker that uses OneRecvLLMRollout instead of standard vLLMRollout.
    """
    def _build_rollout(self, trust_remote_code=False):
        # We only override the two_stage rollout path
        if self.config.rollout.name == "two_stage":
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
            from verl.utils.profiler import log_gpu_memory_usage

            # Original logic from ActorRolloutRefWorker._build_rollout
            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            assert self.world_size % infer_tp == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            )
            device_name = get_device_name()
            rollout_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
            )

            log_gpu_memory_usage(f"Before building vllm rollout (OneRec Custom)", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
            lora_kwargs = (
                {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}}
                if self._is_lora
                else {}
            )
            
            # Use our custom class!
            # We check for async mode but currently only support Sync OneRecvLLMRollout
            if self.config.rollout.mode == "async":
                 logger.warning("OneRecvLLMRollout currently only supports SYNC mode fully. Async might fallback or fail if logic differs.")
                 # If you implemented AsyncOneRecvLLMRollout, use it here.
                 # For now, we assume sync mode or that OneRecvLLMRollout works for both structure wise 
                 # (vLLMAsyncRollout inherits from different base, so simple substitution might fail for async)
                 # Fallback to original for async if you haven't implemented Async wrapper
                 return super()._build_rollout(trust_remote_code)

            rollout = OneRecvLLMRollout(
                model_path=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=trust_remote_code,
                **lora_kwargs,
            )

            log_gpu_memory_usage(f"After building vllm rollout (OneRec Custom)", logger=logger)
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                rollout_config=self.config.rollout,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)
            
            return rollout, rollout_sharding_manager
        
        else:
            # Fallback to parent implementation for other backends
            return super()._build_rollout(trust_remote_code)
