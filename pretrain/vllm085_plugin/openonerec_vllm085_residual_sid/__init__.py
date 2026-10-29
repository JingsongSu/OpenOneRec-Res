# -*- coding: utf-8 -*-
"""OpenOneRec-Res vLLM 0.8.5 model plugin registration.

The original model.py is unchanged. We intentionally re-register the residual
SID architecture to point to constrained_model.py, even when this architecture
has already been registered in the process. vLLM 0.8.5 supports overwriting
existing model registration entries.
"""

RESIDUAL_ARCHITECTURE = "Qwen3ForResidualSIDPoolingV085"
AR_BASELINE_ARCHITECTURE = "Qwen3ForCausalLMIgnoreResidualSIDV085"


def register() -> None:
    from vllm import ModelRegistry

    # CRITICAL: no "if RESIDUAL_ARCHITECTURE not in supported" here.
    # That guard could silently keep the old unconstrained model registration.
    ModelRegistry.register_model(
        RESIDUAL_ARCHITECTURE,
        "openonerec_vllm085_residual_sid.constrained_model:"
        "Qwen3ForResidualSIDPoolingV085",
    )

    # Keep the existing autoregressive baseline unchanged.
    supported = set(ModelRegistry.get_supported_archs())
    if AR_BASELINE_ARCHITECTURE not in supported:
        ModelRegistry.register_model(
            AR_BASELINE_ARCHITECTURE,
            "openonerec_vllm085_residual_sid.model:"
            "Qwen3ForCausalLMIgnoreResidualSIDV085",
        )
