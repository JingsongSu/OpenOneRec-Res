RESIDUAL_ARCHITECTURE = "Qwen3ForResidualSIDPoolingV085"
AR_BASELINE_ARCHITECTURE = "Qwen3ForCausalLMIgnoreResidualSIDV085"


def register() -> None:
    """Register OpenOneRec custom model architectures in vLLM."""
    from vllm import ModelRegistry

    supported = set(ModelRegistry.get_supported_archs())

    if RESIDUAL_ARCHITECTURE not in supported:
        ModelRegistry.register_model(
            RESIDUAL_ARCHITECTURE,
            (
                "openonerec_vllm085_residual_sid.model:"
                "Qwen3ForResidualSIDPoolingV085"
            ),
        )

    if AR_BASELINE_ARCHITECTURE not in supported:
        ModelRegistry.register_model(
            AR_BASELINE_ARCHITECTURE,
            (
                "openonerec_vllm085_residual_sid.model:"
                "Qwen3ForCausalLMIgnoreResidualSIDV085"
            ),
        )
