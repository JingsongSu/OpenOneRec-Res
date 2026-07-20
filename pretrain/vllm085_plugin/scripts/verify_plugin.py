import os
os.environ.setdefault("VLLM_USE_V1", "0")

from importlib.metadata import entry_points, version

from vllm import ModelRegistry


def main() -> None:
    points = list(entry_points(group="vllm.general_plugins"))
    matching = [
        point
        for point in points
        if point.name == "openonerec_residual_sid_v085"
    ]
    if not matching:
        raise RuntimeError(
            "The openonerec_residual_sid_v085 plugin is not installed."
        )

    matching[0].load()()
    required = {
        "Qwen3ForResidualSIDPoolingV085",
        "Qwen3ForCausalLMIgnoreResidualSIDV085",
    }
    supported = set(ModelRegistry.get_supported_archs())
    missing = required - supported
    if missing:
        raise RuntimeError(f"Missing architectures: {missing}")

    print(
        {
            "vllm_version": version("vllm"),
            "registered": sorted(required),
        }
    )


if __name__ == "__main__":
    main()
