"""
Item Understand Task Configuration
"""

# Item Understand Task Configuration
ITEM_UNDERSTAND_CONFIG = {
    "name": "item_understand",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 500,
    "sample_size": 500,
    "description": "Video SID to Caption generation task",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {
        "enable_thinking": False,  # Enable thinking mode for apply_chat_template
        "custom_chat_template": "qwen3_soft_switch.jinja2",  # Custom jinja2 template (file in v1_0 directory)
    },
    # Generation parameter configuration
    "generation_config": {
        "num_return_sequences": 1,
        "max_new_tokens": 128,
        "temperature": 0.01,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "do_sample": False,
        "num_return_thinking_sequences": 1,
        "max_new_thinking_tokens": 1000,
    },
    "evaluation_config": {
        "metrics": ["macro_wip_double_weighted_f1", "micro_wip_double_weighted_f1"],
        "bertscore_model_type": "bert-base-chinese",
        "bertscore_num_layers": 9,
        "bertscore_lang": "zh",
        # WIP (Weighted Information Points) evaluation config
        "wip_enabled": True,                      # Whether to enable WIP evaluation
        "wip_judge_model": "gemini",             # Judge LLM type: gemini/deepseek/claude
        "wip_max_workers": 1,                      # Concurrent workers for LLM calls
        "wip_core_threshold": 5,                   # Core threshold for importance score (1-5)
        "wip_max_samples": 500,                    # Max samples to evaluate (None for all)
    }
}

