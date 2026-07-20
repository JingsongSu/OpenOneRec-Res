"""
Recommendation Reason Task Configuration
"""

# Recommendation Reason Task Configuration
REC_REASON_CONFIG = {
    "name": "rec_reason",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 470,
    "sample_size": 470,
    "description": "Recommendation reason inference",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {
        "enable_thinking": True,  # Enable thinking mode for apply_chat_template
        "custom_chat_template": "qwen3_soft_switch.jinja2",  # Custom jinja2 template (file in v1_0 directory)
    },
    "generation_config": {
        "num_return_sequences": 1,
        "max_new_tokens": 2000,
        "temperature": 0.01,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "do_sample": False,
        "num_return_thinking_sequences": 1,
        "max_new_thinking_tokens": 10000,
    },
    "evaluation_config": {
        "metrics": ["avg_score"],
        # LLM multi-dimensional evaluation config
        "llm_eval_enabled": True,                  # Whether to enable LLM evaluation
        "llm_judge_model": "gemini",               # Judge LLM type: gemini/deepseek/claude
        "llm_max_workers": 1,                      # Concurrent workers for LLM calls
        "llm_max_samples": 470,                    # Max samples to evaluate (None for all)
    }
}

