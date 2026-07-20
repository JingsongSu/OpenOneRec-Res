"""
Label Prediction Task Configuration

This is a classification task for predicting user engagement with video content.
Uses logprobs-based classification with AUC and wuAUC metrics.
"""

# Label Pred Task Configuration
LABEL_PRED_CONFIG = {
    "name": "label_pred",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 346190,
    "sample_size": 346190,
    "description": "Predict user engagement with video content (yes/no classification)",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {
        "enable_thinking": False,  # Enable thinking mode for apply_chat_template
        "custom_chat_template": "qwen3_soft_switch.jinja2",  # Custom jinja2 template (file in v1_0 directory)
    },
    "generation_config": {
        "max_new_tokens": 1,
        "temperature": 1,
        "top_p": 1,
        "top_k": -1,
        "do_sample": True,
        "num_return_sequences": 1,
        "return_logprobs": True,  # Need to return logprobs for probability extraction
        "logprobs": 10000,  # Return top-10 logprobs to ensure "是" and "否" are included
        "target_tokens": ["是", "否"],  # Target tokens for logprobs extraction (classification)
        "max_new_thinking_tokens": 1000
    },
    "evaluation_config": {
        "metrics": ["auc"],
    },
    "task_type": "logprobs_classification",  # Special task type
}

