# Unified LLM API Wrapper

This is a unified LLM API wrapper library that provides a clean and elegant interface for calling different large language models.

## Supported Models

- **Claude** - Anthropic Claude models
- **Gemini** - Google Vertex AI Gemini models
- **DeepSeek** - DeepSeek models via Baidu Qianfan platform

## Model Pricing Comparison 

- Claude: https://claude.com/pricing
- Gemini: https://ai.google.dev/gemini-api/docs/pricing
- DeepSeek: https://api-docs.deepseek.com/quick_start/pricing


## Quick Start

### Installation

```bash
pip install openai google-cloud-aiplatform anthropic tqdm
```

### Using Configuration File

First, edit `api/config/llm_config.json` to fill in your configuration:

Then use the following code to test:

```python
from api import get_client_from_config

# Create client
client = get_client_from_config("gemini")

# Generate text
response = client.generate("Tell me a joke")
print(response)
```

