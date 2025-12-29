# Supported Providers

Meridian integrates with 7 provider families, offering 17+ model configurations out of the box.

> [!NOTE]
> Pricing estimates as of December 2025. Check provider documentation for current rates.

---

## Provider Matrix

| Provider | Models | Logprobs | Streaming | Tools | Cost Range |
|----------|--------|:--------:|:---------:|:-----:|------------|
| **DeepSeek** | `deepseek_chat`, `deepseek_coder` | ✅ | ✅ | ✅ | $0.07–0.28/M tokens |
| **OpenAI** | `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo` | ✅ | ✅ | ✅ | $0.50–10/M tokens |
| **Anthropic** | `claude-2`, `claude-3-*` | ❌ | ✅ | ✅ | $3–15/M tokens |
| **Mistral** | `small`, `medium`, `large` | ✅ | ✅ | ✅ | $0.25–2/M tokens |
| **Groq** | `llama-70b`, `llama-8b`, `mixtral` | ❌ | ✅ | ✅ | Free tier available |
| **Together** | `llama-70b`, `mixtral`, `codellama` | ✅ | ✅ | ✅ | $0.20–0.90/M tokens |
| **Local** | `distilgpt2`, `flan-t5-small` | ✅ | N/A | N/A | Free (your hardware) |

---

## Provider Details

### DeepSeek

**Recommended for**: Cost-effective evaluation, strong reasoning

```bash
# Setup
export DEEPSEEK_API_KEY=your_key

# Usage
python -m meridian.cli run --suite rag_evaluation --model deepseek_chat
```

| Model | Context | Best For |
|-------|---------|----------|
| `deepseek_chat` | 32K | General evaluation |
| `deepseek_coder` | 16K | Code analysis suites |

**SDK**: Built-in (OpenAI-compatible API)

---

### OpenAI

**Recommended for**: Baseline comparisons, maximum capability

```bash
export OPENAI_API_KEY=sk-...

python -m meridian.cli run --suite business_analysis --model openai_gpt4
```

| Model | Meridian ID | Notes |
|-------|-------------|-------|
| GPT-3.5 Turbo | `openai_gpt35` | Fast, cost-effective |
| GPT-4 | `openai_gpt4` | Best accuracy |
| GPT-4 Turbo | `openai_gpt4_turbo` | 128K context |

**SDK**: `openai>=1.0.0` (included in dependencies)

---

### Anthropic

**Recommended for**: Long-context evaluation, safety testing

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python -m meridian.cli run --suite document_processing --model anthropic_claude_2
```

| Model | Meridian ID | Notes |
|-------|-------------|-------|
| Claude 2 | `anthropic_claude_2` | 100K context |

**SDK**: `anthropic>=0.7.0` (included in dependencies)

**Limitation**: Logprobs not available via API

---

### Mistral

**Recommended for**: European hosting, balanced cost/performance

```bash
export MISTRAL_API_KEY=your_key

python -m meridian.cli run --suite code_analysis --model mistral_large
```

| Model | Meridian ID | Best For |
|-------|-------------|----------|
| Mistral Small | `mistral_small` | Fast iteration |
| Mistral Medium | `mistral_medium` | Balanced |
| Mistral Large | `mistral_large` | Complex reasoning |

**SDK**: `mistralai>=0.0.7` (optional, install with `pip install meridian[providers]`)

---

### Groq

**Recommended for**: Speed testing, free tier evaluation

```bash
export GROQ_API_KEY=your_key

python -m meridian.cli run --suite instruction_following --model groq_llama70b
```

| Model | Meridian ID | Notes |
|-------|-------------|-------|
| Llama 3 70B | `groq_llama70b` | Best accuracy |
| Llama 3 8B | `groq_llama8b` | Fastest |
| Mixtral 8x7B | `groq_mixtral` | Balanced |

**SDK**: `groq>=0.4.0` (optional, install with `pip install meridian[providers]`)

**Note**: Free tier has rate limits. Paid tier recommended for full suite runs.

---

### Together AI

**Recommended for**: Open-source model evaluation, fine-tuning targets

```bash
export TOGETHER_API_KEY=your_key

python -m meridian.cli run --suite security_adversarial --model together_llama70b
```

| Model | Meridian ID | Notes |
|-------|-------------|-------|
| Llama 3 70B | `together_llama70b` | General purpose |
| Mixtral 8x7B | `together_mixtral` | MoE efficiency |
| Code Llama 34B | `together_codellama` | Code-focused |

**SDK**: `together>=0.2.0` (optional, install with `pip install meridian[providers]`)

---

### Local Models

**Recommended for**: Offline testing, development, privacy

```bash
export DEVICE=cuda  # or cpu, mps

python -m meridian.cli run --suite math_short --model local_distilgpt2
```

| Model | Meridian ID | Size | Notes |
|-------|-------------|------|-------|
| DistilGPT-2 | `local_distilgpt2` | 82M | Fast, limited capability |
| Flan-T5 Small | `local_flan_t5_small` | 60M | Instruction-tuned |

**Requirements**: 
- CPU: Works on any machine
- GPU: CUDA 11.7+ recommended for larger models

---

## Adding Custom Providers

Extend `ModelAdapter` to add your own providers:

```python
from meridian.model_adapters.base import ModelAdapter, GenerationConfig

class MyCustomAdapter(ModelAdapter):
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        # Your implementation
        return response
    
    def get_logprobs(self, prompt: str, response: str) -> list[float]:
        # Optional: return [] if not supported
        return []
```

Register in `meridian/model_adapters/__init__.py`:

```python
AVAILABLE_ADAPTERS["my_custom_model"] = MyCustomAdapter
```

---

## Feature Support Matrix

| Feature | DeepSeek | OpenAI | Anthropic | Mistral | Groq | Together | Local |
|---------|:--------:|:------:|:---------:|:-------:|:----:|:--------:|:-----:|
| Text Generation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Logprobs | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| Function Calling | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Vision | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Embeddings | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |

---

## Quick Setup

```bash
# Install with all optional providers
pip install meridian[providers]

# Or install specific SDKs
pip install mistralai groq together

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

See [.env.example](../.env.example) for all configuration options.
