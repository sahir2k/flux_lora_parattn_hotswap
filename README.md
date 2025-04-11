Quick Script for Fast [FLUX.1 Dev](https://github.com/black-forest-labs/flux) LoRA image generation with [ParaAttention](https://github.com/chengzeyi/ParaAttention) and diffusers [hotswapping](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters#lora).

## Performance

- 2 seconds generation time on 4xA100 SXM
- 0.5 seconds LoRA loading once compiled

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create cache directories (optional)
mkdir -p /workspace/huggingface_cache
mkdir -p /workspace/torch_compile_cache

# Add your HF token to main.py
# Replace YOUR_HUGGINGFACE_TOKEN in the code
```

## Usage

```bash
# Run with torchrun (adjust GPU count as needed)
torchrun --nproc_per_node=4 main.py
```

Follow the prompts to:

1. Enter LoRA adapter ID (or use default)
2. Provide generation prompt
3. Set inference steps

Output is saved as `output.png`
