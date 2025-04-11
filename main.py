import os
import subprocess
import time
import torch
import torch.distributed as dist
from diffusers import FluxPipeline
import traceback

# Import necessary modules from ParaAttention
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

# Environment Setup
os.environ["HUGGINGFACE_TOKEN"] = "YOUR_HUGGINGFACE_TOKEN"  # Replace with your token
os.environ["HF_HOME"] = "/workspace/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/huggingface_cache"
os.environ["TORCH_COMPILE_CACHE_DIR"] = "/workspace/torch_compile_cache"

print(f"HF_HOME is set to: {os.environ['HF_HOME']}")
print(f"HUGGINGFACE_HUB_CACHE is set to: {os.environ['HUGGINGFACE_HUB_CACHE']}")
print(f"TORCH_COMPILE_CACHE_DIR is set to: {os.environ['TORCH_COMPILE_CACHE_DIR']}")


def setup_pipeline():
    """
    Loads the FLUX pipeline with context parallelism and caching.
    Enables LoRA hotswap support but does not load any adapter.
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
    torch.cuda.set_device(local_rank)

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="/workspace/huggingface_cache",
    ).to("cuda")

    mesh = init_context_parallel_mesh(pipe.device.type, max_ring_dim_size=2)
    parallelize_pipe(pipe, mesh=mesh)
    parallelize_vae(pipe.vae, mesh=mesh._flatten())

    apply_cache_on_pipe(pipe, residual_diff_threshold=0.12)

    pipe.enable_lora_hotswap(target_rank=128)

    return pipe


def generate_image(pipe, prompt, num_inference_steps=25):
    """
    Generates an image from a text prompt.
    Distributed inference: rank 0 returns a PIL image; others return tensor.
    """
    try:
        output_type = "pil" if dist.get_rank() == 0 else "pt"
        result = pipe(
            prompt, num_inference_steps=num_inference_steps, output_type=output_type
        )
        return result.images[0]
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"Error during image generation: {str(e)}")
            if hasattr(e, "__traceback__"):
                print(traceback.format_exc())
        return None


def load_lora_adapter(pipe, adapter_id, adapter_loaded):
    """
    Loads a LoRA adapter.
    - For the first adapter load: if adapter_id is empty, uses the default adapter id.
    - For subsequent loads: performs a hotswap.
    """
    start = time.time()
    success = False

    try:
        if not adapter_loaded:
            if not adapter_id:
                adapter_id = "alvdansen/flux-koda"
            print(f"Loading first LoRA adapter: {adapter_id}")
            pipe.load_lora_weights(adapter_id)
            pipe.transformer = torch.compile(
                pipe.transformer, mode="max-autotune-no-cudagraphs"
            )
            adapter_loaded = True
            success = True
        else:
            print(f"Hotswapping LoRA adapter to: {adapter_id}")
            pipe.load_lora_weights(adapter_id, hotswap=True, adapter_name="default_0")
            success = True

        end = time.time()
        print(f"LoRA adapter loading took {end - start:.2f} seconds.")
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"Error loading adapter '{adapter_id}': {str(e)}")
            print("Make sure the adapter ID is valid (no spaces or special characters)")
            if hasattr(e, "__traceback__"):
                print(traceback.format_exc())
        success = False

    return adapter_loaded, success


def main():
    try:
        pipe = setup_pipeline()
        print(f"Rank {dist.get_rank()}: Pipeline setup completed and ready.")

        print(f"Rank {dist.get_rank()} warming up...")
        for _ in range(2):
            _ = generate_image(pipe, "Warm up prompt", num_inference_steps=10)

        adapter_loaded = False

        while True:
            try:
                if dist.get_rank() == 0:
                    adapter_input = input(
                        "Enter LoRA adapter ID (or press enter to use default on first load / keep current): "
                    ).strip()
                else:
                    adapter_input = ""
                adapter_list = [adapter_input]
                dist.broadcast_object_list(adapter_list, src=0)
                adapter_input = adapter_list[0]

                if not adapter_loaded or adapter_input:
                    adapter_loaded, success = load_lora_adapter(
                        pipe, adapter_input, adapter_loaded
                    )
                    if not success and not adapter_loaded:
                        continue

                if dist.get_rank() == 0:
                    prompt = input("Enter text prompt (or 'exit' to quit): ").strip()
                else:
                    prompt = ""
                prompt_list = [prompt]
                dist.broadcast_object_list(prompt_list, src=0)
                prompt = prompt_list[0]

                if prompt.lower() == "exit":
                    break

                if dist.get_rank() == 0:
                    steps_input = input(
                        "Enter number of inference steps (default 25): "
                    ).strip()
                    steps = int(steps_input) if steps_input.isdigit() else 25
                else:
                    steps = 25
                steps_list = [steps]
                dist.broadcast_object_list(steps_list, src=0)
                steps = steps_list[0]

                print(f"Using {steps} inference steps.")
                start_time = time.time()
                output_image = generate_image(pipe, prompt, num_inference_steps=steps)

                if output_image is not None:
                    end_time = time.time()
                    if dist.get_rank() == 0:
                        print(f"Inference time: {end_time - start_time:.2f} seconds.")
                        output_filename = "output.png"
                        output_image.save(output_filename)
                        print(f"Image saved as {output_filename}.")
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"Error in main loop: {str(e)}")
                    print(traceback.format_exc())
                    print("Continuing to next prompt...")
                dist.barrier()
                continue

        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"Critical error in main function: {str(e)}")
            print(traceback.format_exc())
        try:
            dist.barrier()
            dist.destroy_process_group()
        except:
            pass


if __name__ == "__main__":
    main()
