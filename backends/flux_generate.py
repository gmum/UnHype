import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import argparse
import random
import time
from functools import partial
from contextlib import contextmanager

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from diffusers import FluxPipeline
from accelerate.utils import set_seed as hf_set_seed
from huggingface_hub import login
from transformers import CLIPTextModel, CLIPTokenizer

from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora
from core.prompts import coerce_prompt, load_config


def _setup_hf_env():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token)
    else:
        print("Warning: HF_TOKEN not set.")


@contextmanager
def temporary_global_seed(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()

    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

def load_checkpoint(path, device='cpu'):
    """Load checkpoint, handling both new (embedded config) and legacy formats."""
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and 'state_dict' in raw:
        return raw['state_dict'], raw.get('config', {})
    if isinstance(raw, dict) and 'module' in raw:
        return raw['module'], {}
    if not isinstance(raw, dict):
        raise ValueError(f"Loaded LoRA checkpoint is not a dict: {type(raw)}")
    return raw, {}  # legacy format


def load_lora_weights(model_wrapper, lora_state_dict, device):
    """Load pre-extracted LoRA state_dict into model."""
    transformer = model_wrapper.transformer

    tensor_map = {n: p for n, p in transformer.named_parameters()}
    buffer_map = {n: b for n, b in transformer.named_buffers()}
    tensor_map.update(buffer_map)

    common = [k for k in lora_state_dict.keys() if k in tensor_map]
    print(f"Loading {len(common)}/{len(lora_state_dict)} keys from checkpoint")

    with torch.no_grad():
        for k in common:
            t = tensor_map[k]
            v = lora_state_dict[k].to(device=t.device, dtype=t.dtype)
            t.copy_(v)

    return model_wrapper


def main():
    _setup_hf_env()

    parser = argparse.ArgumentParser(description="Generate images with Flux HyperLoRA from CSV")
    parser.add_argument("--config", type=str, default=None,
                       help="Config YAML (reads rank, lora_alpha, etc. as defaults)")
    parser.add_argument("--csv_path", type=str, default="data/I2P_prompts_4703.csv")
    parser.add_argument("--output_dir", type=str, default="generated_results_flux_lora")
    parser.add_argument("--save_folder", type=str, default="images")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--nudity", type=bool, default=True)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--n_images", type=int, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None,
                       help="LoRA rank (must match training config)")
    parser.add_argument("--lora_alpha", type=float, default=None,
                       help="LoRA alpha (must match training config)")
    parser.add_argument("--hyper_train_steps", type=int, default=None,
                       help="Hypernetwork timesteps (must match training config)")
    parser.add_argument("--use_pooler", type=bool, default=None,
                       help="Use CLIP pooler output")
    parser.add_argument("--use_orig_concat", type=bool, default=None,
                       help="Use original concat in HyperLoRA (must match training config)")
    parser.add_argument("--internal_size", type=int, default=None,
                       help="HyperLoRA hidden size (must match training config)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    # Load checkpoint early to extract embedded config
    ckpt_config = {}
    lora_state_dict = None
    if args.lora_path and os.path.exists(args.lora_path):
        print(f"Loading checkpoint: {args.lora_path}")
        lora_state_dict, ckpt_config = load_checkpoint(args.lora_path)
        if ckpt_config:
            print(f"Found embedded config in checkpoint: {list(ckpt_config.keys())}")

    if args.config:
        config, _ = load_config(args.config)
        config_defaults = {
            'rank': config['rank'],
            'lora_alpha': config['lora_alpha'],
            'hyper_train_steps': config.get('hyper_train_steps', 300),
            'use_pooler': config.get('use_pooler', True),
            'use_orig_concat': config.get('use_orig_concat', False),
            'internal_size': config.get('internal_size', 100),
        }
        for key, default in config_defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, default)

    # Embedded checkpoint config as final fallback (after CLI args and YAML config)
    if ckpt_config:
        ckpt_defaults = {
            'rank': ckpt_config.get('rank'),
            'lora_alpha': ckpt_config.get('lora_alpha'),
            'hyper_train_steps': ckpt_config.get('hyper_train_steps'),
            'use_pooler': ckpt_config.get('use_pooler'),
            'use_orig_concat': ckpt_config.get('use_orig_concat'),
            'internal_size': ckpt_config.get('internal_size'),
        }
        for key, default in ckpt_defaults.items():
            if getattr(args, key) is None and default is not None:
                setattr(args, key, default)

    if args.rank is None:
        raise ValueError("--rank is required (via config, checkpoint, or CLI)")
    if args.lora_alpha is None:
        raise ValueError("--lora_alpha is required (via config, checkpoint, or CLI)")
    if args.hyper_train_steps is None:
        args.hyper_train_steps = 300
    if args.use_pooler is None:
        args.use_pooler = True
    if args.use_orig_concat is None:
        args.use_orig_concat = False
    if args.internal_size is None:
        args.internal_size = 100

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    hf_set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.set_num_threads(torch.get_num_threads())

    cache_dir = "./models"
    os.makedirs(cache_dir, exist_ok=True)
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir=cache_dir)
    pipe = pipe.to(device)

    pipe_device = device

    df = pd.read_csv(args.csv_path, index_col=0)
    model_wrapper = pipe.transformer

    save_dir = os.path.join(args.output_dir, args.save_folder)
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(pipe_device).eval()

    images_generated = 0

    print("Setting up HyperLoRA...")
    model_wrapper.hyper = HypernetworkManager()

    clip_size = 768 if args.use_pooler else 512
    target_modules = ["attn.add_v_proj", "attn.to_v", "attn.to_out.0"]
    load_seed = 42  # global seed for hypernetwork init/load only

    with temporary_global_seed(load_seed):
        hyper_lora_factory = partial(
            HyperLoRALinear,
            clip_size=clip_size,
            rank=args.rank,
            alpha=args.lora_alpha,
            train_steps=args.hyper_train_steps,
            use_orig_concat=args.use_orig_concat,
            internal_size=args.internal_size,
        )
    hyper_lora_layers = inject_hyper_lora(
        model_wrapper, target_modules, hyper_lora_factory
    )

    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(model_wrapper)
        model_wrapper.hyper.add_hyperlora(layer_name, layer.hyper_lora)

    print(f"Injected HyperLoRA into {len(hyper_lora_layers)} layers")

    if lora_state_dict is None:
        if not args.lora_path or not os.path.exists(args.lora_path):
            raise FileNotFoundError(
                f"LoRA weights not found at: {args.lora_path}\n"
                f"Make sure training completed successfully and saved to this path."
            )
        lora_state_dict, _ = load_checkpoint(args.lora_path)
    load_lora_weights(pipe, lora_state_dict, device)

    final_save_path = "./test_ckp"
    os.makedirs(final_save_path, exist_ok=True)

    lora_path = os.path.join(final_save_path, f"hyper_lora_.pth")
    hyperlora_state_dict = {k: v.detach().cpu() for k, v in model_wrapper.state_dict().items() if ".hyper_lora." in k}

    lora_path = os.path.join(final_save_path, f"hyper_lora.pth")
    torch.save(hyperlora_state_dict, lora_path)

    print(f"HyperLoRA saved to: {lora_path}")

    df = pd.read_csv(args.csv_path, index_col=0)

    if args.nudity and "nudity_percentage" in df.columns:
        # Ensure numeric values
        df["nudity_percentage"] = pd.to_numeric(
            df["nudity_percentage"], errors="coerce"
        )

        # Sort by highest nudity first
        df = df.sort_values(by="nudity_percentage", ascending=False)

    save_dir = os.path.join(args.output_dir, args.save_folder)
    os.makedirs(save_dir, exist_ok=True)

    images_generated = 0
    for i, row in tqdm(enumerate(df.to_dict("records")), total=len(df)):
        if args.n_images is not None and images_generated >= args.n_images:
            break

        if "case_number" in row and row["case_number"] not in (None, ""):
            case_number = int(row["case_number"])
            image_path = os.path.join(save_dir, f"{case_number}.png")
        else:
            image_path = os.path.join(save_dir, f"{i:05d}.png")

        if os.path.exists(image_path):
            continue
        prompt = coerce_prompt(row.get("prompt", ""))
        if not isinstance(prompt, str) or not prompt.strip():
            print(f"Skip [{i}] empty prompt")
            continue

        weight_dtype = torch.bfloat16

        hyper_device = (
            model_wrapper.hyper.hyper_layers[0].alpha.device
            if model_wrapper.hyper.hyper_layers
            else torch.device("cpu")
        )

        with torch.no_grad():
            inputs = tokenizer(
                prompt,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(pipe_device).input_ids

            with torch.no_grad():
                if args.use_pooler:
                    context_emb = clip_text_encoder(inputs).pooler_output.detach()
                else:
                    context_emb = clip_text_encoder(inputs).last_hidden_state.detach()

        context_emb = context_emb.to(device=hyper_device)
        timestep = torch.tensor([args.hyper_train_steps], device=hyper_device)

        model_wrapper.hyper.set_context(context_emb.to(device=hyper_device),
                                       timestep)
        model_wrapper.hyper.compute_and_cache_loras(context_emb.to(device=hyper_device),
                                           timestep)

        seed = int(row.get("evaluation_seed", 0))
        hf_set_seed(seed)
        generator = torch.Generator(device).manual_seed(seed)

        start = time.time()
        image = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            num_inference_steps=args.num_inference_steps,
            height=args.image_size,
            width=args.image_size,
            generator=generator,
            max_sequence_length=256
        ).images[0]
        image.save(image_path)
        images_generated += 1
        end = time.time()
        print(f"Prompt [{prompt}] processed in {end - start:.2f} seconds. Saved to {image_path}")


if __name__ == "__main__":
    main()
