#!/usr/bin/env python3

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import re
import time
import json
import torch
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora
from ldm.models.diffusion.ddim import DDIMSampler
from utils import load_model_from_config, set_seed
from core.prompts import coerce_prompt, load_config
from core.cfg_models import CombinedCFGModel
from core.embeddings import setup_embedding_model


def load_checkpoint(path, device='cpu'):
    """Load checkpoint, handling both new (embedded config) and legacy formats."""
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and 'state_dict' in raw:
        return raw['state_dict'], raw.get('config', {})
    return raw, {}  # legacy format


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified image generation with HyperLoRA'
    )
    parser.add_argument('--task', type=str, required=True,
                        choices=['celebrity', 'nudity', 'cifar10'],
                        help='Task type: celebrity, nudity, or cifar10')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to training config YAML (optional if checkpoint has embedded config)')
    parser.add_argument('--lora-path', type=str, required=True,
                        help='Path to trained HyperLoRA weights (.pth file or directory)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save generated images')
    parser.add_argument('--prompts-csv', type=str, default=None,
                        help='CSV file with prompts (columns: prompt, evaluation_seed)')
    parser.add_argument('--prompts-json', type=str, default=None,
                        help='JSON file with prompts (for cifar10: target, synonyms, other)')
    parser.add_argument('--n-images', type=int, default=None,
                        help='Limit number of images to generate')
    parser.add_argument('--samples-per-prompt', type=int, default=1,
                        help='Number of images per prompt (for cifar10)')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of DDIM sampling steps')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
                        help='CFG guidance scale')
    parser.add_argument('--ddim-eta', type=float, default=0.0,
                        help='DDIM eta parameter')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Image size for generation')
    parser.add_argument('--filter-nudity', action='store_true',
                        help='Filter CSV to keep only rows with nudity_percentage > 0 and sort by nudity descending')
    parser.add_argument('--model-config', type=str, default=None,
                        help='Override: path to SD model config YAML')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Override: path to SD model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run generation on')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed for initialization')

    return parser.parse_args()


def load_lora_weights(model, lora_state_dict: dict, device: torch.device):
    """Load pre-extracted LoRA state_dict into model."""
    print(f"Loading {len(lora_state_dict)} LoRA parameters")

    diffusion_model = model.model.diffusion_model
    sd = diffusion_model.state_dict()

    updated = 0
    skipped = []

    with torch.no_grad():
        for k, v in lora_state_dict.items():
            if k in sd:
                if torch.is_tensor(sd[k]) and torch.is_tensor(v) and sd[k].shape == v.shape:
                    sd[k].copy_(v.to(sd[k].dtype).to(device))
                    updated += 1
                else:
                    skipped.append((k, f"shape mismatch"))
            else:
                skipped.append((k, "no such key"))

    print(f"[LoRA] Loaded {updated} tensors, skipped {len(skipped)}")
    return model


def generate_image(sampler, model, prompt: str, device: torch.device,
                   steps: int = 50, eta: float = 0.0, start_code=None,
                   guidance_scale: float = 7.5):
    if start_code is None:
        start_code = torch.randn(1, 4, 64, 64, device=device)

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        cond = model.get_learned_conditioning([prompt])
        uncond = model.get_learned_conditioning([""])

        samples, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=1,
            shape=start_code.shape[1:],
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning={"c_crossattn": [uncond]},
            eta=eta,
            x_T=start_code,
        )
        decoded = model.decode_first_stage(samples)
        decoded = (decoded + 1.0) / 2.0
        decoded = torch.clamp(decoded, 0.0, 1.0)
        return decoded


def build_prompts_celebrity(config: dict, csv_path: str = None):
    if csv_path:
        df = pd.read_csv(csv_path)
        if 'prompt' not in df.columns or 'evaluation_seed' not in df.columns:
            raise ValueError("Celebrity CSV must have 'prompt' and 'evaluation_seed' columns")
        prompts = []
        for idx, row in df.iterrows():
            prompt = coerce_prompt(row['prompt'])
            seed = int(row['evaluation_seed'])
            safe_prompt = re.sub(r'[^\w\s-]', '', prompt)[:50].strip()
            filename = f"{safe_prompt}_{seed}.png"
            prompts.append((prompt, seed, filename))
        return prompts
    else:
        concepts = config.get('concepts', [])
        prompts = []
        base_seed = config.get('seed', 2024)
        for i, concept in enumerate(concepts):
            prompt = f"a photo of {concept}"
            seed = base_seed + i
            safe_name = re.sub(r'[^\w\s-]', '', concept)[:30].strip().replace(' ', '_')
            filename = f"{safe_name}_{seed}.png"
            prompts.append((prompt, seed, filename))
        return prompts


def build_prompts_nudity(config: dict, csv_path: str, filter_nudity: bool = False):
    if not csv_path:
        raise ValueError("Nudity task requires --prompts-csv")

    df = pd.read_csv(csv_path, index_col=0)

    if "nudity_percentage" in df.columns:
        df["nudity_percentage"] = pd.to_numeric(df["nudity_percentage"], errors="coerce")

        # Optional filtering
        if filter_nudity:
            df = df[df["nudity_percentage"].gt(0)]
            print(f"[Filter] Kept {len(df)} rows with nudity_percentage > 0")

        # Always sort (even if not filtering)
        df = df.sort_values(by="nudity_percentage", ascending=False, na_position="last")

    prompts = []
    for idx, row in df.iterrows():
        prompt = coerce_prompt(row.get("prompt", ""))
        if not prompt:
            continue
        seed = int(row.get("evaluation_seed", 0))
        filename = f"{idx:05d}.png"
        prompts.append((prompt, seed, filename))

    return prompts


def build_prompts_cifar10(config: dict, json_path: str = None, samples_per_prompt: int = 1):
    if json_path:
        with open(json_path, 'r') as f:
            data = json.load(f)
        all_prompts = [data.get("target")] + data.get("synonyms", []) + data.get("other", [])
        all_prompts = [p for p in all_prompts if p]
    else:
        concepts = config.get('concepts', [])
        all_prompts = [f"a photo of {c}" for c in concepts]
        diag = config.get('diagnostic_prompts', [])
        all_prompts.extend(diag)

    base_seed = config.get('seed', 2024)
    prompts = []

    for prompt in all_prompts:
        # Extract class name for subdirectory
        class_name = prompt.split()[-1] if prompt else "unknown"
        class_name = re.sub(r'[^\w-]', '', class_name)

        for i in range(samples_per_prompt):
            seed = base_seed + i
            filename = f"{i:05d}.png"
            prompts.append((prompt, seed, filename, class_name))

    return prompts


def main():
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)

    args = parse_args()

    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(f"cuda:{LOCAL_RANK}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device} (RANK {RANK}/{WORLD_SIZE})")

    set_seed(args.seed)

    # Load checkpoint early to extract embedded config
    lora_file = args.lora_path
    if os.path.isdir(args.lora_path):
        candidates = [f for f in os.listdir(args.lora_path) if f.startswith('hyper_lora') and f.endswith('.pth')]
        if candidates:
            candidates.sort()
            lora_file = os.path.join(args.lora_path, candidates[-1])
        else:
            lora_file = os.path.join(args.lora_path, "hyper_lora.pth")

    print(f"Loading checkpoint: {lora_file}")
    lora_state_dict, ckpt_config = load_checkpoint(lora_file, device)
    if ckpt_config:
        print(f"Found embedded config in checkpoint: {list(ckpt_config.keys())}")

    # Load YAML config if provided (overrides embedded config)
    if args.config:
        config, config_name = load_config(args.config)
        print(f"Loaded config '{config_name}' from {args.config}")
    else:
        if not ckpt_config:
            raise ValueError("No --config provided and checkpoint has no embedded config")
        config = {}
        config_name = "embedded"
        print("Using embedded config from checkpoint")

    # Merge: YAML config takes precedence over embedded checkpoint config
    embedding_model = config.get('embedding_model', ckpt_config.get('embedding_model', 'clip'))
    use_pooler = config.get('use_pooler', ckpt_config.get('use_pooler', True))
    rank = config.get('rank', ckpt_config.get('rank'))
    lora_alpha = config.get('lora_alpha', ckpt_config.get('lora_alpha'))
    hidden_size = config.get('internal_size', ckpt_config.get('internal_size'))
    hyper_train_steps = config.get('hyper_train_steps', ckpt_config.get('hyper_train_steps', 300))
    use_orig_concat = config.get('use_orig_concat', ckpt_config.get('use_orig_concat', False))

    if rank is None:
        raise ValueError("rank not found in config or checkpoint")
    if lora_alpha is None:
        raise ValueError("lora_alpha not found in config or checkpoint")
    if hidden_size is None:
        raise ValueError("internal_size not found in config or checkpoint")

    model_config = args.model_config or config.get('model_config', './configs/stable-diffusion/v1-inference.yaml')
    ckpt_path = args.ckpt or config.get('pretrained_model_name_or_path', 'models/sd-v1-4.ckpt')

    print(f"\nHyperLoRA config:")
    print(f"  embedding_model: {embedding_model}")
    print(f"  rank: {rank}, alpha: {lora_alpha}, hidden_size: {hidden_size}")
    print(f"  hyper_train_steps: {hyper_train_steps}, use_orig_concat: {use_orig_concat}")

    embed_fn, clip_size = setup_embedding_model(embedding_model, use_pooler, device)
    print(f"  clip_size: {clip_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nTask: {args.task}")
    if args.task == 'celebrity':
        prompt_data = build_prompts_celebrity(config, args.prompts_csv)
        use_subdirs = False
    elif args.task == 'nudity':
        prompt_data = build_prompts_nudity(config, args.prompts_csv, args.filter_nudity)
        use_subdirs = False
    else:  # cifar10
        prompt_data = build_prompts_cifar10(config, args.prompts_json, args.samples_per_prompt)
        use_subdirs = True

    print(f"Found {len(prompt_data)} prompts to generate")

    if args.n_images is not None:
        prompt_data = prompt_data[:args.n_images]
        print(f"Limited to {len(prompt_data)} images")

    print(f"\nLoading Stable Diffusion from {ckpt_path}...")
    model = load_model_from_config(model_config, ckpt_path, device)

    print(f"\nSetting up HyperLoRA (rank={rank}, clip_size={clip_size})...")
    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=clip_size,
        rank=rank,
        alpha=lora_alpha,
        train_steps=hyper_train_steps,
        internal_size=hidden_size,
        use_orig_concat=use_orig_concat,
    )

    model.hyper = HypernetworkManager()
    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, ["attn2.to_k", "attn2.to_v"], hyper_lora_factory
    )
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(model)
        model.hyper.add_hyperlora(layer_name, layer.hyper_lora)

    print(f"Injected HyperLoRA into {len(hyper_lora_layers)} layers")

    load_lora_weights(model, lora_state_dict, device)

    combined_model = CombinedCFGModel(model).eval()
    sampler = DDIMSampler(model=combined_model)

    print(f"\nGenerating images...")
    images_generated = 0

    for i, item in enumerate(tqdm(prompt_data, desc="Generating")):
        if use_subdirs:
            prompt, seed, filename, subdir = item
            save_dir = os.path.join(args.output_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
        else:
            prompt, seed, filename = item
            save_dir = args.output_dir

        image_path = os.path.join(save_dir, filename)

        if os.path.exists(image_path):
            continue

        if i % WORLD_SIZE != RANK:
            continue

        start_time = time.time()

        set_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed)
        start_code = torch.randn(
            1, 4, args.image_size // 8, args.image_size // 8,
            generator=gen, device=device
        )

        t_prompt = embed_fn(prompt)
        hyper_device = model.hyper.hyper_layers[0].alpha.device if model.hyper.hyper_layers else device
        weight_dtype = model.hyper.hyper_layers[0].alpha.dtype if model.hyper.hyper_layers else torch.float64

        t_prompt = t_prompt.to(dtype=weight_dtype, device=hyper_device)
        timestep = torch.tensor([hyper_train_steps], dtype=weight_dtype, device=hyper_device)

        model.hyper.set_context(t_prompt, timestep)
        model.hyper.compute_and_cache_loras(t_prompt, timestep)

        with torch.no_grad():
            img = generate_image(
                sampler=sampler,
                model=combined_model,
                prompt=prompt,
                device=device,
                steps=args.steps,
                eta=args.ddim_eta,
                start_code=start_code,
                guidance_scale=args.guidance_scale,
            )

        img_np = img[0].cpu().permute(1, 2, 0).numpy()
        img_pil = to_pil_image((img_np * 255).astype(np.uint8))
        img_pil.save(image_path, format='PNG')

        images_generated += 1
        elapsed = time.time() - start_time

        if images_generated <= 3 or images_generated % 50 == 0:
            print(f"[{images_generated}] '{prompt[:40]}...' -> {filename} ({elapsed:.2f}s)")

    print(f"\nDone! Generated {images_generated} images in {args.output_dir}")


if __name__ == "__main__":
    main()
