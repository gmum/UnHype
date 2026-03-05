#!/usr/bin/env python3

import sys, os as _os

sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import random
import copy
from pathlib import Path
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from utils import set_seed
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora
from ldm.models.diffusion.ddim import DDIMSampler
from utils.sampling import sample_model
from utils import load_model_from_config, print_trainable_parameters
from core.prompts import prompt_augmentation, load_config
from core.caching import HyperCache
from core.cfg_models import CombinedCFGModel
from core.embeddings import setup_embedding_model, _load_nv_embed_module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simplified HyperLoRA Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--use_huge",
        action="store_true",
        default=False,
        help="Use largest CLIP model (ViT-G/14, 1280 dim) instead of ViT-L/14 (768 dim)",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging (requires `wandb login`)",
    )
    return parser.parse_args()


def create_quick_sampler(model, sampler, image_size: int, ddim_steps: int, ddim_eta: float):
    return lambda conditioning, scale, start_code, till_T: sample_model(
        model,
        sampler,
        conditioning,
        image_size,
        image_size,
        ddim_steps,
        scale,
        ddim_eta,
        start_code=start_code,
        till_T=till_T,
        verbose=False,
    )


def generate_images(
        sampler,
        model,
        prompt: str,
        device: torch.device,
        steps: int = 50,
        eta: float = 0.0,
        batch_size: int = 1,
        start_code: torch.Tensor = None,
        guidance_scale: float = 7.5,
):
    if start_code is None:
        start_code = torch.randn(batch_size, 4, 64, 64, device=device)

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        cond = model.get_learned_conditioning([prompt] * start_code.shape[0])
        uncond = model.get_learned_conditioning([""] * start_code.shape[0])

        samples, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=start_code.shape[0],
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


def main():
    args = parse_args()

    # Load configuration
    config, config_name = load_config(args.config)
    if args.use_wandb:
        config['report_to'] = 'wandb'
    print(f"=== Training with config: {config_name} ===")
    print(f"Config file: {args.config}")

    # Extract key parameters with defaults
    learning_rate_remove = config.get('learning_rate_remove', 1e-5)
    learning_rate_retain = config.get('learning_rate_retain', 1e-5)
    max_train_steps = config.get('max_train_steps', 120)
    hyper_train_steps = config.get('hyper_train_steps', 500)
    rank_lora = config.get('rank', 1)  # named rank_lora to avoid confusion with proc rank
    lora_alpha = config.get('lora_alpha', 8)
    internal_size = config.get('internal_size', 100)
    seed = config.get('seed', 2024)
    resolution = config.get('resolution', 512)
    diagnostic_freq = config.get('diagnostic_freq', 500)
    use_orig_concat = config.get('use_orig_concat', False)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

    # Multi-concept configuration
    concepts = config.get('concepts', [])
    mapping_concept = config.get('mapping_concept', [])
    retain_csv_path = config.get('retain_csv_path', None)
    min_retain_sample = config.get('min_retain_sample', 10)

    # Augmentation flags
    augment_target = config.get('augment_target', True)
    augment_retain = config.get('augment_retain', False)
    celebrity_mode = config.get('celebrity_mode', False)
    use_huge = config.get('use_huge', False)
    verbose = config.get('verbose', 0)

    # Embedding model configuration: "clip", "clip_huge", or "nv_embed"
    embedding_model = config.get('embedding_model', 'clip')
    use_pooler = config.get('use_pooler', True)

    # Retain balancing parameters
    retain_steps_per_remove = config.get('retain_steps_per_remove', 1)
    retain_batch_size = config.get('retain_batch_size', min(min_retain_sample, retain_steps_per_remove))
    learning_rate_retain = learning_rate_retain

    # Paths
    output_dir = config.get('output_dir', './output')
    final_save_path = config.get('final_save_path', './saved_model/LoRA_fusion_model')
    pretrained_model_path = config.get('pretrained_model_name_or_path', './models/sd-v1-4.ckpt')
    model_config_path = config.get('model_config', './configs/stable-diffusion/v1-inference.yaml')

    # Training settings
    ddim_steps = 50
    ddim_eta = 0.0
    negative_guidance = config['negative_guidance']
    guidance_scale = config.get('guidance_scale', 7.5)
    start_guidance = config.get('start_guidance', 9.0)
    internal_lr = config.get('internal_lr', 1e-4)

    diagnostic_prompts = config.get('diagnostic_prompts', [])
    if not diagnostic_prompts:
        diagnostic_prompts = [
            f"a photo of {concepts[0]}" if concepts else "a photo of a person",
            "a photo of a cat",
            "a photo of a car"
        ]

    # Config to embed in LoRA checkpoints (generation-relevant params only)
    checkpoint_config = {
        'rank': rank_lora,
        'lora_alpha': lora_alpha,
        'internal_size': internal_size,
        'hyper_train_steps': hyper_train_steps,
        'use_orig_concat': use_orig_concat,
        'use_pooler': use_pooler,
        'embedding_model': embedding_model,
        'backend': 'sd',
    }

    print(f"Training steps: {max_train_steps}")
    print(f"Hypernetwork steps: {hyper_train_steps}")
    print(f"Learning rate (remove): {learning_rate_remove}")
    print(f"Learning rate (retain): {learning_rate_retain}")
    print(f"Retain steps per remove: {retain_steps_per_remove}")
    print(f"Retain batch size: {retain_batch_size}")
    print(f"LoRA rank: {rank_lora}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"Target concepts: {len(concepts)}")
    print("=" * 48)

    if seed is not None:
        set_seed(seed)

    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=config.get('logging_dir', 'logs'),
    )

    # Handle report_to - None or 'none' means no logging
    report_to = config.get('report_to', None)
    if report_to in (None, 'none', 'None', ''):
        log_with = None
    else:
        log_with = report_to

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=config.get('mixed_precision', None),
        log_with=log_with,
        project_config=accelerator_project_config,
    )

    is_main = accelerator.is_main_process

    use_wandb = config.get('report_to') == 'wandb'
    if is_main and use_wandb:
        wandb.init(
            project="UnHype",
            name=f"{config_name}_training",
            config=config
        )

    # Load model (only one)
    model = load_model_from_config(
        model_config_path, pretrained_model_path, accelerator.device
    )

    # Freeze backbone
    for p in model.model.diffusion_model.parameters():
        p.requires_grad = False

    # Setup HyperLoRA
    model.hyper = HypernetworkManager()

    # Determine embedding dimension based on embedding_model config
    if embedding_model == 'nv_embed':
        nv_embed_mod = _load_nv_embed_module()
        clip_size = nv_embed_mod.NV_EMBED_DIM  # 4096
    elif embedding_model == 'clip_huge':
        clip_size = 1280
    else:  # default: clip
        clip_size = 768 if use_pooler else 512

    target_modules = ["attn2.to_k", "attn2.to_v"]

    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=clip_size,
        rank=rank_lora,
        alpha=lora_alpha,
        train_steps=hyper_train_steps,
        use_orig_concat=use_orig_concat,
        internal_size=internal_size,
    )

    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, target_modules, hyper_lora_factory
    )

    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(model)

    # Setup optimizer
    trainable_params = list(filter(lambda p: p.requires_grad, model.model.diffusion_model.parameters()))

    if is_main:
        print(f"Total trainable parameter tensors: {len(trainable_params)}")
        if verbose:
            print_trainable_parameters(model)

    optimizer_remove = torch.optim.Adam(trainable_params, lr=learning_rate_remove)
    optimizer_retain = torch.optim.Adam(trainable_params, lr=learning_rate_retain)

    gamma = config.get('gamma', 0.9)
    step_size = config.get('step_size', 300)

    scheduler_remove = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_remove, milestones=[step_size], gamma=gamma
    )
    scheduler_retain = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_retain, milestones=[step_size], gamma=gamma
    )
    if is_main:
        print(f"Using MultiStepLR schedulers (step_size={step_size}, gamma={gamma})")

    model, optimizer_remove, optimizer_retain = accelerator.prepare(model, optimizer_remove, optimizer_retain)

    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(accelerator.unwrap_model(model))
        accelerator.unwrap_model(model).hyper.add_hyperlora(layer_name, layer.hyper_lora)

    sampler = DDIMSampler(accelerator.unwrap_model(model))

    # Setup embedding model based on config
    nv_embed_model = None
    nv_embed_tokenizer = None
    clip_text_encoder = None
    tokenizer = None
    use_open_clip = False

    if embedding_model == 'nv_embed':
        nv_embed_mod = _load_nv_embed_module()
        print(f"Loading {nv_embed_mod.NV_EMBED_MODEL_NAME} for HyperLoRA context embeddings...")
        nv_embed_model, nv_embed_tokenizer = nv_embed_mod.load_nv_embed_model(
            accelerator.device, torch.float16
        )
        # Still need CLIP tokenizer for the diffusion model conditioning
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device).eval()

        def get_embedding(text: str):
            with torch.no_grad():
                return nv_embed_mod.compute_nv_embed(
                    [text], nv_embed_model, nv_embed_tokenizer, accelerator.device
                ).detach()

        embed_model_name = "nv_embed"

    elif embedding_model == 'clip_huge':
        import open_clip
        print("Using HUGE CLIP model: ViT-bigG-14 (1280 dim) via open_clip")
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
        clip_text_encoder = clip_model.to(accelerator.device).eval()
        tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
        use_open_clip = True

        def get_embedding(text: str):
            with torch.no_grad():
                tokens = tokenizer(text).to(accelerator.device)
                return clip_text_encoder.encode_text(tokens).detach()

        embed_model_name = "clip_huge"

    else:  # default: clip
        print("Using standard CLIP model: ViT-L/14 (768 dim)")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device).eval()

        def get_embedding(text: str):
            with torch.no_grad():
                inputs = tokenizer(
                    text,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(accelerator.device).input_ids
                if use_pooler:
                    return clip_text_encoder(inputs).pooler_output.detach()
                else:
                    return clip_text_encoder(inputs).last_hidden_state.detach()

        embed_model_name = "clip"

    # Print embedding dimensionality for verification
    target_concepts = [c for c in concepts]
    if target_concepts:
        test_emb = get_embedding(target_concepts[0])
        print(f"Embedding shape ({embed_model_name}): {test_emb.shape} (first concept: '{target_concepts[0]}')")

    # Build list of all prompts to cache
    all_prompts_to_cache = []

    # Add augmented target prompts
    all_augmented_targets = []
    for concept in target_concepts:
        if augment_target:
            augmented = prompt_augmentation(concept, augment=True, celebrity=celebrity_mode)
            all_augmented_targets.extend(augmented)
        else:
            all_augmented_targets.append(concept)
    all_prompts_to_cache.extend(all_augmented_targets)

    # Add augmented mapping prompts
    all_augmented_mappings = []
    for concept in mapping_concept:
        if augment_target:
            augmented = prompt_augmentation(concept, augment=True, celebrity=celebrity_mode)
            all_augmented_mappings.extend(augmented)
        else:
            all_augmented_mappings.append(concept)
    all_prompts_to_cache.extend(all_augmented_mappings)

    # Add diagnostic prompts
    all_prompts_to_cache.extend(diagnostic_prompts)

    retain_prompts = []
    if retain_csv_path and os.path.exists(retain_csv_path):
        print(f"Loading retain prompts from CSV: {retain_csv_path}")
        df = pd.read_csv(retain_csv_path)
        if 'prompt' not in df.columns:
            raise ValueError(f"CSV file must have a 'prompt' column. Found columns: {df.columns.tolist()}")

        base_prompts = df['prompt'].dropna().tolist()
        if augment_retain:
            for prompt in base_prompts:
                if prompt.startswith("A photo of the "):
                    prompt = prompt[len("A photo of the "):]
                augmented = prompt_augmentation(prompt, augment=True)
                retain_prompts.extend(augmented)
        else:
            retain_prompts = base_prompts
        all_prompts_to_cache.extend(retain_prompts)

    # Remove duplicates while preserving order
    seen = set()
    unique_prompts = []
    for p in all_prompts_to_cache:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)
    all_prompts_to_cache = unique_prompts

    # Setup cache directory and path
    cache_dir = os.path.join(output_dir, "cache")
    if is_main:
        os.makedirs(cache_dir, exist_ok=True)

    cache_name = concepts[0].replace(' ', '_').replace(',', '')[:30] if concepts else 'default'
    cache_path = os.path.join(cache_dir, f"hyper_cache_{cache_name}.pt")

    # Load or create HyperCache
    hyper_cache = None
    if os.path.exists(cache_path):
        hyper_cache = HyperCache.load(cache_path, expected_prompts=all_prompts_to_cache)

    if hyper_cache is None:
        if is_main:
            hyper_cache = HyperCache(
                prompts=all_prompts_to_cache,
                embed_fn=get_embedding,
                device=accelerator.device,
                batch_size=8,
                embed_model_name=embed_model_name,
            )
            hyper_cache.save(cache_path)
        accelerator.wait_for_everyone()
        if not is_main:
            hyper_cache = HyperCache.load(cache_path)

    print(f"[HyperCache] {len(hyper_cache)} prompts cached ({embed_model_name})")

    criterion = torch.nn.MSELoss()
    losses = []

    quick_sampler = create_quick_sampler(
        accelerator.unwrap_model(model), sampler, resolution, ddim_steps, ddim_eta
    )

    pbar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    remove_weight = config.get('remove_weight', 1.0)
    retain_weight = config.get('retain_weight', 0.001)

    for iteration in pbar:
        base = accelerator.unwrap_model(model)

        t_enc = torch.randint(ddim_steps, (1,), device=accelerator.device)
        og_num = round((int(t_enc) / ddim_steps) * 100)
        og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=accelerator.device)
        start_code = torch.randn((1, 4, resolution // 8, resolution // 8), device=accelerator.device)

        # Zero gradients at start of iteration (fix: mirrors train_flux_simple.py pattern)
        optimizer_remove.zero_grad(set_to_none=True)
        optimizer_retain.zero_grad(set_to_none=True)

        with accelerator.accumulate(model):
            rank_proc = accelerator.process_index
            world_size = accelerator.num_processes
            valid_indices = list(range(rank_proc, len(target_concepts), world_size))

            if len(valid_indices) == 0:
                concept_idx = rank_proc % len(target_concepts)
            else:
                concept_idx = random.choice(valid_indices)

            target_text = target_concepts[concept_idx]
            mapping_text = mapping_concept[concept_idx] if concept_idx < len(mapping_concept) else mapping_concept[0]

            if augment_target:
                augmented_prompts = prompt_augmentation(target_text, augment=True, celebrity=celebrity_mode)
                valid_aug_indices = list(range(rank_proc, len(augmented_prompts), world_size))
                aug_idx = random.choice(valid_aug_indices) if len(valid_aug_indices) > 0 else rank_proc % len(
                    augmented_prompts)
                target_text_augmented = augmented_prompts[aug_idx]
                augmented_mapping = prompt_augmentation(mapping_text, augment=True, celebrity=celebrity_mode)
                mapping_text_augmented = augmented_mapping[aug_idx % len(augmented_mapping)]

                if verbose:
                    pbar.write(f"{target_text_augmented}  ->  {mapping_text_augmented}")

            else:
                target_text_augmented = target_text
                mapping_text_augmented = mapping_text

            # Get embedding from cache (moves from CPU to device)
            target_emb = hyper_cache.get(target_text_augmented, accelerator.device)

            with torch.no_grad():
                emb_p = base.get_learned_conditioning([target_text_augmented])
                emb_n = base.get_learned_conditioning([target_text_augmented])
                emb_m = base.get_learned_conditioning([mapping_text_augmented])

            valid_timesteps = torch.arange(rank_proc, hyper_train_steps, world_size, device=accelerator.device)
            rtimestep = int(valid_timesteps[torch.randint(0, valid_timesteps.numel(),
                                                          (1,))]) if valid_timesteps.numel() > 0 else int(
                torch.randint(0, hyper_train_steps, (1,), device=accelerator.device))

            base.hyper.set_context(target_emb, torch.tensor([rtimestep], device=accelerator.device))
            _, current_timestep = base.hyper.get_context()
            base.hyper.compute_and_cache_loras(target_emb, current_timestep)

            with torch.no_grad():
                # Use base model without LoRA for reference outputs
                with base.hyper.no_lora():
                    z = quick_sampler(emb_p, start_guidance, start_code, int(t_enc))
                    e_m = base.apply_model(z, t_enc_ddpm, emb_m)
                    e_p = base.apply_model(z, t_enc_ddpm, emb_p)

            base.hyper.set_context(target_emb, current_timestep)
            base.hyper.compute_and_cache_loras(target_emb, current_timestep)
            base.hyper.retain_grad_for_cached_lora()
            e_n = base.apply_model(z, t_enc_ddpm, emb_n)

            target = e_m - (negative_guidance * (e_p - e_m))
            loss_aux = criterion(e_n, target)
            accelerator.backward(loss_aux)

            grads_flat_t = base.hyper.flatten_cached_grads_from_cache()
            if grads_flat_t is None:
                raise RuntimeError("No gradients found in cached LoRA tensors.")

            grads_flat_t = (-1.0 * internal_lr) * grads_flat_t.detach()

            base.hyper.set_context(target_emb, current_timestep)
            base.hyper.compute_and_cache_loras(target_emb, current_timestep)
            tensors_flat_t = base.hyper.flatten_cached_from_cache()

            base.hyper.set_context(target_emb, current_timestep + 1)
            base.hyper.compute_and_cache_loras(target_emb, current_timestep + 1)
            tensors_flat_t1 = base.hyper.flatten_cached_from_cache()

            delta_live = tensors_flat_t1 - tensors_flat_t
            loss_remove = remove_weight * criterion(delta_live, grads_flat_t)
            accelerator.backward(loss_remove)

            loss_remove_log = loss_remove.clone().detach()

            if accelerator.sync_gradients:
                optimizer_remove.step()
                optimizer_remove.zero_grad(set_to_none=True)
                scheduler_remove.step()

            loss_retain_total = torch.tensor(0.0, device=accelerator.device)
            if len(retain_prompts) > 0:
                for _ in range(retain_steps_per_remove):
                    num_retain_samples = min(retain_batch_size, len(retain_prompts))
                    sampled_retain_prompts = random.sample(retain_prompts, num_retain_samples)
                    batch_retain_embs = hyper_cache.get_batch(sampled_retain_prompts, accelerator.device)

                    hyper = base.hyper
                    B = batch_retain_embs.shape[0]
                    perm = torch.randperm(B, device=batch_retain_embs.device)
                    batch_prompts = batch_retain_embs[perm]

                    # Compute LoRAs at t=0
                    weight_dtype = next(hyper.parameters()).dtype  # hyper’s param dtype (bf16 if you casted it)

                    hyper.compute_and_cache_loras(
                        batch_prompts.to(dtype=weight_dtype).to(dtype=weight_dtype),
                        torch.zeros(B, device=accelerator.device, dtype=weight_dtype),
                    )

                    tensors_flat_t0 = hyper.flatten_cached_from_cache()

                    t_ = torch.randint(
                        0,
                        hyper_train_steps + 1,
                        (B,),
                        device=accelerator.device
                    )

                    hyper.compute_and_cache_loras(batch_prompts, t_)
                    tensors_flat_t1 = hyper.flatten_cached_from_cache()

                    delta = tensors_flat_t1 - tensors_flat_t0
                    loss_retain = retain_weight * delta.pow(2).mean()
                    loss_retain_total = loss_retain_total + loss_retain.detach()

                    accelerator.backward(loss_retain)

                    if accelerator.sync_gradients:
                        optimizer_retain.step()
                        optimizer_retain.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    loss_retain_total /= retain_steps_per_remove
                    scheduler_retain.step()

            loss_retain_log = loss_retain_total / max(1, retain_steps_per_remove)

        with torch.no_grad():
            loss_retain_reduced = accelerator.gather(loss_retain_log).mean()
            loss_remove_reduced = accelerator.gather(loss_remove_log).mean()
        losses.append(float(loss_remove_reduced.item() + loss_retain_reduced.item()))

        if is_main and use_wandb:
            current_lr_remove = optimizer_remove.param_groups[0]['lr']
            current_lr_retain = optimizer_retain.param_groups[0]['lr']
            wandb.log({
                "loss_retain": float(loss_retain_reduced.item()),
                "loss_remove": float(loss_remove_reduced.item()),
                "learning_rate_remove": current_lr_remove,
                "learning_rate_retain": current_lr_retain,
                "retain_steps_per_remove": retain_steps_per_remove,
            }, step=iteration)

        if is_main:
            pbar.set_postfix({"retain": f"{float(loss_retain_reduced.item()):.3e}",
                              "remove": f"{float(loss_remove_reduced.item()):.3e}"})

        if is_main and use_wandb and (iteration + 1) % diagnostic_freq == 0:
            for diag_idx, diag_prompt in enumerate(diagnostic_prompts):
                # Get diagnostic embedding from cache
                diag_emb = hyper_cache.get(diag_prompt, accelerator.device)

                diag_time_steps = [0, hyper_train_steps // 2, hyper_train_steps]
                gen = torch.Generator(device=accelerator.device)
                gen.manual_seed(seed)

                start_code_diag = torch.randn(
                    (1, 4, resolution // 8, resolution // 8),
                    generator=gen,
                    device=accelerator.device
                )
                imgs_per_prompt = []

                for h_step in diag_time_steps:
                    h_step_tensor = torch.tensor([h_step], device=accelerator.device)
                    base.hyper.set_context(diag_emb, h_step_tensor)
                    base.hyper.compute_and_cache_loras(diag_emb, h_step_tensor)

                    # Toggle LoRA for unconditional CFG pass inside CombinedCFGModel
                    combined_model = CombinedCFGModel(model=base).eval()
                    combined_sampler = DDIMSampler(model=combined_model)

                    imgs = generate_images(
                        sampler=combined_sampler,
                        model=combined_model,
                        prompt=diag_prompt,
                        device=accelerator.device,
                        steps=50,
                        guidance_scale=guidance_scale,
                        start_code=start_code_diag,
                    )
                    imgs_per_prompt.append(imgs)

                if len(imgs_per_prompt) > 0:
                    row_tensors = []
                    for imgs in imgs_per_prompt:
                        if imgs is None: continue
                        img = imgs[0].clamp(0, 1)
                        im_uint8 = (img * 255).round().to(torch.uint8).cpu()
                        row_tensors.append(im_uint8)
                    if len(row_tensors) > 0:
                        row = torch.cat(row_tensors, dim=2)
                        safe_key = diag_prompt.replace(" ", "_").replace(",", "")[:50]
                        wandb.log({f"diagnostic_{diag_idx}_{safe_key}": wandb.Image(to_pil_image(row),
                                                                                    caption=f"{diag_prompt} | hyper steps: {diag_time_steps}")},
                                  step=iteration)

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(final_save_path, exist_ok=True)

            # Save LoRA weights (state_dict includes both params and buffers)
            model_unwrapped = accelerator.unwrap_model(model)
            lora_state_dict = {k: v.detach().cpu().clone() for k, v in model_unwrapped.model.diffusion_model.state_dict().items() if "hyper_lora" in k}

            lora_path = os.path.join(final_save_path, f"hyper_lora_{iteration}.pth")
            accelerator.save({'state_dict': lora_state_dict, 'config': checkpoint_config}, lora_path)
            print(f"Model saved to: {lora_path}")

    accelerator.wait_for_everyone()
    if is_main:
        print(f"Final loss: {losses[-1]:.3e}")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(final_save_path, exist_ok=True)
        model_unwrapped = accelerator.unwrap_model(model)
        lora_state_dict = {k: v.detach().cpu().clone() for k, v in model_unwrapped.model.diffusion_model.state_dict().items() if "hyper_lora" in k}
        lora_path = os.path.join(final_save_path, f"hyper_lora_final.pth")
        accelerator.save({'state_dict': lora_state_dict, 'config': checkpoint_config}, lora_path)

        config_save = {
            "config_name": config_name,
            "concepts": concepts,
            "rank": rank_lora,
            "learning_rate_remove": learning_rate_remove,
            "learning_rate_retain": learning_rate_retain,
            "max_train_steps": max_train_steps,
            "celebrity_mode": celebrity_mode,
            "final_loss": losses[-1],
        }
        with open(os.path.join(final_save_path, "train_config.json"), "w") as f:
            json.dump(config_save, f, indent=2)

    # Save cache if modified during training
    if is_main and hyper_cache is not None and hyper_cache.dirty:
        print(f"[HyperCache] Saving updated cache...")
        hyper_cache.save(cache_path)

    if is_main and use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
