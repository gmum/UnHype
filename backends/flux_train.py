#!/usr/bin/env python3

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import random
from typing import List, Optional
from functools import partial

import pandas as pd
import torch
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = hasattr(wandb, 'init')
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from diffusers import AutoencoderKL, FluxPipeline, FluxTransformer2DModel
from tools.prompt_process import encode_prompt, _get_clip_prompt_embeds
from tools.scheduler_process import FlowMatchEulerDiscreteScheduler
from torchvision.transforms.functional import to_pil_image, to_tensor
from backends.flux_bare_generate import generate_one_image_from_prompt
from accelerate import Accelerator
from utils.esd_utils import latent_sample, predict_noise
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from accelerate.utils import ProjectConfiguration, set_seed as hf_set_seed
from tqdm import tqdm

from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora
from utils import print_trainable_parameters
from diffusers.utils.torch_utils import randn_tensor
from core.prompts import prompt_augmentation, load_config


class Cache:

    def __init__(
        self,
        target_prompts: List[str],
        mapping_prompts: List[str],
        diagnostic_prompts: List[str],
        transformer,
        noise_scheduler,
        text_encoders: List,
        tokenizers: List,
        device: torch.device,
        max_ddim_steps: int = 28,
        height: int = 512,
        width: int = 512,
        num_channels_latents: int = 16,
        seed: int = 42,
        weight_dtype: torch.dtype = torch.bfloat16,
        guidance: float = 3.0,
        cache_path: str = None,
    ):
        if cache_path and not os.path.exists(cache_path):
            raise ValueError("Cache path doesn't exist!")

        if max_ddim_steps < 1 or max_ddim_steps > 1000:
            raise ValueError(f"max_ddim_steps must be between 1 and 1000, got {max_ddim_steps}")

        self.target_prompts = target_prompts
        self.mapping_prompts = mapping_prompts
        self.diagnostic_prompts = diagnostic_prompts
        self.max_ddim_steps = max_ddim_steps
        self.seed = seed
        self.device = device
        self.weight_dtype = weight_dtype
        self.dirty = False  # Track if cache was modified (for lazy caching)

        self.target_prompt_to_idx = {prompt: idx for idx, prompt in enumerate(target_prompts)}
        self.mapping_prompt_to_idx = {prompt: idx for idx, prompt in enumerate(mapping_prompts)}
        self.diagnostic_prompt_to_idx = {prompt: idx for idx, prompt in enumerate(diagnostic_prompts)}

        vae_scale_factor = 8
        latent_h = height // vae_scale_factor
        latent_w = width // vae_scale_factor
        self.latent_image_ids = self._prepare_latent_image_ids(
            1, latent_h // 2, latent_w // 2, device, weight_dtype
        )

        print(f"[Cache] Caching {len(target_prompts)} target prompts × {max_ddim_steps} steps (seed={seed})...")
        print(f"[Cache] Caching {len(mapping_prompts)} mapping prompts...")
        print(f"[Cache] Caching {len(diagnostic_prompts)} diagnostic prompts...")

        self.target_embeddings = self._compute_text_embeddings(
            target_prompts, text_encoders, tokenizers, device, desc="Target embeddings"
        )

        if mapping_prompts:
            self.mapping_embeddings = self._compute_text_embeddings(
                mapping_prompts, text_encoders, tokenizers, device, desc="Mapping embeddings"
            )
        else:
            self.mapping_embeddings = None

        print("[Cache] Computing unconditional embeddings...")
        uncond_embeds, uncond_pooled, uncond_text_ids = compute_text_embeddings(
            "", text_encoders, tokenizers, device
        )
        self.uncond_embeddings = {
            'prompt_embeds': uncond_embeds.cpu(),
            'pooled_prompt_embeds': uncond_pooled.cpu(),
            'text_ids': uncond_text_ids.cpu(),
        }

        if diagnostic_prompts:
            self.diagnostic_embeddings = self._compute_text_embeddings(
                diagnostic_prompts, text_encoders, tokenizers, device, desc="Diagnostic embeddings"
            )
        else:
            self.diagnostic_embeddings = None

        self.latents = self._compute_all_latents(
            transformer, noise_scheduler, num_channels_latents,
            height, width, seed, guidance
        )

        self._print_memory_usage()

    def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        latent_image_ids = latent_image_ids.reshape(height * width, 3)
        return latent_image_ids.to(device=device, dtype=dtype)

    def _compute_text_embeddings(self, prompts, text_encoders, tokenizers, device, desc="Text embeddings"):
        embeddings = {'prompt_embeds': [], 'pooled_prompt_embeds': [], 'text_ids': []}
        for prompt in tqdm(prompts, desc=desc):
            prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                prompt, text_encoders, tokenizers, device
            )
            embeddings['prompt_embeds'].append(prompt_embeds.cpu())
            embeddings['pooled_prompt_embeds'].append(pooled_prompt_embeds.cpu())
            embeddings['text_ids'].append(text_ids.cpu())

        embeddings['prompt_embeds'] = torch.cat(embeddings['prompt_embeds'], dim=0)
        embeddings['pooled_prompt_embeds'] = torch.cat(embeddings['pooled_prompt_embeds'], dim=0)
        embeddings['text_ids'] = torch.cat(embeddings['text_ids'], dim=0)
        return embeddings

    def _compute_all_latents(self, transformer, noise_scheduler, num_channels_latents, height, width, seed, guidance):
        from utils.esd_utils import flux_pack_latents, calculate_shift, retrieve_timesteps
        import numpy as np

        num_prompts = len(self.target_prompts)
        vae_scale_factor = 8
        latent_h = height // vae_scale_factor
        latent_w = width // vae_scale_factor
        packed_seq_len = (latent_h // 2) * (latent_w // 2)
        packed_channels = num_channels_latents * 4

        latents_cache = torch.zeros(
            num_prompts, self.max_ddim_steps, packed_seq_len, packed_channels,
            dtype=self.weight_dtype, device='cpu'
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        shape = (1, num_channels_latents, latent_h, latent_w)
        initial_noise = randn_tensor(shape, generator=generator, dtype=self.weight_dtype, device=self.device)
        guidance_tensor = torch.tensor([guidance], device=self.device, dtype=self.weight_dtype)

        pbar = tqdm(total=num_prompts * self.max_ddim_steps, desc="Latents")

        with torch.no_grad():
            for prompt_idx in range(num_prompts):
                prompt_embeds = self.target_embeddings['prompt_embeds'][prompt_idx:prompt_idx+1].to(self.device)
                pooled_prompt_embeds = self.target_embeddings['pooled_prompt_embeds'][prompt_idx:prompt_idx+1].to(self.device)
                text_ids = self.target_embeddings['text_ids'][prompt_idx:prompt_idx+1].to(self.device)
                if text_ids.dim() == 3:
                    text_ids = text_ids[0]
                text_ids = text_ids.to(dtype=self.weight_dtype)

                for ddim_step in range(1, self.max_ddim_steps + 1):
                    latents = flux_pack_latents(initial_noise.clone(), 1, num_channels_latents, latent_h, latent_w)
                    image_seq_len = latents.shape[1]
                    sigmas = np.linspace(1.0, 1 / ddim_step, ddim_step)
                    mu = calculate_shift(
                        image_seq_len,
                        noise_scheduler.config.base_image_seq_len,
                        noise_scheduler.config.max_image_seq_len,
                        noise_scheduler.config.base_shift,
                        noise_scheduler.config.max_shift,
                    )
                    timesteps_tensor, _ = retrieve_timesteps(noise_scheduler, ddim_step, transformer.device, None, sigmas, mu=mu)
                    latents = latents.to(transformer.device).to(self.weight_dtype)

                    for i, t in enumerate(timesteps_tensor):
                        timestep = t.expand(latents.shape[0]).to(self.weight_dtype)
                        noise_pred = transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance_tensor,
                            pooled_projections=pooled_prompt_embeds.to(self.weight_dtype),
                            encoder_hidden_states=prompt_embeds.to(self.weight_dtype),
                            txt_ids=text_ids,
                            img_ids=self.latent_image_ids,
                            return_dict=False,
                        )
                        if isinstance(noise_pred, (tuple, list)):
                            noise_pred = noise_pred[0]
                        latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    latents_cache[prompt_idx, ddim_step - 1] = latents.cpu()
                    pbar.update(1)

        pbar.close()
        return latents_cache

    def get_target(self, prompt: str, ddim_step: int, device: torch.device) -> tuple:
        prompt_idx = self.target_prompt_to_idx[prompt]
        latent = self.latents[prompt_idx, ddim_step - 1].unsqueeze(0).to(device)
        prompt_embeds = self.target_embeddings['prompt_embeds'][prompt_idx:prompt_idx+1].to(device)
        pooled_prompt_embeds = self.target_embeddings['pooled_prompt_embeds'][prompt_idx:prompt_idx+1].to(device)
        text_ids = self.target_embeddings['text_ids'][prompt_idx:prompt_idx+1].to(device)
        return latent, prompt_embeds, pooled_prompt_embeds, text_ids, self.latent_image_ids.to(device)

    def get_target_pooled(self, prompt: str, device: torch.device) -> torch.Tensor:
        idx = self.target_prompt_to_idx[prompt]
        return self.target_embeddings['pooled_prompt_embeds'][idx:idx+1].to(device)

    def get_mapping(self, prompt: str, device: torch.device) -> tuple:
        if self.mapping_embeddings is None:
            raise ValueError(f"No mapping prompts cached")
        if prompt not in self.mapping_prompt_to_idx:
            raise KeyError(f"Mapping prompt '{prompt}' not in cache. Use add_embedding() first.")
        idx = self.mapping_prompt_to_idx[prompt]
        return (
            self.mapping_embeddings['prompt_embeds'][idx:idx+1].to(device),
            self.mapping_embeddings['pooled_prompt_embeds'][idx:idx+1].to(device),
            self.mapping_embeddings['text_ids'][idx:idx+1].to(device),
        )

    def get_mapping_pooled(self, prompt: str, device: torch.device) -> torch.Tensor:
        if self.mapping_embeddings is None:
            raise ValueError(f"No mapping prompts cached")
        idx = self.mapping_prompt_to_idx[prompt]
        return self.mapping_embeddings['pooled_prompt_embeds'][idx:idx+1].to(device)

    def add_embedding(
        self,
        prompt: str,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        embedding_type: str = 'mapping',
    ):

        type_config = {
            'target': (self.target_prompts, self.target_prompt_to_idx, 'target_embeddings'),
            'mapping': (self.mapping_prompts, self.mapping_prompt_to_idx, 'mapping_embeddings'),
            'diagnostic': (self.diagnostic_prompts, self.diagnostic_prompt_to_idx, 'diagnostic_embeddings'),
        }

        prompts_list, prompt_to_idx, embeddings_attr = type_config[embedding_type]

        if prompt in prompt_to_idx:
            return  # Already cached

        embeddings_dict = getattr(self, embeddings_attr)

        if embeddings_dict is None:
            embeddings_dict = {
                'prompt_embeds': prompt_embeds.cpu(),
                'pooled_prompt_embeds': pooled_prompt_embeds.cpu(),
                'text_ids': text_ids.cpu(),
            }
            setattr(self, embeddings_attr, embeddings_dict)
        else:
            embeddings_dict['prompt_embeds'] = torch.cat(
                [embeddings_dict['prompt_embeds'], prompt_embeds.cpu()], dim=0
            )
            embeddings_dict['pooled_prompt_embeds'] = torch.cat(
                [embeddings_dict['pooled_prompt_embeds'], pooled_prompt_embeds.cpu()], dim=0
            )
            embeddings_dict['text_ids'] = torch.cat(
                [embeddings_dict['text_ids'], text_ids.cpu()], dim=0
            )

        idx = len(prompts_list)
        prompts_list.append(prompt)
        prompt_to_idx[prompt] = idx
        self.dirty = True
        print(f"[Cache] Added {embedding_type} embedding for: {prompt[:50]}... (total: {len(prompts_list)})")

    def get_uncond(self, device: torch.device) -> tuple:
        return (
            self.uncond_embeddings['prompt_embeds'].to(device),
            self.uncond_embeddings['pooled_prompt_embeds'].to(device),
            self.uncond_embeddings['text_ids'].to(device),
        )

    def get_diagnostic(self, prompt: str, device: torch.device) -> tuple:
        if self.diagnostic_embeddings is None:
            raise ValueError(f"No diagnostic prompts cached")
        idx = self.diagnostic_prompt_to_idx[prompt]
        return (
            self.diagnostic_embeddings['prompt_embeds'][idx:idx+1].to(device),
            self.diagnostic_embeddings['pooled_prompt_embeds'][idx:idx+1].to(device),
            self.diagnostic_embeddings['text_ids'][idx:idx+1].to(device),
        )

    def get_diagnostic_pooled(self, prompt: str, device: torch.device) -> torch.Tensor:
        if self.diagnostic_embeddings is None:
            raise ValueError(f"No diagnostic prompts cached")
        idx = self.diagnostic_prompt_to_idx[prompt]
        return self.diagnostic_embeddings['pooled_prompt_embeds'][idx:idx+1].to(device)

    def __contains__(self, prompt: str) -> bool:
        return prompt in self.target_prompt_to_idx

    def __len__(self) -> int:
        return len(self.target_prompts) * self.max_ddim_steps

    def _print_memory_usage(self):
        latent_mem = self.latents.element_size() * self.latents.nelement() / (1024 ** 2)
        target_mem = sum(t.element_size() * t.nelement() for t in self.target_embeddings.values()) / (1024 ** 2)
        mapping_mem = 0
        if self.mapping_embeddings:
            mapping_mem = sum(t.element_size() * t.nelement() for t in self.mapping_embeddings.values()) / (1024 ** 2)
        uncond_mem = sum(t.element_size() * t.nelement() for t in self.uncond_embeddings.values()) / (1024 ** 2)
        diag_mem = 0
        if self.diagnostic_embeddings:
            diag_mem = sum(t.element_size() * t.nelement() for t in self.diagnostic_embeddings.values()) / (1024 ** 2)
        total = latent_mem + target_mem + mapping_mem + uncond_mem + diag_mem
        print(f"[Cache] Memory: latents={latent_mem:.1f}MB, target={target_mem:.1f}MB, mapping={mapping_mem:.1f}MB, uncond={uncond_mem:.1f}MB, diag={diag_mem:.1f}MB, total={total:.1f}MB")

    def save(self, path: str):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        data = {
            'target_prompts': self.target_prompts,
            'mapping_prompts': self.mapping_prompts,
            'diagnostic_prompts': self.diagnostic_prompts,
            'max_ddim_steps': self.max_ddim_steps,
            'seed': self.seed,
            'latents': self.latents,
            'target_embeddings': self.target_embeddings,
            'mapping_embeddings': self.mapping_embeddings,
            'uncond_embeddings': self.uncond_embeddings,
            'diagnostic_embeddings': self.diagnostic_embeddings,
            'latent_image_ids': self.latent_image_ids.cpu(),
        }
        torch.save(data, path)
        print(f"[Cache] Saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device,
        weight_dtype: torch.dtype = torch.bfloat16,
        expected_target_prompts: List[str] = None,
        expected_mapping_prompts: List[str] = None,
        expected_diagnostic_prompts: List[str] = None,
        expected_ddim_steps: int = None,
        expected_seed: int = None,
    ):
        print(f"[Cache] Loading from {path}...")
        data = torch.load(path, map_location='cpu', weights_only=False)

        instance = object.__new__(cls)
        instance.target_prompts = data['target_prompts']
        instance.mapping_prompts = data.get('mapping_prompts', [])
        instance.diagnostic_prompts = data.get('diagnostic_prompts', [])
        instance.max_ddim_steps = data['max_ddim_steps']
        instance.seed = data.get('seed')

        instance.device = device
        instance.weight_dtype = weight_dtype
        instance.target_prompt_to_idx = {prompt: idx for idx, prompt in enumerate(instance.target_prompts)}
        instance.mapping_prompt_to_idx = {prompt: idx for idx, prompt in enumerate(instance.mapping_prompts)}
        instance.diagnostic_prompt_to_idx = {prompt: idx for idx, prompt in enumerate(instance.diagnostic_prompts)}
        instance.latents = data['latents']
        instance.target_embeddings = data['target_embeddings']
        instance.mapping_embeddings = data.get('mapping_embeddings', None)
        instance.uncond_embeddings = data['uncond_embeddings']
        instance.diagnostic_embeddings = data.get('diagnostic_embeddings', None)
        instance.latent_image_ids = data['latent_image_ids'].to(device=device, dtype=weight_dtype)
        instance.dirty = False  # Will be set to True if embeddings are added on-the-fly
        instance._print_memory_usage()
        return instance


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simplified HyperLoRA Training for Stable Diffusion"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging (requires `wandb login`)",
    )
    return parser.parse_args()


def load_text_encoders(class_one, class_two, pretrained_model_name_or_path):
    text_encoder_one = class_one.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=None, variant=None
    )
    text_encoder_two = class_two.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2", revision=None, variant=None
    )

    # Disable gradients + set eval mode
    for enc in (text_encoder_one, text_encoder_two):
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)

    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def compute_text_embeddings(prompts, text_encoders, tokenizers, device, max_sequence_length=256):
    # prompts: List[str] or str
    if isinstance(prompts, str):
        prompts = [prompts]

    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
        text_encoders, tokenizers, prompts, max_sequence_length
    )
    return (
        prompt_embeds.to(device),
        pooled_prompt_embeds.to(device),
        text_ids.to(device),
    )


def compute_clip_pooled_embeddings(prompts, text_encoder, tokenizer, device):

    model_device = text_encoder.text_model.embeddings.token_embedding.weight.device

    # If caller provided a device, you can optionally enforce it,
    # but the safest is to compute on model_device.
    compute_device = model_device if device is None else torch.device(device)

    # If compute_device != model_device, move the model (or just ignore device)
    if compute_device != model_device:
        text_encoder = text_encoder.to(compute_device)
        model_device = compute_device

    pooled_embeds = _get_clip_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompts,
        device=device,
        num_images_per_prompt=1,
    )
    return pooled_embeds.to(device)


def main():
    args = parse_args()

    # Load configuration
    config, config_name = load_config(args.config)
    if args.use_wandb:
        config['report_to'] = 'wandb'
    print(f"=== Training with config: {config_name} ===")
    print(f"Config file: {args.config}")

    # Extract key parameters with defaults
    learning_rate_remove = config['learning_rate_remove']
    learning_rate_retain = config['learning_rate_retain']
    max_train_steps = config['max_train_steps']
    hyper_train_steps = config.get('hyper_train_steps', 300)
    rank = config['rank']
    lora_alpha = config['lora_alpha']
    retain_steps_per_remove = config.get('retain_steps_per_remove', 1)
    internal_size = config['internal_size']
    seed = config.get('seed', 2024)

    min_retain_sample = config.get('min_retain_sample', 10)
    resolution = config.get('resolution', 512)
    diagnostic_freq = config.get('diagnostic_freq', 500)
    use_pooler = config.get('use_pooler', True)
    use_orig_concat = config.get('use_orig_concat', False)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

    # Multi-concept configuration
    concepts = config.get('concepts', [])
    mapping_concept = config.get('mapping_concept', [])
    retain_csv_path = config.get('retain_csv_path', None)  # Path to CSV with retain prompts

    # Augmentation flags
    augment_target = config.get('augment_target', True)  # Whether to augment target concepts
    augment_retain = config.get('augment_retain', False)  # Whether to augment retain prompts from CSV
    verbose = config.get('verbose', 0)

    # Paths
    output_dir = config.get('output_dir', './output')
    final_save_path = config.get('final_save_path', './saved_model/LoRA_fusion_model')
    pretrained_model_name_or_path = config.get('pretrained_model_name_or_path', "black-forest-labs/FLUX.1-dev")

    # Training settings
    ddim_steps = 28
    negative_guidance = config['negative_guidance']
    internal_lr = config['internal_lr']

    # Diagnostic prompts for image generation during training
    diagnostic_prompts = config.get('diagnostic_prompts', [])
    if not diagnostic_prompts:
        # Default diagnostic prompts if none provided
        diagnostic_prompts = [
            f"a photo of {concepts[0]}" if concepts else "a photo of a person",
            "a photo of a cat",
            "a photo of a car"
        ]

    # Config to embed in LoRA checkpoints (generation-relevant params only)
    checkpoint_config = {
        'rank': rank,
        'lora_alpha': lora_alpha,
        'internal_size': internal_size,
        'hyper_train_steps': hyper_train_steps,
        'use_orig_concat': use_orig_concat,
        'use_pooler': use_pooler,
        'backend': 'flux',
    }

    print(f"Training steps: {max_train_steps}")
    print(f"Hypernetwork steps: {hyper_train_steps}")
    print(f"Learning rate (remove): {learning_rate_remove}")
    print(f"Learning rate (retain): {learning_rate_retain}")
    print(f"LoRA rank: {rank}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"Target concepts: {len(concepts)}")
    print("=" * 48)

    # Set seed
    if seed is not None:
        hf_set_seed(seed)

    # Setup Accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=config.get('logging_dir', 'logs'),
    )

    # Handle report_to - None or 'none' means no logging
    report_to = config.get('report_to', None)
    if report_to in (None, 'none', 'None', ''):
        log_with = None
    elif WANDB_AVAILABLE:
        log_with = report_to
    else:
        log_with = None
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=config.get('mixed_precision', None),
        log_with=log_with,
        project_config=accelerator_project_config,
    )

    is_main = accelerator.is_main_process

    # Initialize W&B if needed and available
    use_wandb = config.get('report_to') == 'wandb' and WANDB_AVAILABLE
    if is_main and use_wandb:
        wandb.init(
            project="UnHype",
            name=f"{config_name}_training",
            config=config
        )
        wandb.define_metric("learning_rate_remove", summary="last")
        wandb.define_metric("learning_rate_retain", summary="last")
    elif is_main and config.get('report_to') == 'wandb' and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Disabling wandb logging.")

    tokenizer_one = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    tokenizer_two = T5TokenizerFast.from_pretrained(
        "google/t5-v1_1-base"
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, None, subfolder="text_encoder_2"
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two,
                                                            pretrained_model_name_or_path)
    text_encoder_one =text_encoder_one.to(accelerator.device)
    text_encoder_two = text_encoder_two.to(accelerator.device)
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision=None,
        variant=None,
    )

    weight_dtype = torch.bfloat16 if accelerator.device.type == "cuda" else torch.float32

    model = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, torch_dtype=weight_dtype,
        subfolder="transformer", revision=None, variant=None
    ).to(accelerator.device)

    model.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    model.hyper = HypernetworkManager()

    # Flux's pooled_prompt_embeds is always 768-dim (from built-in CLIP text_encoder_one)
    clip_size = 768
    target_modules = ["attn.add_v_proj", "attn.to_v", "attn.to_out.0"]

    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=clip_size,
        rank=rank,
        alpha=lora_alpha,
        train_steps=hyper_train_steps,
        use_orig_concat=use_orig_concat,
        dtype=torch.float32,
        internal_size=internal_size,
    )

    hyper_lora_layers = inject_hyper_lora(
        model, target_modules, hyper_lora_factory
    )

    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(model)

    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if is_main:
        print(f"Total trainable parameter tensors: {len(trainable_params)}")
        if verbose:
            print_trainable_parameters(model)

    optimizer_remove = torch.optim.Adam(model.parameters(), lr=learning_rate_remove)
    optimizer_retain = torch.optim.Adam(model.parameters(), lr=learning_rate_retain)

    gamma = config.get('gamma', 0.9)
    step_size = config.get('step_size', 300)

    scheduler_remove = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_remove, milestones=step_size, gamma=gamma
    )
    scheduler_retain = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_retain, milestones=step_size, gamma=gamma
    )
    if is_main:
        print(f"Using MultiStepLR schedulers (step_size={step_size}, gamma={gamma})")

    # Prepare for distributed training
    model, optimizer_remove, optimizer_retain = accelerator.prepare(model, optimizer_remove, optimizer_retain)

    # Register HyperLoRA layers after prepare
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(accelerator.unwrap_model(model))
        accelerator.unwrap_model(model).hyper.add_hyperlora(layer_name, layer.hyper_lora)

    target_concepts = list(concepts)
    print(f"Target concepts for removal: {target_concepts}")
    print(f"Total target concepts: {len(target_concepts)}")

    retain_prompts = []
    retain_embeddings = []

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

        cache_dir = os.path.join(output_dir, "cache")
        if is_main:
            os.makedirs(cache_dir, exist_ok=True)

        csv_name = os.path.basename(retain_csv_path).replace('.csv', '')
        cache_key = f"{csv_name}_aug{augment_retain}_pooler{use_pooler}"
        cache_path = os.path.join(cache_dir, f"retain_embeddings_{cache_key}.pt")

        cache_exists = os.path.exists(cache_path)
        if not cache_exists:
            if is_main:
                for prompt in tqdm(retain_prompts, desc="Creating retain embeddings"):
                    with torch.no_grad():
                        emb = compute_clip_pooled_embeddings(
                            prompt, text_encoder_one, tokenizer_one, accelerator.device
                        )
                    retain_embeddings.append(emb.squeeze().cpu())
                torch.save(retain_embeddings, cache_path)
            accelerator.wait_for_everyone()

        retain_embeddings = torch.load(cache_path, map_location='cpu')
        retain_embeddings = [emb.to(accelerator.device) for emb in retain_embeddings]

    print(f"Mapping concepts: {len(mapping_concept)}")
    print(f"Retain prompts: {len(retain_prompts)} prompts loaded")

    # Create augmented target prompts
    all_augmented_prompts = []
    if augment_target:
        for concept in target_concepts:
            all_augmented_prompts.extend(prompt_augmentation(concept, augment=True))
    else:
        all_augmented_prompts = target_concepts.copy()
    print(f"Total augmented target prompts for caching: {len(all_augmented_prompts)}")

    # Create augmented mapping prompts (must match target augmentation)
    all_augmented_mapping = []
    if len(mapping_concept) > 0:
        # Ensure we have a mapping concept for each target concept
        mapping_per_target = []
        for i, concept in enumerate(target_concepts):
            # Use corresponding mapping concept, or reuse first one if not enough
            if i < len(mapping_concept):
                mapping_per_target.append(mapping_concept[i])
            else:
                mapping_per_target.append(mapping_concept[0])

        # Apply same augmentation as targets
        if augment_target:
            for mapping_text in mapping_per_target:
                all_augmented_mapping.extend(prompt_augmentation(mapping_text, augment=True))
        else:
            all_augmented_mapping = mapping_per_target.copy()

        print(f"Total augmented mapping prompts for caching: {len(all_augmented_mapping)}")

        # Verify lengths match
        if len(all_augmented_mapping) != len(all_augmented_prompts):
            raise ValueError(
                f"Mismatch: {len(all_augmented_prompts)} target prompts but "
                f"{len(all_augmented_mapping)} mapping prompts. They must match!"
            )
    else:
        print("Warning: No mapping concepts provided. Using empty list.")
        all_augmented_mapping = []

    # Unified cache setup
    use_cache = config.get('use_latent_cache', True)
    cache = None

    cache_dir = config.get('cache_dir', 'latents_cache')
    default_cache_name = concepts[0].replace(' ', '_').replace(',', '')[:30] if concepts else 'default'
    cache_name = config.get('cache_name', default_cache_name)
    cache_path = os.path.join(cache_dir, f"{cache_name}_cache.pt")

    if use_cache and is_main:
        cache_seed = seed if seed else 42
        if os.path.exists(cache_path):
            cache = Cache.load(
                cache_path, accelerator.device, weight_dtype,
                expected_target_prompts=all_augmented_prompts,
                expected_mapping_prompts=all_augmented_mapping,
                expected_diagnostic_prompts=diagnostic_prompts,
                expected_ddim_steps=ddim_steps,
                expected_seed=cache_seed
            )

        if cache is None:
            base_for_cache = accelerator.unwrap_model(model)
            with base_for_cache.hyper.no_lora():
                cache = Cache(
                    target_prompts=all_augmented_prompts,
                    mapping_prompts=all_augmented_mapping,
                    diagnostic_prompts=diagnostic_prompts,
                    transformer=base_for_cache,
                    noise_scheduler=noise_scheduler,
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    device=accelerator.device,
                    max_ddim_steps=ddim_steps,
                    height=512,
                    width=512,
                    num_channels_latents=vae.config.latent_channels,
                    seed=cache_seed,
                    weight_dtype=weight_dtype,
                    guidance=3.0,
                    cache_path=cache_path,
                )
            cache.save(cache_path)

    accelerator.wait_for_everyone()

    criterion = torch.nn.MSELoss()
    losses = []

    base = accelerator.unwrap_model(model)
    diag_pipe = FluxPipeline(
        transformer=base,
        vae=vae,
        scheduler=noise_scheduler,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
    )
    diag_pipe.set_progress_bar_config(disable=True)

    pbar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    # Training weights for combining removal and retain losses
    remove_weight = config['remove_weight']
    retain_weight = config['retain_weight']

    print(f"Loss weights: remove={remove_weight:.3f}, retain={retain_weight:.3f}")

    for iteration in pbar:
        base = accelerator.unwrap_model(model)

        vae_config_block_out_channels = diag_pipe.vae.config.block_out_channels

        t_enc_ddpm = torch.randint(
            low=0, high=ddim_steps, size=(1,), device=accelerator.device
        )
        vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

        num_channels = vae.config.latent_channels

        image_h = resolution
        image_w = resolution

        latent_h = image_h // vae_scale_factor
        latent_w = image_w // vae_scale_factor
        bsz = 1
        model_input = torch.zeros(
            (bsz, num_channels, latent_h, latent_w),
            device=vae.device,
            dtype=weight_dtype,
        )

        # (ESD) start_guidance = 3
        start_guidance = 3.5
        start_guidance = torch.tensor([start_guidance], device=accelerator.device)
        start_guidance = start_guidance.expand(model_input.shape[0])

        with accelerator.accumulate(model):
            # Shard concepts across GPUs
            rank = accelerator.process_index
            world_size = accelerator.num_processes
            valid_indices = list(range(rank, len(target_concepts), world_size))

            if len(valid_indices) == 0:
                # Fallback in case there are fewer samples than processes
                concept_idx = rank % len(target_concepts)
            else:
                # Randomly pick one index from this GPU's slice
                concept_idx = random.choice(valid_indices)

            target_text = target_concepts[concept_idx]

            # Get corresponding mapping concept
            if len(mapping_concept) > 0:
                mapping_text = (
                    mapping_concept[concept_idx]
                    if concept_idx < len(mapping_concept)
                    else mapping_concept[0]
                )
            else:
                mapping_text = ""  # Fallback to empty if no mapping concepts

            if augment_target:
                augmented_prompts = prompt_augmentation(target_text, augment=True)

                # Shard augmentation indices per rank as well
                valid_aug_indices = list(range(rank, len(augmented_prompts), world_size))
                if len(valid_aug_indices) == 0:
                    aug_idx = rank % len(augmented_prompts)
                else:
                    aug_idx = random.choice(valid_aug_indices)

                target_text_augmented = augmented_prompts[aug_idx]

                # Mapping augmentation (must use same aug_idx as target)
                if mapping_text:
                    augmented_mapping = prompt_augmentation(mapping_text, augment=True)
                    if aug_idx >= len(augmented_mapping):
                        aug_idx = aug_idx % len(augmented_mapping)
                    mapping_text_augmented = augmented_mapping[aug_idx]
                else:
                    mapping_text_augmented = ""
            else:
                target_text_augmented = target_text
                mapping_text_augmented = mapping_text

            # CLIP pooled embedding for HyperLoRA context (computed fresh, not from cache)
            hyper_emb_target = compute_clip_pooled_embeddings(
                target_text_augmented, text_encoder_one, tokenizer_one, accelerator.device
            )

            # Get mapping concept embeddings from cache
            if cache is not None and mapping_text_augmented and mapping_text_augmented in cache.mapping_prompt_to_idx:
                emb_0, pooled_emb_0, text_ids_0 = cache.get_mapping(mapping_text_augmented, accelerator.device)
            else:
                # Fallback: compute on-the-fly and add to cache for future use
                with torch.no_grad():
                    if mapping_text_augmented:
                        emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
                            mapping_text_augmented, text_encoders, tokenizers, accelerator.device
                        )
                        # Add to cache for future iterations (lazy caching)
                        if cache is not None:
                            cache.add_embedding(
                                mapping_text_augmented, emb_0, pooled_emb_0, text_ids_0,
                                embedding_type='mapping'
                            )
                    else:
                        # If no mapping concept, fall back to unconditional
                        emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
                            "", text_encoders, tokenizers, accelerator.device
                        )

            # Get target embeddings and latents from cache
            if cache is not None and target_text_augmented in cache:
                z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids = cache.get_target(
                    target_text_augmented, int(t_enc), accelerator.device
                )
            else:
                with torch.no_grad():
                    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
                        target_text_augmented, text_encoders, tokenizers, accelerator.device
                    )

            # Shard HyperLoRA timesteps across GPUs
            rank = accelerator.process_index
            world_size = accelerator.num_processes
            valid_timesteps = torch.arange(rank, hyper_train_steps, world_size, device=accelerator.device)
            if valid_timesteps.numel() == 0:
                # Fallback in case hyper_train_steps < world_size
                rtimestep = int(torch.randint(0, hyper_train_steps, (1,), device=accelerator.device))
            else:
                # Sample index into this rank’s slice
                idx = torch.randint(0, valid_timesteps.numel(), (1,), device=accelerator.device)
                rtimestep = int(valid_timesteps[idx])

            print(
                f"[Rank {rank} | Device {accelerator.device}] "
                f"idx={concept_idx} | Target: {target_text_augmented} | Mapping: {mapping_text_augmented} | Timestep: {rtimestep}"
            )

            with torch.no_grad():
                t_ddpm = t_enc_ddpm.to(accelerator.device)
                base.hyper.set_context(hyper_emb_target.to(dtype=weight_dtype),
                                       torch.tensor([rtimestep], dtype=weight_dtype, device=accelerator.device))
                _, current_timestep = base.hyper.get_context()
                base.hyper.compute_and_cache_loras(hyper_emb_target.to(dtype=weight_dtype),
                                                   current_timestep.to(dtype=weight_dtype))

                # Denoise from noise to get latent at t_ddpm
                z, latent_image_ids, timestep = latent_sample(
                    model, noise_scheduler, 1, model_input.shape[1], resolution, resolution,
                    emb_p.to(accelerator.device), pooled_emb_p.to(accelerator.device),
                    text_ids_p.to(accelerator.device), start_guidance,
                    int(ddim_steps), stop_at_step=int(t_ddpm.item()),
                )

                # Reference noise predictions without LoRA
                with base.hyper.no_lora():
                    e_0 = predict_noise(
                        model, z, emb_0.to(dtype=weight_dtype), pooled_emb_0.to(dtype=weight_dtype),
                        text_ids_0, latent_image_ids, guidance=start_guidance,
                        timesteps=timestep, CPU_only=True,
                    )
                    e_p = predict_noise(
                        model, z, emb_p.to(dtype=weight_dtype), pooled_emb_p.to(dtype=weight_dtype),
                        text_ids_p, latent_image_ids, guidance=start_guidance,
                        timesteps=timestep, CPU_only=True,
                    )

            base.hyper.set_context(hyper_emb_target, torch.tensor([rtimestep], device=accelerator.device))
            _, current_timestep = base.hyper.get_context()
            base.hyper.compute_and_cache_loras(hyper_emb_target, current_timestep)
            base.hyper.retain_grad_for_cached_lora()

            # Noise prediction with LoRA active
            e_n = predict_noise(
                model, z, emb_p.to(dtype=weight_dtype), pooled_emb_p.to(dtype=weight_dtype),
                text_ids_p, latent_image_ids, guidance=start_guidance,
                timesteps=timestep, CPU_only=True,
            )
            e_0.requires_grad = False
            e_p.requires_grad = False

            # Auxiliary loss: push LoRA output toward mapping concept direction
            loss_aux = criterion(
                e_n.float().to(accelerator.device),
                (e_0 - negative_guidance * (e_p - e_0)).float().to(accelerator.device)
            )

            accelerator.backward(loss_aux)

            # Simulated SGD target from cached LoRA gradients
            grads_flat_t = base.hyper.flatten_cached_grads_from_cache()
            if grads_flat_t is None:
                raise RuntimeError(
                    "No gradients found in cached LoRA tensors.")

            grads_flat_t = (-1.0 * internal_lr) * grads_flat_t.detach()

            _, current_timestep = accelerator.unwrap_model(model).hyper.get_context()
            base.hyper.set_context(hyper_emb_target, current_timestep)
            base.hyper.compute_and_cache_loras(hyper_emb_target, current_timestep)
            tensors_flat_t = base.hyper.flatten_cached_from_cache()

            base.hyper.set_context(hyper_emb_target, (current_timestep + 1))
            base.hyper.compute_and_cache_loras(hyper_emb_target, (current_timestep + 1))
            tensors_flat_t1 = base.hyper.flatten_cached_from_cache()

            # Removal loss: match hypernetwork step to simulated SGD step
            delta_live = tensors_flat_t1 - tensors_flat_t
            loss_remove = remove_weight * criterion(delta_live, grads_flat_t)
            accelerator.backward(loss_remove)

            loss_remove_log = loss_remove.clone().detach()

            if accelerator.sync_gradients:
                optimizer_remove.step()
                optimizer_remove.zero_grad(set_to_none=True)
                scheduler_remove.step()

            loss_retain_total = torch.tensor(0.0, device=accelerator.device)
            if len(retain_embeddings) > 0:
                for _ in range(retain_steps_per_remove):
                    # Sample multiple retain concepts
                    num_retain_samples = min(min_retain_sample, len(retain_embeddings))
                    sampled_retain_embs = random.sample(retain_embeddings, num_retain_samples)

                    # Batch process retain concepts
                    batch_retain_embs = (
                        torch.stack(sampled_retain_embs, dim=0)
                            .to(device=accelerator.device, dtype=torch.bfloat16)
                    )
                    hyper = base.hyper
                    B = batch_retain_embs.shape[0]
                    perm = torch.randperm(B, device=batch_retain_embs.device)
                    batch_prompts = batch_retain_embs[perm]

                    # Compute LoRAs at t=0
                    dtype = next(hyper.parameters()).dtype  # hyper’s param dtype (bf16 if you casted it)

                    hyper.compute_and_cache_loras(
                        batch_prompts.to(dtype=dtype).to(dtype=weight_dtype),
                        torch.zeros(B, device=accelerator.device, dtype=dtype),
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

                    # Retain loss: minimize LoRA weight change across timesteps
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

        # Gather loss across devices
        with torch.no_grad():
            loss_retain_reduced = accelerator.gather(loss_retain_log).mean()
            loss_remove_reduced = accelerator.gather(loss_remove_log).mean()
        losses.append(float(loss_remove_reduced.item() + loss_retain_reduced.item()))

        if accelerator.is_main_process and use_wandb:
            current_lr_remove = optimizer_remove.param_groups[0]['lr']
            current_lr_retain = optimizer_retain.param_groups[0]['lr']
            wandb.log({
                "loss_retain": float(loss_retain_reduced.item()),
                "loss_remove": float(loss_remove_reduced.item()),
                "learning_rate_remove": current_lr_remove,
                "learning_rate_retain": current_lr_retain,
            }, step=iteration)

        if is_main:
            pbar.set_postfix({
                "retain": f"{float(loss_retain_reduced.item()):.3e}",
                "remove": f"{float(loss_remove_reduced.item()):.3e}"
            })

        # Generate sample images periodically
        if is_main and use_wandb and (iteration + 1) % diagnostic_freq == 0:
            # Generate images for diagnostic prompts from config
            for diag_idx, diag_prompt in enumerate(diagnostic_prompts):

                # Get diagnostic embeddings for HyperLoRA (CLIP-only, computed fresh - not from cache)
                hyper_emb_diag = compute_clip_pooled_embeddings(
                    diag_prompt, text_encoder_one, tokenizer_one, accelerator.device
                )

                cached_text_emb = compute_text_embeddings(
                    diag_prompt, text_encoders, tokenizers, accelerator.device
                )

                diag_time_steps = [0, hyper_train_steps]

                device = accelerator.device

                # Ensure pipeline uses the live transformer
                base = accelerator.unwrap_model(model)
                diag_pipe.transformer = base  # make sure pipe uses current model

                imgs_per_prompt = []

                for h_step in diag_time_steps:
                    h_step_tensor = torch.tensor([h_step], device=device)

                    base.hyper.set_context(hyper_emb_diag, h_step_tensor)
                    base.hyper.compute_and_cache_loras(hyper_emb_diag, h_step_tensor)

                    imgs = generate_one_image_from_prompt(
                        prompt=diag_prompt,
                        transformer=base,
                        vae=vae,
                        noise_scheduler=noise_scheduler,
                        text_encoders=text_encoders,
                        tokenizers=tokenizers,
                        height=resolution,
                        width=resolution,
                        num_inference_steps=ddim_steps,
                        weight_dtype=weight_dtype,
                        seed=seed,
                        cached_embeddings=cached_text_emb,
                    )

                    imgs_per_prompt.append(imgs)

                if len(imgs_per_prompt) > 0:
                    row_tensors = []

                    for imgs in imgs_per_prompt:
                        if imgs is None:
                            continue

                        # Take the first image (assumed to be PIL.Image)
                        imgs = to_tensor(imgs).clamp(0, 1)
                        row_tensors.append(imgs)

                    if len(row_tensors) > 0:
                        row = torch.cat(row_tensors, dim=2)
                        safe_key = diag_prompt.replace(" ", "_").replace(",", "")[:50]

                        wandb.log(
                            {
                                f"diagnostic_{diag_idx}_{safe_key}": wandb.Image(
                                    to_pil_image(row),
                                    caption=f"{diag_prompt} | hyper steps: {diag_time_steps}",
                                )
                            },
                            step=iteration,
                        )
        # Save model
        accelerator.wait_for_everyone()
        if is_main and ((iteration % 100 == 0) or (iteration == max_train_steps - 1)):
            print(f"Final loss: {losses[-1]:.3e}")
            print(f"Average loss: {sum(losses) / len(losses):.3e}")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(final_save_path, exist_ok=True)

            # Save LoRA weights with embedded config
            model_unwrapped = accelerator.unwrap_model(model)
            lora_state_dict = {k: v.cpu() for k, v in model_unwrapped.state_dict().items() if ".hyper_lora." in k}
            lora_path = os.path.join(final_save_path, f"hyper_lora_em_{iteration}.pth")
            accelerator.save({'state_dict': lora_state_dict, 'config': checkpoint_config}, lora_path)
            print(f"Model saved to: {lora_path}")

        # Save config
        config_save = {
            "config_name": config_name,
            "concepts": concepts,
            "mapping_concept": mapping_concept,
            "retain_csv_path": retain_csv_path,
            "augment_target": augment_target,
            "augment_retain": augment_retain,
            "num_retain_prompts": len(retain_prompts),
            "rank": rank,
            "learning_rate_remove": learning_rate_remove,
            "learning_rate_retain": learning_rate_retain,
            "max_train_steps": max_train_steps,
            "hyper_train_steps": hyper_train_steps,
            "final_loss": losses[-1],
            "average_loss": sum(losses) / len(losses),
        }

        with open(os.path.join(final_save_path, "train_config.json"), "w") as f:
            json.dump(config_save, f, indent=2)

    # Save cache if it was modified (lazy caching added new embeddings)
    if is_main and cache is not None and cache.dirty:
        print(f"[Cache] Saving updated cache with {len(cache.mapping_prompts)} mapping prompts...")
        cache.save(cache_path)

    if is_main and use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
