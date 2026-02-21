import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import argparse
import math
import time
import inspect
from typing import List, Optional, Union

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from diffusers import AutoencoderKL, FluxPipeline, FluxTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTokenizerFast, PretrainedConfig, T5TokenizerFast
from huggingface_hub import login

from tools.prompt_process import encode_prompt
from tools.scheduler_process import FlowMatchEulerDiscreteScheduler
from utils.esd_utils import flux_pack_latents, _prepare_latent_image_ids
from core.prompts import coerce_prompt


def _setup_hf_env():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token)
    else:
        print("Warning: HF_TOKEN not set.")


def load_text_encoders(class_one, class_two, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None, variant=None
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=None, variant=None
    )
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


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    # Adapted from diffusers
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@torch.no_grad()
def inference_latent_sample(transformer, scheduler, batch_size, num_channels_latents, height, width, prompt_embeds,
                  pooled_prompt_embeds, text_ids, guidance, num_inference_steps, generator=None, latents=None):

    height = int(height) // 8
    width = int(width) // 8
    shape = (batch_size, num_channels_latents, height, width)

    if latents is None:
        latents = randn_tensor(shape, generator=generator, dtype=torch.bfloat16, device=transformer.device)

    latents = flux_pack_latents(latents, batch_size, num_channels_latents, height, width)

    latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, transformer.device,
                                                 torch.bfloat16)

    latents = latents.to(transformer.device).bfloat16()
    pooled_prompt_embeds = pooled_prompt_embeds.bfloat16()
    prompt_embeds = prompt_embeds.bfloat16()
    if text_ids.dim() == 3:
        text_ids = text_ids[0]
    elif text_ids.dim() == 2:
        text_ids = text_ids
    else:
        raise ValueError(f"Unexpected txt_ids shape: {text_ids.shape}")

    text_ids = text_ids.to(dtype=torch.bfloat16)

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps = None
    timesteps_tensor, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        transformer.device,
        timesteps,
        sigmas,
        mu=mu,
    )

    for i, t in enumerate(timesteps_tensor):
        timestep = t.expand(latents.shape[0]).to(torch.bfloat16)

        noise_pred = transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )

        if isinstance(noise_pred, (tuple, list)):
            noise_pred = noise_pred[0]

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents, latent_image_ids


def compute_text_embeddings_short(prompts, text_encoders, tokenizers, device="cpu"):
    if isinstance(prompts, str):
        prompts = [prompts]

    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
        text_encoders, tokenizers, prompts, 256
    )
    return (
        prompt_embeds.to(device),
        pooled_prompt_embeds.to(device),
        text_ids.to(device),
    )

@torch.no_grad()
def generate_one_image_from_prompt(
    prompt: str,
    *,
    transformer,
    vae,
    noise_scheduler,
    text_encoders,
    tokenizers,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 28,
    weight_dtype: torch.dtype = torch.bfloat16,
    cached_embeddings = None,
    seed: int | None = None,
):

    device = transformer.device
    bsz = 1

    prompts = [prompt]
    if cached_embeddings is None:
        emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings_short(
            prompts, text_encoders, tokenizers, transformer.device
        )
    else:
        emb_p, pooled_emb_p, text_ids_p = cached_embeddings

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))

    num_channels = vae.config.latent_channels

    image_h = 512
    image_w = 512

    latent_h = image_h // vae_scale_factor
    latent_w = image_w // vae_scale_factor

    model_input = torch.zeros(
        (bsz, num_channels, latent_h, latent_w),
        device=vae.device,
        dtype=weight_dtype,
    )

    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor

    start_guidance = 3.5
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])

    generator = torch.Generator(device).manual_seed(seed)

    with torch.no_grad():
        z, latent_image_ids = inference_latent_sample(
            transformer,
            noise_scheduler,
            bsz,
            num_channels,
            height,
            width,
            emb_p.to(device),
            pooled_emb_p.to(device),
            text_ids_p.to(device),
            start_guidance,
            int(num_inference_steps),
            generator=generator,
        )
    bsz, seq_len, ch = z.shape
    side = int(math.isqrt(seq_len))
    assert side * side == seq_len, f"seq_len={seq_len} not square"

    latent_h = side * 2
    latent_w = side * 2
    img_h = latent_h * vae_scale_factor
    img_w = latent_w * vae_scale_factor

    z = FluxPipeline._unpack_latents(
        z,
        height=img_h,
        width=img_w,
        vae_scale_factor=vae_scale_factor,
    )

    shift = vae.config.shift_factor
    scale = vae.config.scaling_factor

    z = z / scale + shift

    z = z.to(device=vae.device, dtype=torch.float32)
    vae = vae.to(device=vae.device, dtype=torch.float32)

    decoded = vae.decode(z).sample
    img = (decoded / 2 + 0.5).clamp(0, 1)

    import numpy as np
    from PIL import Image

    img = img[0].permute(1, 2, 0).float().cpu().numpy()
    img = (img * 255).round().astype(np.uint8)
    return Image.fromarray(img)


if __name__ == "__main__":
    _setup_hf_env()
    parser = argparse.ArgumentParser(description="Generate images with base Flux from CSV")
    parser.add_argument("--csv_path", type=str, default="data/I2P_prompts_4703.csv")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--output_dir", type=str, default="generated_bare_flux")
    parser.add_argument("--save_folder", type=str, default="images")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--nudity", type=bool, default=True)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--n_images", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer_one = CLIPTokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None,
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, None, subfolder="text_encoder_2"
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two, args)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=None,
        variant=None,
    )

    weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    text_encoder_one = text_encoder_one.to(device=device, dtype=weight_dtype).eval()
    text_encoder_two = text_encoder_two.to(device=device, dtype=weight_dtype).eval()

    vae = vae.to(device=device, dtype=torch.float32).eval()

    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype,
        subfolder="transformer", revision=None, variant=None
    ).to(device)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]


    df = pd.read_csv(args.csv_path, index_col=0)
    if args.nudity and "nudity_percentage" in df.columns:
        df["nudity_percentage"] = pd.to_numeric(df["nudity_percentage"], errors="coerce")
        df = df[df["nudity_percentage"].gt(0)]
        df = df.sort_values(by="nudity_percentage", ascending=False)

    save_dir = os.path.join(args.output_dir, args.save_folder)
    os.makedirs(save_dir, exist_ok=True)

    images_generated = 0
    for image_id, row in tqdm(df.iterrows(), total=len(df)):
        if args.n_images is not None and images_generated >= args.n_images:
            break
        image_path = os.path.join(save_dir, f"{image_id:05d}.png")
        if os.path.exists(image_path):
            continue

        prompt = coerce_prompt(row.get("prompt", ""))
        if not isinstance(prompt, str) or not prompt.strip():
            print(f"Skip [{image_id}] empty prompt")
            continue

        seed = int(row.get("evaluation_seed", 0))

        start = time.time()

        image = generate_one_image_from_prompt(
            prompt=prompt,
            transformer=transformer,
            vae=vae,
            noise_scheduler=noise_scheduler,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.num_inference_steps,
            weight_dtype=weight_dtype,
            seed=seed,  # uses your per-row seed
        )

        image.save(image_path)

        images_generated += 1
        end = time.time()
        print(f"Prompt [{prompt}] processed in {end - start:.2f} seconds. Saved to {image_path}")
