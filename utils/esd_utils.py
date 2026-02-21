import torch
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import inspect

from typing import Any, Callable, Dict, List, Optional, Union


def flux_pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents


def flux_unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


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


@torch.no_grad()
def latent_sample(transformer, scheduler, batch_size, num_channels_latents, height, width, prompt_embeds,
                  pooled_prompt_embeds, text_ids, guidance, num_inference_steps, stop_at_step=None, latents=None):
    height = int(height) // 8
    width = int(width) // 8
    shape = (batch_size, num_channels_latents, height, width)

    if latents is None:
        latents = randn_tensor(shape, generator=None, dtype=torch.bfloat16)
    latents = flux_pack_latents(latents, batch_size, num_channels_latents, height, width)
    latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, transformer.device,
                                                 torch.bfloat16)

    image_seq_len = latents.shape[1]
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )

    if text_ids.dim() == 3:
        text_ids = text_ids[0]
    elif text_ids.dim() == 2:
        text_ids = text_ids
    else:
        raise ValueError(f"Unexpected txt_ids shape: {text_ids.shape}")

    text_ids = text_ids.to(dtype=torch.bfloat16)

    timesteps = None
    timesteps_tensor, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        transformer.device,
        timesteps,
        sigmas,
        mu=mu,
    )

    latents = latents.to(transformer.device).bfloat16()
    pooled_prompt_embeds = pooled_prompt_embeds.bfloat16()
    prompt_embeds = prompt_embeds.bfloat16()
    text_ids = text_ids.bfloat16()

    timestep = None
    for i, t in enumerate(timesteps_tensor):
        if stop_at_step is not None and i >= stop_at_step:
            timestep = t.expand(latents.shape[0]).to(torch.bfloat16)
            return latents, latent_image_ids, timestep

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

    return latents, latent_image_ids, timestep


def predict_noise(transformer, latent_code, prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids, guidance,
                  timesteps, CPU_only=False):

    if CPU_only:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:1")
    if timesteps.dtype in (torch.bfloat16, torch.float16):
        timesteps = timesteps.to(torch.float32)

    if text_ids.dim() == 3:
        text_ids = text_ids[0]
    elif text_ids.dim() == 2:
        text_ids = text_ids
    else:
        raise ValueError(f"Unexpected txt_ids shape: {text_ids.shape}")

    text_ids = text_ids.to(dtype=torch.bfloat16)

    model_pred= transformer(
        hidden_states=latent_code.to(device),
        timestep=(timesteps / 1000).to(device),
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds.to(device),
        encoder_hidden_states=prompt_embeds.to(device),
        txt_ids=text_ids.to(device),
        img_ids=latent_image_ids.to(device),
        return_dict=False,
    )


    return model_pred[0]