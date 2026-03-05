import torch
from typing import Optional

@torch.no_grad()
def sample_model(
    model,
    sampler,
    conditioning,
    h: int,
    w: int,
    ddim_steps: int,
    scale: float,
    ddim_eta: float,
    start_code: Optional[torch.Tensor] = None,
    n_samples: int = 1,
    t_start: int = -1,
    till_T: Optional[int] = None,
    verbose: bool = True,
):
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])

    shape = [4, h // 8, w // 8]
    samples_ddim, _ = sampler.sample(
        S=ddim_steps,
        conditioning=conditioning,
        batch_size=n_samples,
        shape=shape,
        x_T=start_code,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc,
        eta=ddim_eta,
        verbose_iter=verbose,
        verbose=verbose,
        t_start=t_start,
        till_T=till_T,
    )
    return samples_ddim
