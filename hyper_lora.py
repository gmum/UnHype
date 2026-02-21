import weakref
from typing import List, Dict, Tuple, Optional
from contextlib import contextmanager

import torch
import math
import os
import torch.nn.functional as F
from functools import partial
import torch.nn as nn


class HypernetworkManager(nn.Module):
    def __init__(self):
        super().__init__()
        self.hyper_layers = nn.ModuleList()
        self.layer_name_to_idx = {}
        self.lora_weights_cache = {}
        self.current_context = {'clip_emb': None, 'timestep': None}
        self.auto_mode = False
        self.lora_enabled = True

    def enable_lora(self):
        self.lora_enabled = True

    def disable_lora(self):
        self.lora_enabled = False

    @contextmanager
    def no_lora(self):
        old_state = self.lora_enabled
        self.lora_enabled = False
        try:
            yield
        finally:
            self.lora_enabled = old_state

    def add_hyperlora(self, name: str, hyper_lora):
        idx = len(self.hyper_layers)
        self.hyper_layers.append(hyper_lora)
        self.layer_name_to_idx[name] = idx

    def set_context(self, clip_emb, timestep):
        self.current_context['clip_emb'] = clip_emb
        self.current_context['timestep'] = timestep

    def get_context(self):
        return self.current_context['clip_emb'], self.current_context['timestep']

    def compute(self, clip_emb, timestep):
        for name, idx in self.layer_name_to_idx.items():
            hyper = self.hyper_layers[idx]
            _ = hyper.get_lora_matrices(clip_emb, timestep)

    def compute_and_cache_loras(self, clip_emb, timestep):
        self.lora_weights_cache.clear()
        for name, idx in self.layer_name_to_idx.items():
            hyper = self.hyper_layers[idx]
            x_alpha, x_L, x_R = hyper.get_lora_matrices(clip_emb, timestep)
            self.lora_weights_cache[name] = (x_alpha, x_L, x_R)

    def get_cached_lora(self, layer_name):
        return self.lora_weights_cache.get(layer_name, None)

    def flatten_cached_from_cache(self):
        vecs = []
        for name, _ in self.layer_name_to_idx.items():
            for w in self.get_cached_lora(name):
                vecs.append(w.reshape(-1))
        return None if not vecs else torch.cat(vecs, dim=0)

    def flatten_cached_grads_from_cache(self):
        grads = []
        for name, idx in self.layer_name_to_idx.items():
            for w in self.get_cached_lora(name):
                g = getattr(w, "grad", None)
                if g is not None:
                    grads.append(g.clone().reshape(-1))
                    w.grad = None
        return None if not grads else torch.cat(grads, dim=0)

    def retain_grad_for_cached_lora(self):
        for name, idx in self.layer_name_to_idx.items():
            for w in self.get_cached_lora(name):
                if hasattr(w, "retain_grad"):
                    w.retain_grad()


class TimeFourier(nn.Module):
    def __init__(self, T, L=16, dtype=torch.float32):
        super().__init__()
        k = torch.linspace(0, L - 1, L, dtype=dtype)
        freqs = (2.0 * math.pi / T) * torch.pow(torch.tensor(2.0), k)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(dtype=self.freqs.dtype).unsqueeze(-1)
        w = self.freqs
        angles = t * w
        return torch.cat([angles.cos(), angles.sin()], dim=-1)


class HyperLora(nn.Module):

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            rank: int = 4,
            clip_size: int = 768,
            alpha_init: int = 16.0,
            time_embedd: int = 32,
            use_scaling=True,
            original_linear=None,
            train_steps: int = None,
            use_orig_concat: bool = True,
            dtype: torch.dtype = torch.float32,
            internal_size: int = 100,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.clip_size = clip_size
        self.original = original_linear
        self.train_steps = train_steps
        self.use_orig_concat = use_orig_concat
        self.dtype = dtype
        std_dev = 1 / (rank ** 0.5)
        self.register_buffer(
            "xL_const_flat", torch.randn(1, in_dim * rank, dtype=self.dtype) * std_dev
        )
        self.register_buffer(
            "xR_const_flat", torch.zeros(1, out_dim * rank, dtype=self.dtype)
        )
        self.register_buffer(
            "alpha_b", torch.tensor(alpha_init, dtype=self.dtype)
        )

        hyper_input_size = clip_size + time_embedd + (out_dim if use_orig_concat else 0)

        self.left_head = nn.Sequential(
            nn.Linear(hyper_input_size, internal_size),
            nn.ReLU(inplace=True),
            nn.Linear(internal_size, in_dim * rank),
        ).to(dtype=self.dtype)
        self.right_head = nn.Sequential(
            nn.Linear(hyper_input_size, internal_size),
            nn.ReLU(inplace=True),
            nn.Linear(internal_size, out_dim * rank),
        ).to(dtype=self.dtype)

        nn.init.zeros_(self.right_head[-1].weight)
        nn.init.zeros_(self.right_head[-1].bias)

        self.time_feat = TimeFourier(T=self.train_steps + 1, dtype=self.dtype)

        self.use_scaling = use_scaling
        if self.use_scaling:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=self.dtype))

    def forward_linear_L(self, emb, t):

        return self.xL_const_flat + t[:, None] / self.train_steps * self.left_head(emb)

    def forward_linear_R(self, emb, t):
        return self.xR_const_flat + t[:, None] / self.train_steps * self.right_head(emb)

    def forward_alpha(self, t):
        return self.alpha_b + t[:, None] / self.train_steps * self.alpha

    def get_lora_matrices(self, clip, t):
        t_feats = self.time_feat(t).to(dtype=self.dtype)

        emb = clip
        if self.use_orig_concat and clip.shape[-1] == self.clip_size:
            dummy_orig = torch.zeros(clip.shape[0], self.out_dim, device=clip.device, dtype=clip.dtype)
            emb = torch.cat([emb, dummy_orig], dim=-1)

        emb = torch.cat([emb, t_feats], dim=-1)

        if self.use_scaling:
            alpha = self.forward_alpha(t)
            x_L = alpha * self.forward_linear_L(emb, t)
        else:
            x_L = self.forward_linear_L(emb, t)
        x_R = self.forward_linear_R(emb, t)

        if alpha.requires_grad:
            alpha.retain_grad()
        if x_L.requires_grad:
            x_L.retain_grad()
        if x_R.requires_grad:
            x_R.retain_grad()

        x_L = x_L.view(-1, self.in_dim, self.rank)
        x_R = x_R.view(-1, self.rank, self.out_dim)

        return alpha, x_L, x_R

    def forward(self, x, clip, t):
        alpha, x_L, x_R = self.get_lora_matrices(clip, t)

        ret = (x @ x_L) @ x_R
        return ret


class HyperLoRALinear(nn.Module):

    def __init__(
            self,
            original_linear: nn.Linear,
            clip_size: int = 768,
            rank: int = 1,
            alpha: int = 16,
            layer_name: str = None,
            train_steps: int = None,
            use_orig_concat: bool = False,
            dtype: torch.dtype = torch.float32,
            internal_size: int = 100,
    ):
        super().__init__()
        self.original = original_linear
        self.hyper_lora = HyperLora(
            original_linear.in_features,
            original_linear.out_features,
            rank,
            clip_size,
            alpha,
            train_steps=train_steps,
            original_linear=original_linear,
            use_orig_concat=use_orig_concat,
            dtype=dtype,
            internal_size=internal_size,
        )
        self.parent_model = None
        self.layer_name = layer_name

    def set_parent_model(self, model):
        self.parent_model = weakref.ref(model)

    def forward(self, x):
        parent = self.parent_model()

        if hasattr(parent, 'hyper') and parent.hyper is not None:
            if not parent.hyper.lora_enabled:
                return self.original(x)

            if parent.hyper.auto_mode:
                clip_embedding, timestep = parent.hyper.get_context()
                if clip_embedding is None:
                    print("WARNING: clip_embedding is None in auto mode")
                    return self.original(x)

                orig = self.original(x)
                if self.hyper_lora.use_orig_concat:
                    hyper_input = torch.cat([clip_embedding, orig], dim=-1)
                else:
                    hyper_input = clip_embedding

                lora_fp = self.hyper_lora(x.float(), hyper_input.float(), timestep.float()).float()

                return orig + lora_fp.to(dtype=orig.dtype)
            else:
                lora_weights = parent.hyper.get_cached_lora(self.layer_name)
                if lora_weights is None:
                    return self.original(x)
                alpha, x_L, x_R = lora_weights

                batch_size = x.shape[0]
                if x_L.shape[0] == 1 and batch_size > 1:
                    x_L = x_L.expand(batch_size, -1, -1)
                    x_R = x_R.expand(batch_size, -1, -1)

                orig_out = self.original(x)
                x_fp = x.float()
                xL_fp = x_L.float()
                xR_fp = x_R.float()

                lora_fp = (x_fp @ xL_fp) @ xR_fp

                return orig_out + lora_fp.to(dtype=orig_out.dtype)
        else:
            if not hasattr(parent, 'current_conditioning'):
                print("WARNING: parent model has neither 'hyper' nor 'current_conditioning'")
                return self.original(x)

            clip_embedding = parent.current_conditioning
            timestep = getattr(parent, 'time_step', None)

            if clip_embedding is None or timestep is None:
                return self.original(x)

            orig = self.original(x)
            if self.hyper_lora.use_orig_concat:
                hyper_input = torch.cat([clip_embedding, orig], dim=-1)
            else:
                hyper_input = clip_embedding
            orig_out = orig
            lora_out = self.hyper_lora(x, hyper_input, timestep)

            return orig_out + lora_out.to(dtype=orig_out.dtype)


def inject_hyper_lora(
        module: nn.Module, target_modules: List[str], hyper_lora_factory, name: str = ""
):
    hyper_lora_layers = []

    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name

        if isinstance(child, nn.Linear) and any(
                full_name.endswith(t) for t in target_modules
        ):
            device = next(child.parameters()).device
            hyper_lora_layer = hyper_lora_factory(child).to(device)
            hyper_lora_layer.layer_name = full_name
            setattr(module, child_name, hyper_lora_layer)
            hyper_lora_layers.append((full_name, hyper_lora_layer))
        else:
            child_layers = inject_hyper_lora(
                child, target_modules, hyper_lora_factory, full_name
            )
            hyper_lora_layers.extend(child_layers)

    return hyper_lora_layers


def inject_hyper_lora_nsfw(module, hyper_lora_factory, name=""):
    hyper_lora_layers = []

    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name

        if (
                full_name.startswith("out.")
                or "attn2" in full_name
                or "time_embed" in full_name
        ):
            continue

        if isinstance(child, nn.Linear):
            print(f"Injecting HyperLoRA into: {full_name}")
            device = next(child.parameters()).device
            hyper_lora_layer = hyper_lora_factory(child).to(device)
            hyper_lora_layer.layer_name = full_name
            setattr(module, child_name, hyper_lora_layer)
            hyper_lora_layers.append((full_name, hyper_lora_layer))
        else:
            child_layers = inject_hyper_lora_nsfw(child, hyper_lora_factory, full_name)
            hyper_lora_layers.extend(child_layers)

    return hyper_lora_layers
