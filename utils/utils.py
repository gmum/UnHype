import os
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch
import numpy as np
from ldm.models.diffusion.ddim import DDIMSampler
from transformers import set_seed as hf_set_seed

_CKPT_HF_REPOS = {
    "sd-v1-4.ckpt": ("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt"),
}

def set_seed(seed: int):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    hf_set_seed(seed)

def _ensure_checkpoint(ckpt_path: str) -> str:
    if os.path.isfile(ckpt_path):
        return ckpt_path
    filename = os.path.basename(ckpt_path)
    if filename in _CKPT_HF_REPOS:
        repo_id, repo_filename = _CKPT_HF_REPOS[filename]
        print(f"Checkpoint not found at {ckpt_path}, downloading from {repo_id}...")
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        hf_hub_download(repo_id=repo_id, filename=repo_filename, local_dir=os.path.dirname(ckpt_path))
        print(f"Downloaded to {ckpt_path}")
    return ckpt_path


def load_model_from_config(config_path: str, ckpt_path: str, device: str = "cpu"):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    ckpt_path = _ensure_checkpoint(ckpt_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

def get_models(config_path: str, ckpt_path: str, device: str):
    model_orig = load_model_from_config(config_path, ckpt_path, device)
    sampler_orig = DDIMSampler(model_orig)
    model = load_model_from_config(config_path, ckpt_path, device)
    sampler = DDIMSampler(model)
    return model_orig, sampler_orig, model, sampler


def print_trainable_parameters(model, max_params: int = 50):
    print(f"First {max_params} layers with requires_grad == True:")
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
            count += 1
            if count >= max_params:
                break

def apply_lora_to_model(model, lora_state_dict, alpha=4):
    model_sd = model.state_dict()

    for lora_A_key in [k for k in lora_state_dict if k.endswith(".lora.A")]:
        prefix = lora_A_key[:-len(".lora.A")]

        A_key = prefix + ".lora.A"
        B_key = prefix + ".lora.B"
        W_key = prefix + ".weight"  # the original weight in the model


        A = lora_state_dict[A_key].to(model_sd[W_key].device)
        B = lora_state_dict[B_key].to(model_sd[W_key].device)

        delta = A.matmul(B)
        delta = delta.T
        
        model_sd[W_key] = model_sd[W_key] + alpha * delta

    model.load_state_dict(model_sd, strict=False)
