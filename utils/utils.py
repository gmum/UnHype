import torch
import numpy as np


def set_seed(seed: int):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def print_trainable_parameters(model, max_params: int = 50):
    print(f"First {max_params} layers with requires_grad == True:")
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
            count += 1
            if count >= max_params:
                break


def load_model_from_config(*args, **kwargs):
    raise NotImplementedError("Stable Diffusion backend is not yet implemented")


def get_models(*args, **kwargs):
    raise NotImplementedError("Stable Diffusion backend is not yet implemented")


def apply_lora_to_model(*args, **kwargs):
    raise NotImplementedError("Stable Diffusion backend is not yet implemented")
