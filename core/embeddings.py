import torch
from transformers import CLIPTextModel, CLIPTokenizer

_nv_embed_module = None


def _load_nv_embed_module():
    global _nv_embed_module
    if _nv_embed_module is None:
        from utils import nv_embed_utils
        _nv_embed_module = nv_embed_utils
    return _nv_embed_module


def setup_embedding_model(embedding_model: str, use_pooler: bool, device: torch.device):
    if embedding_model == 'nv_embed':
        nv_mod = _load_nv_embed_module()
        nv_model, nv_tokenizer = nv_mod.load_nv_embed_model(device=device, dtype=torch.float16)
        clip_size = nv_mod.NV_EMBED_DIM  # 4096

        def embed_fn(prompt):
            return nv_mod.compute_nv_embed([prompt], nv_model, nv_tokenizer, device, batch_size=1)

        return embed_fn, clip_size

    elif embedding_model == 'clip_huge':
        tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        clip_model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device).eval()
        clip_size = 1024

        def embed_fn(prompt):
            inputs = tokenizer(
                prompt,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device).input_ids
            with torch.no_grad():
                return clip_model(inputs).pooler_output.detach()

        return embed_fn, clip_size

    else:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        clip_size = 768 if use_pooler else 512

        def embed_fn(prompt):
            inputs = tokenizer(
                prompt,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device).input_ids
            with torch.no_grad():
                if use_pooler:
                    return clip_model(inputs).pooler_output.detach()
                else:
                    return clip_model(inputs).last_hidden_state.detach()

        return embed_fn, clip_size
