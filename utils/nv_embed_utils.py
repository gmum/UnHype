import torch
import torch.nn.functional as F
from typing import List, Tuple, Union


NV_EMBED_INSTRUCTION = "Instruct: Represent the visual content described\nQuery: "
NV_EMBED_MODEL_NAME = "nvidia/NV-Embed-v2"
NV_EMBED_DIM = 4096


def load_nv_embed_model(
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        NV_EMBED_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()

    return model, None


def compute_nv_embed(
    prompts: Union[str, List[str]],
    model,
    tokenizer,
    device: Union[str, torch.device] = "cuda",
    batch_size: int = 8,
    instruction: str = NV_EMBED_INSTRUCTION,
    max_length: int = 4096,
) -> torch.Tensor:
    if isinstance(prompts, str):
        prompts = [prompts]

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            embeddings = model.encode(
                batch,
                instruction=instruction,
                max_length=max_length,
            )

            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings, device=device, dtype=torch.float32)
            else:
                embeddings = embeddings.to(device=device, dtype=torch.float32)

            embeddings = F.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings)

    result = torch.cat(all_embeddings, dim=0)
    return result


def compute_nv_embed_single(
    prompt: str,
    model,
    tokenizer,
    device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    return compute_nv_embed([prompt], model, tokenizer, device, batch_size=1)
