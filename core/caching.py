import os
import random

import torch
from tqdm import tqdm


class HyperCache:

    def __init__(
        self,
        prompts: list = None,
        embed_fn=None,
        device: torch.device = None,
        batch_size: int = 8,
        embed_model_name: str = "unknown",
    ):
        self.prompts = []
        self.prompt_to_idx = {}
        self.embeddings = None
        self.dirty = False
        self.embed_model_name = embed_model_name

        if prompts and embed_fn is not None:
            self.prompts = list(prompts)
            self.prompt_to_idx = {p: i for i, p in enumerate(self.prompts)}

            all_embeddings = []
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"Caching {embed_model_name} embeddings"):
                batch = prompts[i:i+batch_size]
                batch_embs = [embed_fn(p).cpu() for p in batch]
                all_embeddings.extend(batch_embs)

            self.embeddings = torch.cat([e if e.dim() == 2 else e.unsqueeze(0) for e in all_embeddings], dim=0)
            self._print_memory_usage()

    def get(self, prompt: str, device: torch.device) -> torch.Tensor:
        idx = self.prompt_to_idx[prompt]
        return self.embeddings[idx:idx+1].to(device)

    def get_by_idx(self, idx: int, device: torch.device) -> torch.Tensor:
        return self.embeddings[idx:idx+1].to(device)

    def get_batch(self, prompts: list, device: torch.device) -> torch.Tensor:
        indices = [self.prompt_to_idx[p] for p in prompts]
        return self.embeddings[indices].to(device)

    def get_batch_by_idx(self, indices: list, device: torch.device) -> torch.Tensor:
        return self.embeddings[indices].to(device)

    def sample_batch(self, n: int, device: torch.device) -> torch.Tensor:
        if self.embeddings is None or len(self.prompts) == 0:
            return None
        n = min(n, len(self.prompts))
        indices = random.sample(range(len(self.prompts)), n)
        return self.embeddings[indices].to(device)

    def has(self, prompt: str) -> bool:
        return prompt in self.prompt_to_idx

    def add(self, prompt: str, embedding: torch.Tensor):
        if prompt in self.prompt_to_idx:
            return
        idx = len(self.prompts)
        self.prompts.append(prompt)
        self.prompt_to_idx[prompt] = idx
        emb = embedding.cpu()
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = torch.cat([self.embeddings, emb], dim=0)
        self.dirty = True

    def __len__(self) -> int:
        return len(self.prompts)

    def __contains__(self, prompt: str) -> bool:
        return self.has(prompt)

    def _print_memory_usage(self):
        if self.embeddings is None:
            print(f"[HyperCache] Empty cache")
            return
        mem_mb = self.embeddings.element_size() * self.embeddings.nelement() / (1024 ** 2)
        print(f"[HyperCache] Memory: {mem_mb:.2f}MB ({len(self.prompts)} prompts)")

    def save(self, path: str):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        data = {
            'prompts': self.prompts,
            'embeddings': self.embeddings,
            'version': 1,
        }
        torch.save(data, path)
        print(f"[HyperCache] Saved to {path}")

    @classmethod
    def load(cls, path: str, expected_prompts: list = None):
        print(f"[HyperCache] Loading from {path}...")
        data = torch.load(path, map_location='cpu', weights_only=False)

        if expected_prompts is not None:
            if set(data['prompts']) != set(expected_prompts):
                print(f"[HyperCache] Prompt mismatch, cache invalid")
                return None

        instance = object.__new__(cls)
        instance.prompts = data['prompts']
        instance.prompt_to_idx = {p: i for i, p in enumerate(instance.prompts)}
        instance.embeddings = data['embeddings']
        instance.dirty = False
        instance.embed_model_name = "loaded"
        instance._print_memory_usage()
        return instance
