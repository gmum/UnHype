import torch


class CombinedCFGModel:
    """Wrapper that uses a single model with LoRA toggled for CFG guidance.

    When DDIMSampler applies classifier-free guidance, it concatenates
    [unconditional, conditional] inputs. This wrapper splits them and routes:
    - Unconditional pass: LoRA OFF (via model.hyper.no_lora())
    - Conditional pass: LoRA ON
    """

    def __init__(self, model):
        self.model = model
        self.device = model.device

    def apply_model(self, x, t, c):
        b2 = x.shape[0]
        assert b2 % 2 == 0
        b = b2 // 2

        x_uncond, x_cond = x[:b], x[b:]
        t_uncond, t_cond = t[:b], t[b:]

        if isinstance(c, dict):
            c_uncond = {k: [v[:b] for v in c[k]] if isinstance(c[k], list) else c[k][:b] for k in c}
            c_cond = {k: [v[b:] for v in c[k]] if isinstance(c[k], list) else c[k][b:] for k in c}
        else:
            c_uncond, c_cond = c[:b], c[b:]

        with self.model.hyper.no_lora():
            out_uncond = self.model.apply_model(x_uncond, t_uncond, c_uncond)

        out_cond = self.model.apply_model(x_cond, t_cond, c_cond)

        return torch.cat([out_uncond, out_cond], dim=0)

    def get_learned_conditioning(self, prompts):
        return self.model.get_learned_conditioning(prompts)

    def decode_first_stage(self, z):
        return self.model.decode_first_stage(z)

    def eval(self):
        self.model.eval()
        return self

    def __getattr__(self, name):
        return getattr(self.model, name)
