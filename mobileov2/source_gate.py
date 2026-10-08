"""Dual-stream editing: the source image's latent, added to the DiT's patch embedding behind a learned gate.

A dual-stream checkpoint carries two extra tensors, `model.dit.source_gate.proj.weight` and
`model.dit.source_gate.gate`. For an EDIT, the source image is encoded by the frozen DC-AE into the same
32x16x16 latent space the DiT denoises, patch-embedded by `proj`, scaled by tanh(gate) and ADDED to the DiT's
own patch embedding of the noisy latent. Token count and cost are unchanged. Generation passes no source,
which leaves the DiT exactly as a single-stream model.

Why it exists: through the VLM alone the DiT only gets a semantic description of the source, so it redraws
the parts the instruction does not touch. The gate hands it the source pixels to copy.

Verbatim (minus training-only helpers) from the research repo's `mobileo/model/source_gate.py`.
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

KEY = ".dit.source_gate."


class SourceLatentGate(nn.Module):
    """[B, C, h, w] frozen-VAE source latent -> [B, h*w/p^2, inner_dim] tokens."""

    def __init__(self, in_channels, patch_size, inner_dim):
        super().__init__()
        # bias-free: a zero latent contributes exactly zero
        self.proj = nn.Conv2d(in_channels, inner_dim, patch_size, patch_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, z_src):
        x = self.proj(z_src.to(self.proj.weight.dtype))
        return torch.tanh(self.gate) * x.flatten(2).transpose(1, 2)


def install(dit):
    """Attach the gate to `dit` (as a submodule, so its weights load with the DiT's) and hook it onto the
    patch embedding. Nothing changes until `set_source` arms it."""
    if getattr(dit, "source_gate", None) is not None:
        return dit.source_gate
    cfg = dit.config
    dit.source_gate = SourceLatentGate(cfg.in_channels, cfg.patch_size,
                                       cfg.num_attention_heads * cfg.attention_head_dim)
    dit.patch_embed._src_tokens = None

    def _add(mod, _inp, out):
        tok = getattr(mod, "_src_tokens", None)
        if tok is None:
            return out
        if out.shape[0] == 2 * tok.shape[0]:      # classifier-free guidance doubles the batch;
            tok = torch.cat([tok, tok], dim=0)    # both branches see the same source
        assert out.shape == tok.shape, f"source tokens {tuple(tok.shape)} != patch embed {tuple(out.shape)}"
        return out + tok.to(out.dtype)

    dit.patch_embed.register_forward_hook(_add)
    return dit.source_gate


def has_gate(state_dict):
    return any(KEY in k for k in state_dict)


def set_source(dit, z_src):
    """Arm the gate with source latents [B, C, h, w] for the following DiT calls; None disarms it."""
    if getattr(dit, "source_gate", None) is None:
        assert z_src is None, "a source latent was given but this checkpoint has no source gate"
        return
    dit.patch_embed._src_tokens = None if z_src is None else dit.source_gate(z_src)


@torch.no_grad()
def encode_sources(vae, pil_images, size=512):
    """Source images -> DC-AE latents in DiT space, with the TRAINING geometry: resize the short side to
    `size` (bicubic), centre-crop to size x size, normalise to [-1, 1]."""
    def geometry(img):
        oh, ow = img.height, img.width
        rs = (size, int(ow * size / oh)) if size / oh > size / ow else (int(oh * size / ow), size)
        return T.Compose([T.Resize(rs, interpolation=InterpolationMode.BICUBIC),
                          T.CenterCrop((size, size))])(img.convert("RGB"))

    to_t = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    px = torch.stack([to_t(geometry(im)) for im in pil_images])
    z = vae.encode(px.to(vae.device, vae.dtype)).latent * vae.config.scaling_factor
    return z.float()
