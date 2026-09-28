"""1024px decode: the DiT stays at 16x16, a learned head turns the decode into 1024px RGB.

Mobile-O's DiT is a latency budget, not a resolution choice -- growing it to a 32x32 latent would
multiply the transformer cost by four. So 1024px is produced AFTER the DiT instead: the frozen DC-AE
decoder turns the 16x16x32 latent into its last hidden features (128ch at 512px, the tensor that
normally feeds conv_out), and a trained head maps those features straight to 1024px RGB, anchored on
a bicubic x2 of the ordinary 512px decode. The VAE and the DiT are both untouched.

Measured on the reconstruction eval that selected it: FID 1.76, OCR 47.9, 85 ms of added decode.
(The alternative, a SwinIR latent upsampler taking z16 -> z32 before the frozen decode, measured
worse on both -- FID 2.077, OCR 40 -- and is not shipped here.)

Weights: `upsampler_1024/head_ema.safetensors` in the same Hugging Face repo as the checkpoint,
99M parameters, fetched on first use.

The VAE MUST be bf16. fp16 underflows inside the DC-AE decoder and silently produces zeros -- the
check below is not decoration, it caught a whole run once.
"""
import math
import os

import torch

from .upsampler import build_preset, decode_c100_components

SF = 0.41407


def load_head(repo, device, filename="upsampler_1024/head_ema.safetensors"):
    """-> the frozen 1024px decoder head. `repo` is the HF repo id or a local directory."""
    from safetensors.torch import load_file
    if os.path.isdir(repo):
        path = os.path.join(repo, filename)
        assert os.path.isfile(path), f"{path} not found"
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo, filename)
    head = build_preset("c100").to(device=device, dtype=torch.bfloat16).eval()
    sd = load_file(path)
    head.load_state_dict(sd, strict=True)
    for p in head.parameters():
        p.requires_grad_(False)
    print(f"[hires] 1024px head: {sum(p.numel() for p in head.parameters())/1e6:.1f}M params, "
          f"{len(sd)} tensors loaded strict", flush=True)
    return head


def check_vae(vae):
    assert vae.dtype == torch.bfloat16, f"VAE must be bf16 (fp16 underflows the DC-AE decoder), got {vae.dtype}"
    assert vae.config.latent_channels == 32, vae.config.latent_channels
    assert math.isclose(float(vae.config.scaling_factor), SF, abs_tol=1e-5), vae.config.scaling_factor
    assert getattr(vae.config, "shift_factor", None) is None, "the head was trained without a shift_factor"


@torch.no_grad()
def decode_1024(vae, head, lat, chunk=8):
    """lat: DiT-space latents [B, 32, 16, 16], exactly what the sampler's last step returns
    (BEFORE dividing by the scaling factor). -> float [B, 3, 1024, 1024] in [-1, 1].

    chunk=8 at 1024px costs the same decoder activation memory as the 512px path's chunk of 32."""
    check_vae(vae)
    assert lat.dim() == 4 and tuple(lat.shape[1:]) == (32, 16, 16), tuple(lat.shape)
    outs = []
    for s in range(0, lat.shape[0], chunk):
        z = lat[s:s + chunk]
        x, _guide = decode_c100_components(vae, head, z)
        x = x.float()
        assert tuple(x.shape) == (z.shape[0], 3, 1024, 1024), tuple(x.shape)
        outs.append(x.clamp(-1, 1))
    return torch.cat(outs)
