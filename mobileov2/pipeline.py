"""Conditioning and sampling. This is the file that decides what the pictures look like.

Three things here are deliberate and were each measured. Changing them will not raise an
error, it will just make the output worse:

1. The VLM forward is PER PROMPT, never batched. Batching moves MiniCPM's hidden states by
   up to 0.5 sigma in bf16 -- reduction-order noise, not padding, since a batch of
   IDENTICAL prompts moves just as much -- and the solver amplifies that to a visible pixel
   drift. Only the connector OUTPUT is padded; the DiT gets the true attention mask and is
   what gets batched.
2. The unconditional branch is a real forward of the empty prompt (generation) or of the
   same source image with an empty instruction (editing). It is NOT a zero vector: zeroing
   the connector output feeds literal zeros into cross-attention, which is far out of
   distribution and inflates the latents until the VAE saturates to solid black.
3. The solver is DPM-Solver++ order 2 with flow sigmas, taken from SANA's own scheduler
   config, and the checkpoints were tuned against it.
"""
import copy

import torch
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from diffusers.utils.torch_utils import randn_tensor

from .modeling import n_fuse
from .source_gate import set_source
from .prompts import (IMAGE_TOKEN_INDEX, UND_MAX_SLICES, build_edit_ids, build_ids,
                      embed_edit_row, process_und_image)


def _pad(outs, nulls, device):
    """Pad the per-prompt connector outputs into one batch. -> ehs, nehs, mask, nmask."""
    nmax = max(max(e.shape[1] for e in outs), max(u.shape[1] for u in nulls))
    ehs = torch.zeros(len(outs), nmax, outs[0].shape[-1], device=device, dtype=outs[0].dtype)
    nehs = torch.zeros_like(ehs)
    mask = torch.zeros(len(outs), nmax, dtype=torch.bool, device=device)
    nmask = torch.zeros_like(mask)
    for i, (e, u) in enumerate(zip(outs, nulls)):
        ehs[i, :e.shape[1]] = e[0]; mask[i, :e.shape[1]] = True
        nehs[i, :u.shape[1]] = u[0]; nmask[i, :u.shape[1]] = True
    return ehs, nehs, mask, nmask


@torch.no_grad()
def encode(model, tokenizer, prompts, device):
    """Generation conditioning. The null branch is the EMPTY prompt, shared by the batch."""
    inner = model.get_model()
    conn = inner.diffusion_connector
    nf = n_fuse(conn)

    def run(ids):
        out = inner.language_model(inputs_embeds=inner.get_input_embeddings()(ids),
                                   attention_mask=torch.ones_like(ids, dtype=torch.bool),
                                   output_hidden_states=True)
        return conn(out.hidden_states[-nf:]).float()

    outs = [run(torch.tensor(build_ids(tokenizer, p)).unsqueeze(0).to(device)) for p in prompts]
    empty = run(torch.tensor(build_ids(tokenizer, "")).unsqueeze(0).to(device))
    return _pad(outs, [empty] * len(prompts), device)


@torch.no_grad()
def encode_edit(model, processor, rows, device):
    """Editing conditioning. rows: [{"image": PIL, "instruction": str}].

    The null branch is the SAME source image with an EMPTY instruction, so guidance pushes
    toward following the instruction rather than toward generating from nothing.
    """
    inner = model.get_model()
    conn = inner.diffusion_connector
    nf = n_fuse(conn)

    def run(embeds, attn):
        out = inner.language_model(inputs_embeds=embeds, attention_mask=attn,
                                   output_hidden_states=True)
        return conn(out.hidden_states[-nf:]).float()

    outs, nulls = [], []
    for r in rows:
        und = process_und_image(r["image"].convert("RGB"), processor.image_processor,
                                max_slice_nums=UND_MAX_SLICES)
        tok = processor.tokenizer
        for text, sink in ((r["instruction"], outs), ("", nulls)):
            ids = build_edit_ids(tok, text).unsqueeze(0).to(device)
            sink.append(run(*embed_edit_row(model, ids, und, device)))
    return _pad(outs, nulls, device)


def make_noise(n, shape, seed, device):
    """One seed-`seed` draw shared by every prompt in the batch -- the eval convention, so
    the same prompt and seed give the same picture regardless of what it is batched with."""
    C, S = shape
    z = randn_tensor((1, C, S, S), generator=torch.Generator("cpu").manual_seed(seed),
                     device=torch.device("cpu"), dtype=torch.float32)
    return z.repeat(n, 1, 1, 1).to(device)


def _apg(uncond, text, w, x_t, sigma):
    """Adaptive Projected Guidance (Sadat et al. 2024, eta=0), in x0 space: drop the part of the guidance
    update that is parallel to the conditional prediction (it mostly pushes saturation and contrast) and keep
    the orthogonal part. With it, cfg 3.0 beats plain cfg 2.0 on DPG, FID, ImageReward and human quality."""
    sg = float(sigma)
    if sg <= 0 or w == 1.0:
        return text
    d_c = x_t.float() - sg * text.float()
    d_u = x_t.float() - sg * uncond.float()
    diff = d_c - d_u
    v1 = torch.nn.functional.normalize(d_c.double().flatten(1), dim=1).view_as(d_c)
    par = ((diff.double() * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1).float()
    d_g = d_c + (w - 1.0) * (diff - par)
    return ((x_t.float() - d_g) / sg).to(text.dtype)


@torch.no_grad()
def sample(model, ehs, nehs, mask, nmask, noise, steps, cfg, size=512, head=None, src=None, guidance="cfg"):
    """-> list of PIL images, one per row of `ehs`.

    size=1024 routes the final latent through the trained decoder head (see hires.py) instead of
    the ordinary DC-AE decode. The DIFFUSION is identical either way -- same latent, same steps --
    so 1024px is a decode-time choice and costs no extra DiT compute.

    src: [B, 32, 16, 16] source latents (source_gate.encode_sources) for EDITS on a dual-stream checkpoint;
    None for generation. guidance: "cfg" (plain classifier-free guidance) or "apg"."""
    assert guidance in ("cfg", "apg"), guidance
    assert size in (512, 1024), size
    assert size == 512 or head is not None, "size=1024 needs the 1024px head (see hires.load_head)"
    inner = model.get_model()
    dit, vae = inner.dit, inner.vae
    bsz = ehs.shape[0]
    assert noise.shape[0] == bsz, f"{noise.shape[0]} noise rows for {bsz} images"
    assert cfg >= 1.0, cfg

    use_cfg = cfg != 1.0
    ehs_in = torch.cat([nehs, ehs]) if use_cfg else ehs
    mask_in = torch.cat([nmask, mask]) if use_cfg else mask

    sched = copy.deepcopy(inner.noise_scheduler)
    sched.set_timesteps(steps)
    lat = noise
    if getattr(dit, "source_gate", None) is not None:
        assert src is None or src.shape[0] == bsz, f"{src.shape[0]} source latents for {bsz} images"
        set_source(dit, src)       # None for generation: the dual-stream DiT then runs single-stream
    else:
        assert src is None, "source latents given, but this checkpoint is single-stream"
    try:
        for i, t in enumerate(sched.timesteps):
            inp = torch.cat([lat, lat]) if use_cfg else lat
            inp = sched.scale_model_input(inp, t)
            pred = dit(hidden_states=inp.to(dit.dtype), encoder_hidden_states=ehs_in.to(dit.dtype),
                       timestep=t.unsqueeze(0).expand(inp.shape[0]).to(lat.device),
                       encoder_attention_mask=mask_in).sample.float()
            if use_cfg:
                uncond, text = pred.chunk(2)
                if guidance == "apg":
                    pred = _apg(uncond, text, cfg, lat, sched.sigmas[i])
                else:
                    pred = uncond + cfg * (text - uncond)
            lat = sched.step(pred, t, lat).prev_sample
    finally:
        if getattr(dit, "source_gate", None) is not None:
            set_source(dit, None)

    if size == 1024:
        from .hires import decode_1024
        x = decode_1024(vae, head, lat)            # takes the DiT-space latent, scales internally
        imgs = numpy_to_pil((x / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy())
        assert len(imgs) == bsz
        return imgs

    lat = lat / vae.config.scaling_factor
    assert getattr(vae.config, "shift_factor", None) is None
    imgs = []
    for s in range(0, bsz, 32):  # 32 measured optimal; bigger chunks are slower AND use more VRAM
        x = (vae.decode(lat[s:s + 32].to(vae.dtype)).sample / 2 + 0.5).clamp(0, 1)
        imgs += numpy_to_pil(x.cpu().permute(0, 2, 3, 1).float().numpy())
    assert len(imgs) == bsz
    return imgs
