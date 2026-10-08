# mobileov2-minicpm

Minimal inference code for **Mobile-O v2** — a 0.5B text-to-image and image-editing model built
from a frozen MiniCPM-V-4.6 VLM, a small conditioning connector, and a SANA-600M diffusion head
at 512×512.

The default checkpoint, [`ahmedheakl/rand-dual`](https://huggingface.co/ahmedheakl/rand-dual), is
**dual-stream**: when editing, the DiT also receives the source image's own latent through a learned
gate, so it can copy what the instruction does not touch instead of redrawing it from the VLM's
description (see *Dual-stream editing* below). Text-to-image is unchanged by it.

This repo is inference only. It is the smallest amount of code that reproduces the released
checkpoints (see *Equivalence with the research code*); the training, RL and evaluation code is not
here.

```bash
pip install -r requirements.txt

python infer.py --prompt "a woman holding a ceramic mug in a sunlit kitchen" --out out/gen.png
python infer.py --image photo.jpg --prompt "make it snow" --out out/edit.png
python infer.py --prompt "a cat" "a dog" --out out/          # --out becomes a directory
python infer.py --size 1024 --prompt "..." --out out/big.png   # 1024x1024
```

Nothing needs to be downloaded by hand. On first run it pulls three things from the Hugging Face
Hub and caches them (~6 GB total):

| what | repo | size |
|---|---|---|
| the trained head (DiT + connector + source gate) | [`ahmedheakl/rand-dual`](https://huggingface.co/ahmedheakl/rand-dual) | 1.2 GB |
| the frozen VLM encoder | `openbmb/MiniCPM-V-4_6` | 2.5 GB |
| the DiT skeleton + DC-AE decoder | `Efficient-Large-Model/Sana_600M_512px_diffusers` | 2.3 GB |

The checkpoint on the Hub is the **head only** — 548 DiT tensors + 54 connector tensors, plus the 2
source-gate tensors on a dual-stream checkpoint. The VLM is frozen during training, so it is not
duplicated there; that is why the other two repos are needed.

**`transformers>=5` is required.** MiniCPM-V-4.6 is a native transformers model, and on 4.x the
import fails with `No module named transformers.models.minicpmv4_6`.

## Checkpoints

| `--ckpt` | what it is | GenEval | DPG | FID | ImageReward | ImgEdit | GEdit |
|---|---|---|---|---|---|---|---|
| `ahmedheakl/rand-dual` (default) | dual-stream; average of five RL runs + ReFL; the best checkpoint | 0.897 | 84.1 | 13.2 | 1.083 | 3.28 | 6.80 |
| `ahmedheakl/rand-mobile` | single-stream `soup3-targets`, the previous release | 0.902 | 82.2 | 14.4 | 0.957 | — | 6.74 |
| `ahmedheakl/rand-mobile-grpo` | the single-stream GRPO baseline it was built from | 0.912 | 80.7 | — | 0.763 | — | 6.52 |

`rand-dual` is measured at its defaults below (text-to-image with APG at cfg 3.0, editing at plain
cfg 2.0, 20 steps); the two `rand-mobile` rows at cfg 1.5 and 12 steps. ImgEdit and GEdit use a local
Qwen2.5-VL-72B judge. A local directory works anywhere a repo id does, and the code detects a
dual-stream checkpoint from its weights, so every checkpoint runs with the same command.

## 1024x1024

Pass `--size 1024`. It works for generation and editing, and the weights
(`upsampler_1024/head_ema.safetensors`, 426 MB) come from the same Hugging Face repo on first use.

**The diffusion model is not involved.** The DiT stays at its 16x16 latent and runs exactly the same
steps; 1024px is produced at DECODE time. The frozen DC-AE decoder emits its last hidden features
(128 channels at 512px, the tensor that normally feeds `conv_out`), and a trained 106M head maps
those features straight to 1024px RGB, anchored on a bicubic x2 of the ordinary 512px decode. The
VAE is frozen too -- only the head is learned.

That is deliberate: the DiT is a latency budget, not a resolution choice. Going to a 32x32 latent
would quadruple the transformer cost; this adds one decoder-side head.

| one image, batch 1, 20 steps, one RTX PRO 6000 | ms |
|---|---|
| VLM + connector encode | 193 |
| diffusion, 20 steps | 595 |
| decode to 512px | 29 |
| **end-to-end, 512px** | **817** |
| **end-to-end, 1024px** (head replaces the decode, 86 ms) | **874** |

**+58 ms, +7.1% end-to-end, for four times the pixels** (median of 30 runs after 5 warm-up). Verified
equivalent: rendering the same prompt and seed at both sizes and downscaling the 1024 result back to
512 gives a mean difference of 1.0/255 against the native 512px output -- the same picture, decoded
better. On nine benchmarks the head is content-neutral (GenEval, DPG and ImgEdit unchanged) and
improves FID by 0.48.

The head shipped here is `c100`, selected on a reconstruction eval against the alternative approach
(a SwinIR upsampler taking the latent 16x16 -> 32x32 *before* the frozen decode): FID 1.76 vs 2.077
and OCR 47.9 vs 40. The latent-upsampler variant is not included.

Note the VAE must stay bf16 -- fp16 underflows inside the DC-AE decoder and silently yields zeros.
`hires.check_vae` asserts it.

## Defaults

The defaults are the measured recipe for `rand-dual`:

- **Text-to-image: APG at cfg 3.0.** APG (adaptive projected guidance, eta 0) removes the part of the
  guidance update that only pushes saturation and contrast and keeps the rest. With it, cfg 3.0 beats
  plain cfg 2.0 on DPG (+1.1), FID and ImageReward, and on a paired human-image benchmark (+0.150,
  t=3.46: realism, hands and group shots all improve; measured on this checkpoint's immediate
  predecessor). Without APG, raising cfg to 3.0 makes people look
  worse, which is why the older checkpoints shipped at cfg 1.5.
- **Editing: plain cfg 2.0.** APG at cfg 3.0 raised GEdit by 0.15 but lowered ImgEdit by 0.06; at
  cfg 2.0 it was neutral.
- **20 DPM-Solver++ steps.**

Change them with `--cfg`, `--guidance {apg,cfg}` and `--steps`. For the single-stream `rand-mobile`
checkpoints, their own measured recipe is `--cfg 1.5 --steps 12 --guidance cfg`.

## Dual-stream editing

A dual-stream checkpoint carries a small extra module, the **source gate**: a bias-free patch
projection from the 32-channel DC-AE latent to the DiT width, and a scalar gate. For an edit, the
source image is resized and centre-cropped to 512x512 exactly as in training, encoded by the frozen
DC-AE, projected, multiplied by tanh(gate) and **added** to the DiT's patch embedding of the noisy
latent at every step. Both guidance branches get the same source. Token count and DiT cost are
unchanged; the only extra work is one VAE encode of the source (52 ms). For text-to-image nothing is
added. `mobileov2/source_gate.py` is the whole implementation.

## Speed

One image, batch 1, 20 steps, `rand-dual`, one RTX PRO 6000 Blackwell, median of 30 runs after 5
warm-up:

| | ms |
|---|---|
| text-to-image: VLM + connector encode | 207 |
| text-to-image: 20 steps + decode, APG cfg 3.0 | 693 (plain cfg: 690) |
| **text-to-image, end-to-end** | **895** |
| editing: VLM + connector encode (instruction and null branch) | 692 |
| editing: source latent encode (dual-stream only) | 52 |
| editing: 20 steps + decode, cfg 2.0 | 689 |
| **editing, end-to-end** | **1584** |

APG costs 2 ms over plain guidance. An edit runs the VLM over the source image twice (with the
instruction and with an empty one), which is most of its extra time. The first call in a process
pays ~2 s of CUDA warm-up, so pass all your prompts in one command.

## What the code does

```
infer.py              CLI: parse, load, encode, sample, save
mobileov2/modeling.py the model — frozen VLM + connector + DiT + DC-AE, and load_model()
mobileov2/blocks.py   the mcptf connector (verbatim from the research repo)
mobileov2/prompts.py  prompt templates and token-id construction (verbatim)
mobileov2/pipeline.py conditioning, the DPM-Solver++ sampling loop, and APG
mobileov2/source_gate.py the dual-stream source gate and the source-image encode
mobileov2/hires.py    1024px decode: load the head, decode_1024()
mobileov2/upsampler.py the c100 decoder head itself (verbatim from the research repo)
```

Three details in `pipeline.py` are deliberate. None of them will raise an error if changed — the
output just gets worse:

1. **The VLM forward is per prompt, never batched.** Batching moves MiniCPM's hidden states by up
   to 0.5σ in bf16. That is reduction-order noise rather than padding (a batch of *identical*
   prompts moves just as much), and the solver amplifies it into visible pixel drift. Only the
   connector *output* is padded; the DiT gets the true attention mask and is what gets batched.
2. **The unconditional branch is a real forward**, of the empty prompt when generating, or of the
   same source image with an empty instruction when editing. It is not a zero vector: zeroing the
   connector output feeds literal zeros into cross-attention, far out of distribution, and inflates
   the latents until the decoder saturates to solid black.
3. **The noise is one seed-`--seed` draw shared across the batch**, so a prompt and seed give the
   same picture no matter what it is batched with.

`prompts.py` is copied verbatim from the training code rather than rewritten, because a drifted
template still produces an image — just a worse one, with no error to tell you.

## Equivalence with the research code

`rand-dual`, checked against the research repo's evaluation path on GPU (3 ImgEdit edits at cfg 2.0
and 3 prompts at APG cfg 3.0, 20 steps, same noise):

- all 1759 loaded tensors identical, including the two source-gate tensors;
- generation and edit conditioning (`ehs`, `nehs`, `mask`, `nmask`) bit-identical;
- source latent within 2 bf16 rounding steps (0.031 max). The first difference is one linear layer
  inside the frozen DC-AE encoder, with identical inputs and weights: a GPU kernel rounding
  differently, not a code difference;
- images: mean difference 0.16–0.20/255 for text-to-image and 0.33–1.6/255 for edits;
- with the gate zeroed, the same edits move by ~55/255, so the gate is doing real work.

Earlier, for `rand-mobile`, on the same checkpoint, settings and seed:

- all 1757 loaded tensors identical;
- edit conditioning (`ehs`, `nehs`, `mask`, `nmask`) bit-identical;
- sampler output bit-identical on the same conditioning;
- generated PNGs byte-identical across processes.

Editing run end-to-end in *separate* processes lands within bf16 kernel nondeterminism — median
difference 0, mean 0.33/255, max 18 on 0.1% of channels — i.e. the same image, not the same bytes.

## Limitations

- 512×512, or 1024×1024 via `--size 1024` (a better decode of the same 512px-latent image, not
  more diffusion detail).
- English prompts.
- The editing model rewrites the scene rather than doing a local patch. The dual-stream gate keeps
  unedited regions much closer to the source than the single-stream checkpoints do, but it is not a
  pixel-exact inpainting model.
- The head alone is not a runnable model; the two upstream repos in the table above are required.
