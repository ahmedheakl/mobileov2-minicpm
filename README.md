# mobileov2-minicpm

Minimal inference code for **Mobile-O v2** — a 0.5B text-to-image and image-editing model built
from a frozen MiniCPM-V-4.6 VLM, a small conditioning connector, and a SANA-600M diffusion head
at 512×512.

This repo is inference only. It is the smallest amount of code that reproduces the released
checkpoints exactly; the training, RL and evaluation code is not here.

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
| the trained head (DiT + connector) | [`ahmedheakl/rand-mobile`](https://huggingface.co/ahmedheakl/rand-mobile) | 1.2 GB |
| the frozen VLM encoder | `openbmb/MiniCPM-V-4_6` | 2.5 GB |
| the DiT skeleton + DC-AE decoder | `Efficient-Large-Model/Sana_600M_512px_diffusers` | 2.3 GB |

The checkpoint on the Hub is the **head only** — 548 DiT tensors + 54 connector tensors. The VLM is
frozen during training, so it is not duplicated there; that is why the other two repos are needed.

**`transformers>=5` is required.** MiniCPM-V-4.6 is a native transformers model, and on 4.x the
import fails with `No module named transformers.models.minicpmv4_6`.

## Checkpoints

| `--ckpt` | what it is | GenEval | DPG | ImageReward | GEdit |
|---|---|---|---|---|---|
| `ahmedheakl/rand-mobile` (default) | `soup3-targets`, the best checkpoint | 0.902 | 82.2 | 0.957 | 6.74 |
| `ahmedheakl/rand-mobile-grpo` | the GRPO baseline it was built from | 0.912 | 80.7 | 0.763 | 6.52 |

A local directory works anywhere a repo id does.

## 1024x1024

Pass `--size 1024`. It works for generation and editing, and the weights
(`upsampler_1024/head_ema.safetensors`, 426 MB) come from the same Hugging Face repo on first use.

**The diffusion model is not involved.** The DiT stays at its 16x16 latent and runs exactly the same
steps; 1024px is produced at DECODE time. The frozen DC-AE decoder emits its last hidden features
(128 channels at 512px, the tensor that normally feeds `conv_out`), and a trained 106M head maps
those features straight to 1024px RGB, anchored on a bicubic x2 of the ordinary 512px decode. The
VAE is frozen too -- only the head is learned.

That is deliberate: the DiT is a latency budget, not a resolution choice. Going to a 32x32 latent
would quadruple the transformer cost; this costs almost nothing.

| batch of 8, one RTX PRO 6000 | ms/image |
|---|---|
| 512px | 459 |
| 1024px | 462 |

**+3 ms/image, 0.7%.** Verified equivalent: rendering the same prompt and seed at both sizes and
downscaling the 1024 result back to 512 gives a mean difference of 1.0/255 against the native 512px
output -- the same picture, decoded better.

The head shipped here is `c100`, selected on a reconstruction eval against the alternative approach
(a SwinIR upsampler taking the latent 16x16 -> 32x32 *before* the frozen decode): FID 1.76 vs 2.077
and OCR 47.9 vs 40. The latent-upsampler variant is not included.

Note the VAE must stay bf16 -- fp16 underflows inside the DC-AE decoder and silently yields zeros.
`hires.check_vae` asserts it.

## Defaults worth not "fixing"

Two defaults look low and are not. Both were measured, and raising either makes the pictures worse
while making some benchmark number better:

- **12 solver steps, not 20.** Quality peaks around 8–12. At 20 the model wins nothing and loses on
  human subjects (+0.100 for 12 over 20 on a paired human-quality metric, t=+2.95) for 40% more
  compute.
- **cfg 1.5, not 3.0.** cfg 3.0 scores higher on GenEval, DPG and ImageReward (0.919 / 83.2 / 1.083)
  and is *significantly worse* on human subjects (−0.147, t=−2.68) — oversaturated skin and
  crunchy detail. Pass `--cfg 3.0` when you are chasing prompt-alignment benchmarks, not photos.

Other flags: `--steps`, `--cfg`, `--seed`, `--gpu`, `--ckpt`, `--base`, `--sana`.

## Speed

On one RTX PRO 6000 at the defaults: **363 ms/image** for a batch of 8, ~2.3 s for a single image.
The first call pays ~2 s of CUDA warm-up, so a one-off sample looks far slower per image than the
model really is — pass all your prompts in one command.

Going from 20 steps to 12 only moved this from 410 to 363 ms. The diffusion steps are not the
bottleneck; the per-prompt VLM encode is, and it is deliberately not batched (see below).

## What the code does

```
infer.py              CLI: parse, load, encode, sample, save
mobileov2/modeling.py the model — frozen VLM + connector + DiT + DC-AE, and load_model()
mobileov2/blocks.py   the mcptf connector (verbatim from the research repo)
mobileov2/prompts.py  prompt templates and token-id construction (verbatim)
mobileov2/pipeline.py conditioning and the DPM-Solver++ sampling loop
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

Checked against the full research repo on the same checkpoint, settings and seed:

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
- The editing model rewrites the scene rather than doing a local patch, so fine source detail is
  not preserved the way an inpainting model preserves it.
- The head alone is not a runnable model; the two upstream repos in the table above are required.
