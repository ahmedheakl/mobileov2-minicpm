"""Mobile-O inference: text-to-image and image editing, one script.

    python infer.py --prompt "a woman holding a ceramic mug in a sunlit kitchen" --out out/gen.png
    python infer.py --image src.jpg --prompt "make it snow" --out out/edit.png
    python infer.py --prompt "a cat" "a dog" --out out/          # --out becomes a directory

The checkpoint is downloaded from the Hugging Face Hub on first use. Defaults are the
measured shipping recipe of the default checkpoint (see README): text-to-image with APG at
cfg 3.0, editing with plain cfg 2.5, 20 DPM-Solver++ steps.
"""
import argparse
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor

from mobileov2 import encode, encode_edit, load_model, make_noise, n_fuse, sample
from mobileov2.hires import load_head
from mobileov2.modeling import SANA_REPO, VLM_REPO
from mobileov2.source_gate import encode_sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", nargs="+", required=True, help="prompt, or edit instruction with --image")
    ap.add_argument("--image", nargs="*", default=[], help="source image(s) -> edit mode; one image is reused for every prompt")
    ap.add_argument("--out", default="out/sample.png", help="a file for one prompt, else a directory")
    ap.add_argument("--ckpt", default="ahmedheakl/rand-dual", help="HF repo id or local dir")
    ap.add_argument("--base", default=VLM_REPO)
    ap.add_argument("--sana", default=SANA_REPO)
    ap.add_argument("--vlm_layers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=None,
                    help="default: 3.0 for text-to-image (with APG), 2.5 for editing")
    ap.add_argument("--guidance", choices=["auto", "apg", "cfg"], default="auto",
                    help="auto = APG for text-to-image, plain CFG for editing")
    ap.add_argument("--size", type=int, default=512, choices=[512, 1024],
                    help="1024 decodes through the trained hi-res head; the diffusion is unchanged")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    edit = bool(args.image)
    if args.cfg is None:
        args.cfg = 2.5 if edit else 3.0
    if args.guidance == "auto":
        args.guidance = "cfg" if edit else "apg"
    srcs = args.image * len(args.prompt) if len(args.image) == 1 else args.image
    assert not edit or len(srcs) == len(args.prompt), \
        f"{len(args.image)} images for {len(args.prompt)} prompts: pass one image, or one per prompt"

    paths = [args.out] if len(args.prompt) == 1 and os.path.splitext(args.out)[1] else \
            [os.path.join(args.out, f"{i}.png") for i in range(len(args.prompt))]
    os.makedirs(os.path.dirname(os.path.abspath(paths[0])), exist_ok=True)

    device = torch.device("cuda", args.gpu) if torch.cuda.is_available() else torch.device("cpu")
    torch.set_grad_enabled(False)
    print(f"[infer] {'edit' if edit else 'gen'} | {len(args.prompt)} prompt(s) | {args.size}px | "
          f"DPM-Solver++ order 2 steps={args.steps} cfg={args.cfg} ({args.guidance}) | {args.ckpt}", flush=True)

    t0 = time.time()
    model = load_model(args.ckpt, device, base=args.base, sana=args.sana, vlm_layers=args.vlm_layers)
    proc = AutoProcessor.from_pretrained(args.base)
    head = load_head(args.ckpt, device) if args.size == 1024 else None
    inner = model.get_model()
    C, S = inner.dit.config.in_channels, inner.dit.config.sample_size
    print(f"[infer] loaded in {time.time() - t0:.1f}s | latent {C}x{S}x{S} | "
          f"{n_fuse(inner.diffusion_connector)} VLM layer(s) fused", flush=True)

    t0 = time.time()
    if edit:
        rows = [{"image": Image.open(p).convert("RGB"), "instruction": q}
                for p, q in zip(srcs, args.prompt)]
        cond = encode_edit(model, proc, rows, device)
        # dual-stream checkpoints also take the source image's latent (None on single-stream ones)
        src = (encode_sources(inner.vae, [r["image"] for r in rows])
               if getattr(inner.dit, "source_gate", None) is not None else None)
    else:
        cond = encode(model, proc.tokenizer, args.prompt, device)
        src = None
    noise = make_noise(len(args.prompt), (C, S), args.seed, device)
    imgs = sample(model, *cond, noise, args.steps, args.cfg, size=args.size, head=head,
                  src=src, guidance=args.guidance)
    dt = time.time() - t0

    assert len(imgs) == len(paths), f"{len(imgs)} images for {len(paths)} outputs"
    for p, im in zip(paths, imgs):
        im.save(p)
        print(f"[infer] {p}", flush=True)
    print(f"[infer] {dt:.2f}s for {len(imgs)} image(s) = {dt / len(imgs) * 1000:.0f} ms/img", flush=True)


if __name__ == "__main__":
    main()
