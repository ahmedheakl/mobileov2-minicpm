"""
Decoder-head latent upsamplers for Mobile-O.

Instead of upsampling the DC-AE latent (16x16 -> 32x32) BEFORE the frozen decode
(the v1-v6 `LuaUpsampler`), these modules are the **last layer of the VAE decode**:
the frozen DC-AE decoder turns z_low (16x16x32, a 512px image's latent) into its
last hidden features (128ch @ 512x512, the tensor that feeds the frozen conv_out),
and a NEW trainable head maps those features -> 1024px RGB directly. The VAE is
NEVER trained. Two variants (user request; pixel-space heads explicitly excluded):

  A. FeatHeadUNet   -- generic NAFNet-style U-shape on the 128ch@512 features.
     Params sit at coarse scales (128^2/256^2) so a ~100M model stays cheap in
     FLOPs; skip connections + a fixed bicubic RGB anchor preserve the text detail
     that already lives in the 512px decode (recon reads OCR ~67).

  C. DecUpBlockHead -- extends the decoder with its OWN block vocabulary: DC-AE
     ResBlock / EfficientViTBlock at 512, a native DCUpBlock2d pixel-shuffle x2 to
     1024, then a thin RMSNorm+SiLU+conv_out. The most "native" continuation of the
     frozen decoder.

Both output a residual on top of `base` (bicubic x2 of the frozen 512px decode),
initialised so out ~= base at step 0 (stable identity-ish start). `base` is a fixed,
non-learned anchor tensor, NOT a learned pixel-space module -- the trainable path
operates purely on decoder features.
"""
import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

# DC-AE native blocks (variant C) -- diffusers 0.35.2
from diffusers.models.autoencoders.autoencoder_dc import ResBlock, EfficientViTBlock, DCUpBlock2d
from diffusers.models.normalization import RMSNorm


# Runtime contract for the trained c100 checkpoint.  Keeping these invariants
# beside the architecture gives packaging, evaluation, and inference one source
# of truth instead of letting an incompatible VAE fail deep inside a forward.
C100_LATENT_CHANNELS = 32
C100_LATENT_SIZE = 16
C100_SCALING_FACTOR = 0.41407
C100_VAE_COMPRESSION = 32
C100_DECODER_FEATURE_CHANNELS = 128

DEFAULT_DECODER_UPSAMPLER_POSTPROCESS = {
    "mode": "decoded_textblend",
    "alpha": 1.0,
    "sigma": 5.0,
    "mask_gain": 6.0,
    "unsharp": 5.0,
}


def normalize_decoder_upsampler_postprocess(recipe):
    """Validate and canonicalize the optional deployable postprocess recipe."""
    if recipe is None:
        return None
    if not isinstance(recipe, Mapping):
        raise ValueError(
            "decoder_upsampler_postprocess must be null or a mapping with "
            "mode='decoded_textblend'"
        )

    allowed = set(DEFAULT_DECODER_UPSAMPLER_POSTPROCESS)
    unknown = sorted(set(recipe) - allowed)
    if unknown:
        raise ValueError(
            "decoder_upsampler_postprocess contains unsupported keys: "
            + ", ".join(unknown)
        )
    if recipe.get("mode") != "decoded_textblend":
        raise ValueError(
            "decoder_upsampler_postprocess.mode must be 'decoded_textblend'"
        )

    normalized = dict(DEFAULT_DECODER_UPSAMPLER_POSTPROCESS)
    normalized.update(recipe)
    normalized["mode"] = "decoded_textblend"
    for key in ("alpha", "sigma", "mask_gain", "unsharp"):
        value = normalized[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"decoder_upsampler_postprocess.{key} must be a finite number"
            )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(
                f"decoder_upsampler_postprocess.{key} must be a finite number"
            )
        normalized[key] = value

    if not 0.0 <= normalized["alpha"] <= 1.0:
        raise ValueError("decoder_upsampler_postprocess.alpha must be in [0, 1]")
    if normalized["sigma"] <= 0.0:
        raise ValueError("decoder_upsampler_postprocess.sigma must be > 0")
    if normalized["mask_gain"] < 0.0:
        raise ValueError("decoder_upsampler_postprocess.mask_gain must be >= 0")
    if normalized["unsharp"] < 0.0:
        raise ValueError("decoder_upsampler_postprocess.unsharp must be >= 0")
    return normalized


def _edge_mask(img_m11, sigma):
    """Return a soft text/edge-density mask for an image in [-1, 1]."""
    gray = (
        0.299 * img_m11[:, 0:1]
        + 0.587 * img_m11[:, 1:2]
        + 0.114 * img_m11[:, 2:3]
    )
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_x.transpose(2, 3), padding=1)
    kernel_size = max(3, int(6 * sigma) | 1)
    mask = gaussian_blur(
        grad_x.abs() + grad_y.abs(), kernel_size=kernel_size, sigma=sigma
    )
    return (
        mask / (mask.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    ).clamp(0, 1)


# Public compatibility name used by the standalone textblend utility.
edge_mask = _edge_mask


@torch.no_grad()
def decoded_textblend_refine(
    learned,
    decoded_guide,
    alpha=1.0,
    sigma=5.0,
    mask_gain=6.0,
    unsharp=5.0,
):
    """Blend detail from ``decode(z_low)`` into a learned 1024px result.

    ``decoded_guide`` may be the native 512px VAE decode or its already
    bicubic-upsampled 1024px anchor.  It must never be the unavailable source
    dataset image.
    """
    if learned.ndim != 4 or learned.shape[1] != 3:
        raise ValueError(
            "learned decoder-upsampled image must have shape [B, 3, H, W]"
        )
    if decoded_guide.ndim != 4 or decoded_guide.shape[1] != 3:
        raise ValueError("decoded guide must have shape [B, 3, H, W]")
    if decoded_guide.shape[0] != learned.shape[0]:
        raise ValueError("learned image and decoded guide batch sizes must match")

    # Postprocess semantics are defined on image-domain tensors.  Clamp here,
    # rather than relying on individual callers, so packaged inference and all
    # evaluators remain numerically identical even when bicubic interpolation
    # or the learned residual overshoots [-1, 1].
    learned_f = learned.float().clamp(-1, 1)
    guide = F.interpolate(
        decoded_guide.float(),
        size=learned.shape[-2:],
        mode="bicubic",
        align_corners=False,
    ).clamp(-1, 1)
    source = guide
    if unsharp > 0:
        source = (
            guide
            + unsharp
            * (guide - gaussian_blur(guide, kernel_size=5, sigma=1.0))
        ).clamp(-1, 1)
    mask = (mask_gain * _edge_mask(guide, sigma=sigma)).clamp(0, 1)
    return (learned_f + alpha * mask * (source - learned_f)).clamp(-1, 1)


def apply_decoder_upsampler_postprocess(learned, decoded_guide, recipe):
    """Return the canonical fp32 image-domain output for runtime and evaluation."""
    normalized = normalize_decoder_upsampler_postprocess(recipe)
    if normalized is None:
        return learned.float().clamp(-1, 1)
    return decoded_textblend_refine(
        learned,
        decoded_guide,
        alpha=normalized["alpha"],
        sigma=normalized["sigma"],
        mask_gain=normalized["mask_gain"],
        unsharp=normalized["unsharp"],
    )


def validate_c100_compatibility(vae, dit=None, latents=None):
    """Fail early when c100 is paired with anything except its trained contract."""
    issues = []
    vae_config = getattr(vae, "config", None)
    if vae_config is None:
        issues.append("VAE has no config")
    else:
        latent_channels = getattr(vae_config, "latent_channels", None)
        if latent_channels != C100_LATENT_CHANNELS:
            issues.append(
                f"VAE latent_channels must be {C100_LATENT_CHANNELS}, "
                f"got {latent_channels!r}"
            )
        scaling_factor = getattr(vae_config, "scaling_factor", None)
        if (
            not isinstance(scaling_factor, (int, float))
            or not math.isclose(
                float(scaling_factor),
                C100_SCALING_FACTOR,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
        ):
            issues.append(
                f"VAE scaling_factor must be {C100_SCALING_FACTOR}, "
                f"got {scaling_factor!r}"
            )
        shift_factor = getattr(vae_config, "shift_factor", None)
        if shift_factor is not None:
            issues.append(
                "VAE shift_factor is unsupported; c100 was trained without one"
            )

    compression = getattr(
        vae,
        "spatial_compression_ratio",
        getattr(vae_config, "spatial_compression_ratio", None),
    )
    if compression != C100_VAE_COMPRESSION:
        issues.append(
            f"VAE spatial compression must be {C100_VAE_COMPRESSION}x, "
            f"got {compression!r}"
        )

    decoder = getattr(vae, "decoder", None)
    conv_out = getattr(decoder, "conv_out", None)
    feature_channels = getattr(conv_out, "in_channels", None)
    if feature_channels != C100_DECODER_FEATURE_CHANNELS:
        issues.append(
            "VAE decoder.conv_out.in_channels must be "
            f"{C100_DECODER_FEATURE_CHANNELS}, got {feature_channels!r}"
        )
    if getattr(vae, "use_tiling", False):
        issues.append("VAE tiled decoding is unsupported by the decoder feature tap")
    if getattr(vae, "use_slicing", False):
        issues.append("VAE sliced decoding is unsupported by the decoder feature tap")

    if dit is not None:
        dit_config = getattr(dit, "config", None)
        sample_size = getattr(dit_config, "sample_size", None)
        valid_sample_size = sample_size == C100_LATENT_SIZE or (
            isinstance(sample_size, (tuple, list))
            and tuple(sample_size) == (C100_LATENT_SIZE,) * 2
        )
        if not valid_sample_size:
            issues.append(
                f"DiT sample_size must be {C100_LATENT_SIZE}, "
                f"got {sample_size!r}; 1024-native DiTs are incompatible"
            )
        in_channels = getattr(dit_config, "in_channels", None)
        if in_channels != C100_LATENT_CHANNELS:
            issues.append(
                f"DiT in_channels must be {C100_LATENT_CHANNELS}, "
                f"got {in_channels!r}"
            )

    if latents is not None:
        expected = (
            C100_LATENT_CHANNELS,
            C100_LATENT_SIZE,
            C100_LATENT_SIZE,
        )
        if latents.ndim != 4 or tuple(latents.shape[1:]) != expected:
            issues.append(
                "scaled latents must have shape "
                f"[B, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(latents.shape)}"
            )

    if issues:
        raise ValueError(
            "decoder_upsampler_preset='c100' is incompatible: "
            + "; ".join(issues)
        )
    return True


# ----------------------------------------------------------------------------
# Frozen decoder feature tap
# ----------------------------------------------------------------------------
class DecoderFeatureTap:
    """Runs the frozen DC-AE decode of z_low and returns (feats, base):
      feats : 128ch @ 512x512, input to frozen conv_out (detached)
      base  : bicubic x2 of the frozen 512px RGB decode -> 1024px anchor (detached)
    Nothing here is trainable; grad never flows into the VAE."""
    def __init__(self, vae, sf, dtype=None, output_dtype=torch.float32):
        self.vae, self.sf = vae, sf
        # use the VAE's own param dtype by default (bf16 in training, fp16 in the
        # stock evaluators) -- forward activations are O(1), so fp16 vs bf16 features
        # are numerically equivalent here (the fp16 issue was gradient underflow only).
        self.dtype = dtype or next(vae.parameters()).dtype
        self.output_dtype = output_dtype

    @torch.no_grad()
    def __call__(self, z_low):
        cap = {}
        h = self.vae.decoder.conv_out.register_forward_pre_hook(
            lambda m, inp: cap.__setitem__("f", inp[0]))
        try:
            rgb512 = self.vae.decode((z_low / self.sf).to(self.dtype)).sample
        finally:
            h.remove()
        if "f" not in cap:
            raise RuntimeError(
                "VAE decoder feature tap did not observe decoder.conv_out"
            )
        feats = cap["f"].detach().to(self.output_dtype)
        base = F.interpolate(rgb512.float(), scale_factor=2, mode="bicubic",
                             align_corners=False).to(self.output_dtype)
        return feats, base


@torch.no_grad()
def decode_c100_components(
    vae,
    head,
    scaled_latents,
    dit=None,
):
    """Return the raw c100 output and decoded guide using runtime-exact dtypes."""
    validate_c100_compatibility(vae, dit=dit, latents=scaled_latents)

    try:
        vae_param = next(vae.parameters())
        head_param = next(head.parameters())
    except StopIteration as exc:
        raise ValueError("VAE and decoder upsampler must have parameters") from exc

    if scaled_latents.device != vae_param.device:
        raise ValueError(
            "scaled latents and VAE must be on the same device, got "
            f"{scaled_latents.device} and {vae_param.device}"
        )
    if head_param.device != vae_param.device:
        raise ValueError(
            "decoder upsampler and VAE must be on the same device, got "
            f"{head_param.device} and {vae_param.device}"
        )

    tap = DecoderFeatureTap(
        vae,
        vae.config.scaling_factor,
        dtype=vae_param.dtype,
        output_dtype=head_param.dtype,
    )
    features, decoded_guide = tap(scaled_latents)
    expected_features = (
        scaled_latents.shape[0],
        C100_DECODER_FEATURE_CHANNELS,
        512,
        512,
    )
    if tuple(features.shape) != expected_features:
        raise ValueError(
            "c100 expected VAE final decoder features shaped "
            f"{expected_features}, got {tuple(features.shape)}"
        )
    expected_guide = (scaled_latents.shape[0], 3, 1024, 1024)
    if tuple(decoded_guide.shape) != expected_guide:
        raise ValueError(
            "c100 expected a bicubic decoded guide shaped "
            f"{expected_guide}, got {tuple(decoded_guide.shape)}"
        )

    learned = head(features, decoded_guide)
    if tuple(learned.shape) != expected_guide:
        raise ValueError(
            f"c100 must output shape {expected_guide}, got {tuple(learned.shape)}"
        )
    return learned, decoded_guide


@torch.no_grad()
def run_c100_decoder_upsampler(
    vae,
    head,
    scaled_latents,
    postprocess=None,
    dit=None,
):
    """Decode sf-scaled 512px latents through the frozen VAE and c100 head."""
    learned, decoded_guide = decode_c100_components(
        vae,
        head,
        scaled_latents,
        dit=dit,
    )
    return apply_decoder_upsampler_postprocess(
        learned, decoded_guide, postprocess
    )


# ----------------------------------------------------------------------------
# Variant A: NAFNet-style U-shape head
# ----------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.g = nn.Parameter(torch.ones(c)); self.b = nn.Parameter(torch.zeros(c))
    def forward(self, x):
        u = x.mean(1, keepdim=True); s = (x - u).var(1, keepdim=True, unbiased=False)
        return (x - u) / torch.sqrt(s + 1e-6) * self.g[None, :, None, None] + self.b[None, :, None, None]


class NAFBlock(nn.Module):
    """Simplified NAFNet block (Chen et al. ECCV'22): depthwise conv + SimpleGate +
    simplified channel attention, then a gated FFN. No expensive softmax attention,
    so it is cheap per-param -> good for a high-res SR head."""
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, 1, 1, groups=dw)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.norm2 = LayerNorm2d(c)
        ff = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ff, 1)
        self.conv5 = nn.Conv2d(ff // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    @staticmethod
    def _gate(x):
        a, b = x.chunk(2, dim=1); return a * b

    def forward(self, x):
        y = self.conv2(self.conv1(self.norm1(x)))
        y = self._gate(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta
        y = self.conv5(self._gate(self.conv4(self.norm2(x))))
        return x + y * self.gamma


class FeatHeadUNet(nn.Module):
    """Variant A. NAFNet U-shape over the frozen decoder features, run at LOW
    internal resolution for efficiency.

      feats 128ch@512  --pixel_unshuffle(2)-->  512ch@256   (lossless downsample)
        intro -> w0@256 -> [enc @256, down @128, down @64] -> deep mid @64
        -> [up @128 +skip, up @256 +skip] -> pixel-shuffle 256->512->1024 (thin)
        -> conv 3ch residual on `base`.

    Heavy blocks sit at 64^2/128^2 (FLOP- and memory-cheap), so a ~100M model
    trains at a real batch size; the 512/1024 tail is thin. `base` (bicubic x2 of
    the frozen 512px decode) + the lossless-unshuffled input carry the text detail.
    grad_ckpt=True checkpoints every block (big activation-memory cut, ~30% slower)."""
    def __init__(self, in_ch=128, widths=(128, 256, 512), mid_blocks=28,
                 enc_blocks=(4, 4, 6), dec_blocks=(4, 4), out_ch=3, grad_ckpt=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        w0, w1, w2 = widths
        self.unshuffle = nn.PixelUnshuffle(2)                     # 128@512 -> 512@256
        self.intro = nn.Conv2d(in_ch * 4, w0, 3, 1, 1)
        self.enc0 = nn.Sequential(*[NAFBlock(w0) for _ in range(enc_blocks[0])])
        self.down0 = nn.Conv2d(w0, w1, 2, 2)                      # 256 -> 128
        self.enc1 = nn.Sequential(*[NAFBlock(w1) for _ in range(enc_blocks[1])])
        self.down1 = nn.Conv2d(w1, w2, 2, 2)                      # 128 -> 64
        self.enc2 = nn.Sequential(*[NAFBlock(w2) for _ in range(enc_blocks[2])])
        self.mid = nn.Sequential(*[NAFBlock(w2) for _ in range(mid_blocks)])   # @64, deep
        self.up1 = nn.Sequential(nn.Conv2d(w2, w1 * 4, 1), nn.PixelShuffle(2))  # 64 -> 128
        self.dec1 = nn.Sequential(*[NAFBlock(w1) for _ in range(dec_blocks[0])])
        self.up0 = nn.Sequential(nn.Conv2d(w1, w0 * 4, 1), nn.PixelShuffle(2))  # 128 -> 256
        self.dec0 = nn.Sequential(*[NAFBlock(w0) for _ in range(dec_blocks[1])])
        # thin tail: 256 -> 512 -> 1024 (pixel-shuffle), residual head (starts at 0)
        wt = max(48, w0 // 2)
        self.up_a = nn.Sequential(nn.Conv2d(w0, wt * 4, 3, 1, 1), nn.PixelShuffle(2))   # 256->512
        self.up_b = nn.Sequential(nn.Conv2d(wt, wt * 4, 3, 1, 1), nn.PixelShuffle(2))   # 512->1024
        self.out = nn.Conv2d(wt, out_ch, 3, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def _run(self, seq, x):
        if self.grad_ckpt and self.training:
            for blk in seq:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            return x
        return seq(x)

    def forward(self, feats, base):
        x = self.intro(self.unshuffle(feats))
        e0 = self._run(self.enc0, x)
        e1 = self._run(self.enc1, self.down0(e0))
        e2 = self._run(self.enc2, self.down1(e1))
        m = self._run(self.mid, e2)
        d1 = self._run(self.dec1, self.up1(m) + e1)
        d0 = self._run(self.dec0, self.up0(d1) + e0)
        res = self.out(F.silu(self.up_b(F.silu(self.up_a(d0)))))
        return base + res


# ----------------------------------------------------------------------------
# Variant C: DC-AE native decoder-extension head
# ----------------------------------------------------------------------------
class DecUpBlockHead(nn.Module):
    """Variant C. Continue the frozen decoder with its OWN block vocabulary, used
    the way DC-AE actually uses it: EfficientViT (linear attention) + ResBlock at
    LOW resolution, then native DCUpBlock2d pixel-shuffle upsamples back to 1024.

      feats 128ch@512  --pixel_unshuffle(down_factor)-->  proj to `width` @ (512/df)
        -> N x [ResBlock ... EfficientViTBlock]           (deep, low-res: cheap)
        -> DCUpBlock2d x2 ... up to 1024, halving channels each step, ResBlocks between
        -> RMSNorm + SiLU + conv_out 3ch (residual on `base`).

    down_factor=4 -> work at 128^2 (df=2 -> 256^2). This mirrors the frozen decoder
    (ViT deep/low-res, pixel-shuffle ups) instead of forcing 512^2 attention (OOM)."""
    def __init__(self, in_ch=128, width=512, n_blocks=10, up_ch=128, n_up_blocks=1,
                 out_ch=3, use_vit=True, down_factor=4, grad_ckpt=False):
        super().__init__()
        assert down_factor in (2, 4)
        self.grad_ckpt = grad_ckpt
        self.down_factor = down_factor
        self.unshuffle = nn.PixelUnshuffle(down_factor)          # 512 -> 512/df res
        self.proj = nn.Conv2d(in_ch * down_factor * down_factor, width, 3, 1, 1)
        blocks = []
        for i in range(n_blocks):
            if use_vit and i >= n_blocks // 3:                   # first third ResBlock, rest ViT
                blocks.append(EfficientViTBlock(width, norm_type="rms_norm"))
            else:
                blocks.append(ResBlock(width, width, norm_type="rms_norm", act_fn="silu"))
        self.body = nn.ModuleList(blocks)
        # native pixel-shuffle ups from (512/df) back to 1024: that is log2(1024*df/512)=1+log2(df) steps
        n_up = 1 + (down_factor // 2)                            # df=2 -> 2 ups (256->512->1024); df=4 -> 3 ups
        ups, up_bodies, ch = [], [], width
        for k in range(n_up):
            oc = up_ch if k == n_up - 1 else max(up_ch, ch // 2)
            ups.append(DCUpBlock2d(ch, oc, interpolate=False, shortcut=False))
            up_bodies.append(nn.Sequential(
                *[ResBlock(oc, oc, norm_type="rms_norm", act_fn="silu") for _ in range(n_up_blocks)]))
            ch = oc
        self.ups = nn.ModuleList(ups)
        self.up_bodies = nn.ModuleList(up_bodies)
        self.norm_out = RMSNorm(ch, 1e-5, elementwise_affine=True, bias=True)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight); nn.init.zeros_(self.conv_out.bias)

    def _ckpt(self, blk, x):
        if self.grad_ckpt and self.training:
            return torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        return blk(x)

    def forward(self, feats, base):
        h = self.proj(self.unshuffle(feats))
        for blk in self.body:
            h = self._ckpt(blk, h)
        for up, body in zip(self.ups, self.up_bodies):
            h = up(h)
            for blk in body:
                h = self._ckpt(blk, h)
        h = self.norm_out(h.movedim(1, -1)).movedim(-1, 1)
        res = self.conv_out(F.silu(h))
        return base + res


# Named ~100M presets (measured on RTX PRO 6000, batch=1):
#   a100 : 107M | fwd 39ms | fwd+bwd 130ms | 18GB  (efficient U-Net, primary)
#   c100 : 106.5M | fwd~110ms | fwd+bwd~330ms| 23GB (native DC-AE decoder extension)
PRESETS = {
    "a100": ("a", dict(widths=(128, 256, 512), mid_blocks=44, enc_blocks=(4, 4, 10), dec_blocks=(4, 4))),
    "c100": ("c", dict(width=704, n_blocks=9, up_ch=128, n_up_blocks=1, use_vit=True, down_factor=4)),
}


def build_preset(name, grad_ckpt=False):
    variant, kw = PRESETS[name]
    return build_head(variant, grad_ckpt=grad_ckpt, **kw)


def build_head(variant, **kw):
    variant = variant.lower()
    if variant in ("a", "feat", "featunet", "unet"):
        return FeatHeadUNet(**{k: v for k, v in kw.items() if k in
                               ("in_ch", "widths", "mid_blocks", "enc_blocks", "dec_blocks",
                                "out_ch", "grad_ckpt")})
    if variant in ("c", "dec", "decup", "native"):
        return DecUpBlockHead(**{k: v for k, v in kw.items() if k in
                                 ("in_ch", "width", "n_blocks", "up_ch", "n_up_blocks",
                                  "out_ch", "use_vit", "down_factor", "grad_ckpt")})
    raise ValueError(f"unknown head variant {variant}")


# ----------------------------------------------------------------------------
# Self-test: param counts + shape + latency
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import time, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    dev = args.device

    builders = {
        "A/FeatHeadUNet-100M": lambda: FeatHeadUNet(widths=(128, 256, 512), mid_blocks=28,
                                                    enc_blocks=(4, 4, 6), dec_blocks=(4, 4)),
        "C/DecUp-df4-vit": lambda: DecUpBlockHead(width=768, n_blocks=12, up_ch=128,
                                                  n_up_blocks=1, use_vit=True, down_factor=4),
        "C/DecUp-df2-vit": lambda: DecUpBlockHead(width=512, n_blocks=10, up_ch=128,
                                                  n_up_blocks=1, use_vit=True, down_factor=2),
    }
    print(f"{'variant':<24} {'params(M)':>10} {'out_shape':>20} {'fwd_ms':>9} {'fwd+bwd_ms':>12} {'mem_GB':>8}")
    for name, build in builders.items():
        try:
            feats = torch.randn(1, 128, 512, 512, device=dev)
            base = torch.randn(1, 3, 1024, 1024, device=dev)
            head = build().to(dev)
            n = sum(p.numel() for p in head.parameters()) / 1e6
            torch.cuda.reset_peak_memory_stats()
            for _ in range(5):
                out = head(feats, base); out.mean().backward(); head.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.iters):
                with torch.no_grad():
                    out = head(feats, base)
            torch.cuda.synchronize(); fwd = (time.time() - t0) / args.iters * 1e3
            t0 = time.time()
            for _ in range(args.iters):
                out = head(feats, base); out.mean().backward(); head.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); fb = (time.time() - t0) / args.iters * 1e3
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"{name:<24} {n:>10.1f} {str(tuple(out.shape)):>20} {fwd:>9.1f} {fb:>12.1f} {mem:>8.1f}")
        except Exception as e:
            print(f"{name:<24} FAILED: {type(e).__name__}: {str(e)[:80]}")
        finally:
            del head, out, feats, base
            torch.cuda.empty_cache()
