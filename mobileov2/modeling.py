"""The Mobile-O model: frozen MiniCPM-V-4.6 VLM -> mcptf connector -> SANA DiT -> DC-AE.

Only the inference path is here. The training forward, the flow-matching loss, the SFT
input packing and the non-mcptf connectors are all in the research repo and none of them
are reachable when sampling, so carrying them would only be code a reader has to rule out.

What the checkpoint on the Hub actually contains is the HEAD: 548 DiT tensors + 54
connector tensors. The VLM is frozen during training and comes from `openbmb/MiniCPM-V-4_6`;
the DiT skeleton and the DC-AE come from `Efficient-Large-Model/Sana_600M_512px_diffusers`.
load_model builds those two, then overlays the head with a strict check that every tensor
in the file landed somewhere.
"""
import json
import os

import torch
import torch.nn as nn
from diffusers import AutoencoderDC, DPMSolverMultistepScheduler, SanaTransformer2DModel
from safetensors.torch import load_file
from transformers.generation import GenerationMixin
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import (
    MiniCPMV4_6Merger,
    MiniCPMV4_6PreTrainedModel,
    MiniCPMV4_6VisionModel,
)
from transformers import AutoModel

try:
    from transformers import MiniCPMV4_6Config
except ImportError:  # not re-exported in some builds
    from transformers.models.minicpmv4_6.configuration_minicpmv4_6 import MiniCPMV4_6Config

from huggingface_hub.dataclasses import strict

from .blocks import McptfConditioningProjector
from .source_gate import has_gate, install

SANA_REPO = "Efficient-Large-Model/Sana_600M_512px_diffusers"
VLM_REPO = "openbmb/MiniCPM-V-4_6"


@strict
class MobileOConfig(MiniCPMV4_6Config):
    """MiniCPM-V-4.6's config plus the diffusion-head fields."""
    model_type = "mobileo_minicpm"

    diffusion_name_or_path: str = SANA_REPO
    vlm_num_layers: int = 4
    is_train: bool = False


class MobileOModel(MiniCPMV4_6PreTrainedModel):
    """Submodule names match a native MiniCPM-V-4.6 checkpoint so the VLM loads 1:1."""
    config_class = MobileOConfig

    def __init__(self, config):
        super().__init__(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.vision_tower = MiniCPMV4_6VisionModel._from_config(config.vision_config)
        self.merger = MiniCPMV4_6Merger(config)
        # dit / vae / diffusion_connector are NOT in the VLM checkpoint; they are built
        # after weight loading so post_init() cannot overwrite SANA's pretrained weights.

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def build_diffusion_head(self, sana=SANA_REPO, vlm_num_layers=1):
        self.diffusion_connector = McptfConditioningProjector(
            input_dim=self.config.text_config.hidden_size,
            hidden_dim=512, output_dim=2304, num_layers=vlm_num_layers,
            num_refinement_blocks=2, num_transformer_layers=2,
            num_heads=8, mlp_ratio=4.0,
        )
        self.dit = SanaTransformer2DModel.from_pretrained(
            sana, subfolder="transformer", low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True, torch_dtype=torch.float16)
        self.vae = AutoencoderDC.from_pretrained(sana, subfolder="vae", torch_dtype=torch.float16)
        self.noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(sana, subfolder="scheduler")


class MobileOForCausalLM(MiniCPMV4_6PreTrainedModel, GenerationMixin):
    config_class = MobileOConfig
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = MobileOModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def encode_images(self, pixel_values, target_sizes, num_patches_per_image, downsample_mode=None):
        """MiniCPM vision path. ONE feature block per image (source + slices concatenated),
        so a single IMAGE_TOKEN_INDEX expands to that image's tokens."""
        model = self.get_model()
        downsample_mode = downsample_mode or self.config.downsample_mode
        use_vit_merger = downsample_mode != "4x"

        vt = model.vision_tower
        pixel_values = pixel_values.to(dtype=vt.dtype, device=vt.device)
        target_sizes = target_sizes.to(device=vt.device)

        vision_output = vt(pixel_values, target_sizes=target_sizes, use_vit_merger=use_vit_merger)
        merger_sizes = target_sizes // 2 if use_vit_merger else target_sizes
        per_patch = model.merger(vision_output.last_hidden_state, merger_sizes)

        image_features, idx = [], 0
        for n in num_patches_per_image:
            n = int(n)
            image_features.append(torch.cat(per_patch[idx:idx + n], dim=0))
            idx += n
        return image_features


def n_fuse(conn):
    """How many VLM layers the connector fuses. The weights are authoritative: a config
    field elsewhere in this project can say 4 while the checkpoint has 1."""
    n, lw = conn.fusion.num_layers, conn.fusion.layer_weights
    assert n == lw.shape[0], f"connector fuses {n} layers but has {lw.shape[0]} weights"
    return n


def load_model(ckpt, device, base=VLM_REPO, sana=SANA_REPO, vlm_layers=1):
    """Build the frozen VLM + SANA head, then overlay the checkpoint's dit/connector weights.

    `ckpt` is a local directory or a Hugging Face repo id (downloaded on first use).
    """
    if not os.path.isdir(ckpt):
        from huggingface_hub import snapshot_download
        print(f"[mobileov2] fetching {ckpt} from the Hugging Face Hub", flush=True)
        ckpt = snapshot_download(ckpt)

    cfg_path = os.path.join(ckpt, "config.json")
    if os.path.isfile(cfg_path):
        ctype = json.load(open(cfg_path)).get("connector_type", "mcptf")
        assert ctype == "mcptf", (
            f"this repo only implements the 'mcptf' connector, checkpoint says {ctype!r}")

    model = MobileOForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model.get_model().build_diffusion_head(sana=sana, vlm_num_layers=vlm_layers)

    sd = {}
    for f in sorted(os.listdir(ckpt)):
        if f.endswith(".safetensors"):
            sd.update(load_file(os.path.join(ckpt, f)))
    tgt = {k: v for k, v in sd.items() if ".dit." in k or "diffusion_connector" in k}
    assert tgt, f"no dit/connector weights in {ckpt}"
    assert len(tgt) == len(sd), f"{len(sd) - len(tgt)} tensors in {ckpt} are neither dit nor connector"
    # A dual-stream checkpoint also carries the source gate; it must exist before the load so its two
    # tensors land instead of being reported as unexpected. Single-stream checkpoints are unaffected.
    dual = has_gate(tgt)
    if dual:
        install(model.get_model().dit)
    _, unexpected = model.load_state_dict(tgt, strict=False)
    assert not unexpected, f"{len(unexpected)} tensors did not land: {unexpected[:5]}"
    if dual:
        g = float(torch.tanh(model.get_model().dit.source_gate.gate.detach()).item())
        assert g != 0.0, "dual-stream checkpoint with a gate of exactly 0: the source channel would do nothing"
        print(f"[mobileov2] dual-stream checkpoint: edits feed the source latent through the gate "
              f"(tanh(gate) {g:+.4f})", flush=True)
    return model.to(device, torch.bfloat16).eval()
