"""The mcptf conditioning projector: the connector in the released checkpoints.

Extracted verbatim from Mobile-O's mobile_block.py -- exactly the transitive closure of
what `McptfConditioningProjector` instantiates, nothing else. The repo's other projector
variants (conv, transformer, and the 2D-spatial blocks) are not reachable from these
checkpoints and are not here.

Shapes are fixed by the released weights, not free: input 1024 (MiniCPM hidden), hidden
512, output 2304 (SANA cross-attention), 2 conv refiners + 2 transformer refiners,
8 heads, mlp_ratio 4.0 -> 54 tensors. Changing any of them makes the strict state_dict
load in modeling.load_model fail, which is the intended guard.
"""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


class EfficientChannelAttention(nn.Module):
    """
    Lightweight channel attention mechanism.
    Uses global average pooling + small MLP for channel reweighting.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 8)  # Minimum 8 channels
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.Hardswish(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Hardsigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        # Global context
        y = self.avg_pool(x).view(b, c)
        # Channel importance weights
        y = self.fc(y).view(b, c, 1, 1)
        # Reweight channels
        return x * y


class DepthwiseSeparableConv1D(nn.Module):
    """1D depthwise separable convolution for sequence data."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.Hardswish()
    
    def forward(self, x):
        # x: [B, C, N] - channel-first format
        x = self.depthwise(x)
        x = self.pointwise(x)

        x = x.transpose(1, 2)  # [B, N, C]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, C, N]

        x = self.act(x)
        return x


class SequenceRefinementBlock(nn.Module):
    """Mobile-friendly block for 1D sequence refinement."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = DepthwiseSeparableConv1D(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.channel_attn = EfficientChannelAttention(channels, reduction=4)
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, C] - sequence format
        Returns:
            Refined features [B, N, C]
        """
        b, n, c = x.shape
        identity = x
        
        # 1D convolution: [B, N, C] → [B, C, N] → process → [B, C, N] → [B, N, C]
        x = x.transpose(1, 2)  # [B, C, N]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, N, C]
        
        # Channel attention (works on any sequence length)
        x_2d = x.transpose(1, 2).unsqueeze(-1)  # [B, C, N, 1] for attention
        x_2d = self.channel_attn(x_2d)
        x = x_2d.squeeze(-1).transpose(1, 2)
        
        # Residual + norm
        x = x + identity
        x = self.norm(x)
        
        return x


class TransformerRefinementBlock(nn.Module):
    """Pre-norm transformer block for global sequence mixing.

    The depthwise-conv refiners only mix within a kernel_size window; this block
    gives the projector a global receptive field over the VLM token sequence
    before it is projected to the DiT conditioning dimension.

    The residual branches are zero-initialised (`gamma`), so at init the block is
    an exact identity and the projector behaves like the conv-only version.
    """

    def __init__(self, channels, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
        )
        # LayerScale, zero-init -> identity at step 0
        self.gamma1 = nn.Parameter(torch.zeros(channels))
        self.gamma2 = nn.Parameter(torch.zeros(channels))

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: [B, N, C]
            key_padding_mask: [B, N] bool, True at positions to IGNORE
        Returns:
            [B, N, C]
        """
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.gamma1 * h
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


class McptfLayerwiseFusion(nn.Module):
    """1D sequence-based fusion - NO spatial assumptions."""
    def __init__(
        self,
        num_layers=4,
        input_dim=896,
        hidden_dim=512,
        output_dim=2304,
        num_refinement_blocks=2,
        num_transformer_layers=2,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # Learnable layer weights
        #self.layer_weights = nn.Parameter(torch.ones(num_layers))
        #self.register_buffer('temperature', torch.tensor(2.0))
        
        self.layer_weights = nn.Parameter(torch.linspace(0.8, 1.2, num_layers))
        self.register_buffer('temperature', torch.tensor(2.0))


        # Stage 1: Compression
        self.input_compress = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Hardswish()
        )
        
        # Stage 2: 1D sequence refinement (NO 2D convolutions!)
        self.sequence_refiners = nn.ModuleList([
            SequenceRefinementBlock(hidden_dim, kernel_size=3)
            for _ in range(num_refinement_blocks)
        ])
        
        # Stage 2b: global sequence mixing (conv refiners are local-only)
        self.num_transformer_layers = num_transformer_layers
        self.transformer_refiners = nn.ModuleList([
            TransformerRefinementBlock(hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_transformer_layers)
        ])

        # Stage 3: Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        #with torch.no_grad():
        #    self.layer_weights.data = torch.linspace(0.8, 1.2, num_layers)
    
    def forward(
        self,
        hidden_states_list: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states_list: List of [B, N, D] tensors (variable N is OK!)
            attention_mask: optional [B, N] with 1 for real tokens, 0 for padding.
                Only used by the transformer refiners; the conv path is local and
                unaffected. Strongly recommended when the batch contains padding.
        Returns:
            [B, N, output_dim] - preserves sequence length
        """
        # Select last N layers
        selected_layers = hidden_states_list[-self.num_layers:]
     
        # Weighted fusion
        weights = torch.softmax(self.layer_weights / self.temperature, dim=0)
        b, n, d = selected_layers[0].shape
        fused = torch.zeros_like(selected_layers[0])
        
        for i, layer_hidden_states in enumerate(selected_layers):
            fused = fused + weights[i] * layer_hidden_states
         
        # Stage 1: Compress
        fused = self.input_compress(fused)  # [B, N, hidden_dim]
        
        # Stage 2: 1D sequence refinement
        for refiner in self.sequence_refiners:
            fused = refiner(fused)  # [B, N, hidden_dim]

        # Stage 2b: global mixing
        if len(self.transformer_refiners) > 0:
            key_padding_mask = None
            if attention_mask is not None:
                # nn.MultiheadAttention wants True == "ignore this position"
                key_padding_mask = ~attention_mask.bool()
            for refiner in self.transformer_refiners:
                fused = refiner(fused, key_padding_mask=key_padding_mask)

        # Stage 3: Project to output
        output = self.output_proj(fused)  # [B, N, output_dim]
        
        return output
    
    def anneal_temperature(self, epoch, total_epochs, min_temp=0.5):
        progress = epoch / total_epochs
        new_temp = min_temp + (2.0 - min_temp) * 0.5 * (1 + math.cos(math.pi * progress))
        self.temperature.fill_(new_temp)


class McptfConditioningProjector(nn.Module):
    """
    Complete mobile-optimized conditioning projector.
    Replaces the standard DiffusionConnector in your model.
    
    Usage:
        self.diffusion_connector = MobileConditioningProjector(
            input_dim=896,      # VLM output dimension
            hidden_dim=512,     # Mobile-friendly intermediate dim
            output_dim=2304,    # DiT conditioning dimension
            num_layers=4,       # Fuse last 4 layers
        )
    
    Forward pass:
        # Pass ALL hidden states (tuple/list of tensors):
        encoder_hidden_states = self.diffusion_connector(outputs.hidden_states)
    """
    def __init__(
        self, 
        input_dim: int = 1536,
        hidden_dim: int = 512, 
        output_dim: int = 2304,
        num_layers: int = 8,
        num_refinement_blocks: int = 2,
        spatial_size: int = 24,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        """
        Args:
            input_dim: VLM hidden dimension (default: 896 for Qwen2-0.5B)
            hidden_dim: Intermediate dimension for spatial processing (default: 512)
            output_dim: DiT conditioning dimension (default: 2304 for SANA)
            num_layers: Number of VLM layers to fuse (default: 4)
            num_refinement_blocks: Number of spatial refinement blocks (default: 2)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_refinement_blocks = num_refinement_blocks
        self.num_transformer_layers = num_transformer_layers

        print(f"[Mobile Conditioning] Initializing:")
        print(f"  Input:  {input_dim}")
        print(f"  Hidden: {hidden_dim}")
        print(f"  Output: {output_dim}")
        print(f"  Fusing last {num_layers} VLM layers")
        print(f"  Refinement blocks: {num_refinement_blocks}")
        print(f"  Transformer layers: {num_transformer_layers} (heads={num_heads}, mlp_ratio={mlp_ratio})")
        
        # Main fusion module
        self.fusion = McptfLayerwiseFusion(
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_refinement_blocks=num_refinement_blocks,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        
    def forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Can be either:
                - Tuple/List of tensors [B, N, D] (all VLM layers - RECOMMENDED)
                - Single tensor [B, N, D] (fallback - uses last layer only)
        
        Returns:
            Conditioning features [B, N, output_dim] for DiT cross-attention
        """
        # Handle both single tensor and list/tuple of tensors
        if isinstance(hidden_states, torch.Tensor):
            # Fallback: single tensor (last layer only)
            # Duplicate to match expected number of layers
            print("[WARNING] MobileConditioning received single tensor. "
                  "For best results, pass all hidden states as a list/tuple.")
            hidden_states_list = [hidden_states] * self.num_layers
        elif isinstance(hidden_states, (list, tuple)):
            # Recommended: list/tuple of all VLM layer outputs
            hidden_states_list = list(hidden_states)
        else:
            raise TypeError(
                f"Expected Tensor, List[Tensor], or Tuple[Tensor], "
                f"got {type(hidden_states)}"
            )
        
        # Verify input dimensions
        if len(hidden_states_list) > 0:
            expected_dim = self.input_dim
            actual_dim = hidden_states_list[0].shape[-1]
            
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Input dimension mismatch! "
                    f"Expected {expected_dim}, got {actual_dim}. "
                    f"Model was initialized with input_dim={self.input_dim}, "
                    f"but received tensors with last dimension {actual_dim}. "
                    f"\n"
                    f"This usually means:\n"
                    f"1. The model checkpoint was saved with different dimensions\n"
                    f"2. You need to delete the old checkpoint and retrain\n"
                    f"3. Or the VLM output dimension changed"
                )
        
        # Process through fusion module
        output = self.fusion(hidden_states_list, attention_mask=attention_mask)

        return output
    
    def get_layer_importance(self) -> torch.Tensor:
        """Returns normalized importance weights of each fused layer."""
        weights = torch.softmax(
            self.fusion.layer_weights / self.fusion.temperature, 
            dim=0
        )
        return weights
    
    def anneal_temperature(self, epoch: int, total_epochs: int):
        """
        Anneal temperature during training for better layer selection.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of training epochs
        """
        self.fusion.anneal_temperature(epoch, total_epochs)
    
    def get_metrics(self) -> dict:
        """Get current conditioning metrics for logging."""
        weights = self.get_layer_importance()
        temp = self.fusion.temperature.item()
        
        metrics = {
            'temperature': temp,
            'layer_weights': weights.cpu().tolist(),
        }
        
        
        for i, w in enumerate(weights):
            layer_idx = 24 - self.num_layers + i + 1  # e.g., 21, 22, 23, 24
            metrics[f'layer_{layer_idx}_weight'] = w.item()
        
        return metrics
