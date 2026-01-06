from typing import Optional
from timm.layers import trunc_normal_

import torch
import torch.nn.functional as F
from torch import nn

from .basic_transformer_block import BasicTransformerBlock


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class ResidualMLP(nn.Module):
    def __init__(self, dim_in=768, dim_hidden=1024, dim_out=512, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.dropout = nn.Dropout(dropout)

        self.fc3 = nn.Linear(dim_in, dim_out)

        self.shortcut = nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out)

        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        shortcut = self.fc3(x)
        out = self.norm(out + shortcut)
        return out




class Transformer(nn.Module):
    
    def __init__(self, num_layers=8, dropout=0.0, num_attention_heads=8, condition_dropout=False):
        super().__init__()
        self.num_attention_heads: int = num_attention_heads
        self.attention_head_dim: int = 64
        self.in_channels = 384
        self.out_channels = 384
        self.num_layers: int = num_layers
        self.dropout = dropout
        self.norm_num_groups: int = 32
        self.cross_attention_dim = num_attention_heads * self.attention_head_dim
        self.attention_bias: bool = False
        self.activation_fn: str = "geglu"
        self.only_cross_attention: bool = False
        self.double_self_attention: bool = False
        self.upcast_attention: bool = False
        self.norm_type: str = "layer_norm"
        self.norm_elementwise_affine: bool = True
        self.gradient_checkpointing: bool = False
        self.condition_dropout = condition_dropout
        self.condition_dropout_prob = 0.5
        self.configure()


    def configure(self) -> None:
        inner_dim = self.num_attention_heads * self.attention_head_dim

        linear_cls = nn.Linear

        self.norm = torch.nn.GroupNorm(
            num_groups=self.norm_num_groups,
            num_channels=self.in_channels,
            eps=1e-6,
            affine=True,
        )
        self.proj_in = linear_cls(self.in_channels, inner_dim)
        self.condition_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim)
        )

        if self.condition_dropout:
            self.null_token = nn.Parameter(torch.randn(258, 768))

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    dropout=self.dropout,
                    cross_attention_dim=self.cross_attention_dim,
                    activation_fn=self.activation_fn,
                    attention_bias=self.attention_bias,
                    only_cross_attention=self.only_cross_attention,
                    double_self_attention=self.double_self_attention,
                    upcast_attention=self.upcast_attention,
                    norm_type=self.norm_type,
                    norm_elementwise_affine=self.norm_elementwise_affine,
                )
                for d in range(self.num_layers)
            ]
        )

        self.out_channels = (
            self.in_channels
            if self.out_channels is None
            else self.out_channels
        )

        self.proj_out = linear_cls(inner_dim, self.out_channels)

        self.gradient_checkpointing = self.gradient_checkpointing


    def forward(
        self,
        hidden_states: torch.Tensor,                                # b, 384(f), 512(n)
        encoder_hidden_states: Optional[torch.Tensor] = None,       # b, 258(n), 768(f)
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.

        Returns:
            torch.FloatTensor
        """
        
        if attention_mask is not None and attention_mask.ndim == 2:
        
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch, _, seq_len = hidden_states.shape
        residual = hidden_states

        if self.condition_dropout:
            if self.training:
                if torch.rand(1).item() < self.condition_dropout_prob:
                    encoder_hidden_states = self.null_token.unsqueeze(0).expand(batch, -1, -1)
        # encoder_hidden_states = self.null_token.unsqueeze(0).expand(batch, -1, -1)

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 1).reshape(
            batch, seq_len, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)         # b, 512, 512
        encoder_hidden_states = self.condition_proj(encoder_hidden_states)      # b, 258, 512


        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )       # b, 8192, 1024


        # 3. Output
        hidden_states = self.proj_out(hidden_states)        # b, 512, 384

        output = hidden_states + residual.permute(0, 2, 1).contiguous()     # b, 512, 384


        return output
