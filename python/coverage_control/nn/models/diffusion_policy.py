import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_backbone import CNNBackBone


class DiffusionPolicy(nn.Module):
    """
    Multi-agent diffusion policy for coverage control.

    Conditioning:
      - local coverage maps for each robot
      - robot 2D positions
      - noisy actions a_t
      - diffusion timestep t

    Output:
      - epsilon_hat: predicted Gaussian noise on actions, same shape as a_t
    """

    def __init__(self, config: Dict):
        """
        Args
        ----
        config: dict with at least:
            config["CNNBackBone"]: CNN backbone config (same structure as LPAC)
            config["DiffusionModel"]:
                {
                    "DModel": int,               # Transformer hidden dim
                    "NumLayers": int,
                    "NumHeads": int,
                    "Dropout": float,
                    "HiddenMultiplier": int,
                    "AttentionRadius": float,    # in world units, optional
                    "TimeEmbeddingDim": int      # optional, default = DModel
                }
        """
        super().__init__()

        cnn_cfg = config["CNNBackBone"]
        diff_cfg = config["DiffusionModel"]

        # CNN encoder for local coverage maps
        self.cnn_backbone = CNNBackBone(cnn_cfg)
        cnn_latent = self.cnn_backbone.latent_size

        # Transformer hidden dim; project CNN latent to this dim
        d_model = diff_cfg.get("DModel", 128)
        self.d_model = d_model
        self.cnn_proj = nn.Linear(cnn_latent, d_model)

        num_layers = diff_cfg.get("NumLayers", 4)
        num_heads = diff_cfg.get("NumHeads", 4)
        dropout = diff_cfg.get("Dropout", 0.0)
        ff_mult = diff_cfg.get("HiddenMultiplier", 4)
        self.attn_radius: Optional[float] = diff_cfg.get("AttentionRadius", None)

        # Action and position embeddings (both go to d_model)
        self.action_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Timestep embedding: sinusoidal + MLP
        time_dim = diff_cfg.get("TimeEmbeddingDim", d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.time_dim = time_dim

        # Multi-layer Transformer encoder over all agents
        # 所有 robot (B * N) 展成一条序列，batch_size=1，
        # 用 block-diagonal 的 attention mask 控制“只在同一个图、且局部邻域内”互相注意。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final head to predict noise on 2D actions
        self.noise_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2),
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def _sinusoidal_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Standard sinusoidal embedding for diffusion timestep.

        Args:
            t: (B,) or (B, 1) integer timesteps in [0, T-1]

        Returns:
            (B, time_dim) tensor
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        t = t.float()
        device = t.device

        half_dim = self.time_dim // 2
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(10000.0),
                steps=half_dim,
                device=device,
            )
            * (-1.0 / (half_dim - 1))
        )
        # (B, half_dim)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.time_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def _build_attention_mask(
        self,
        positions: torch.Tensor,
        batch_size: int,
        world_size: int,
    ) -> Optional[torch.Tensor]:
        """
        Build a block-diagonal attention mask with optional local neighborhood.

        Args:
            positions: (B, N, 2)
            batch_size: B
            world_size: N

        Returns:
            attn_mask of shape (B*N, B*N) with 0 for allowed edges and -inf for disallowed,
            or None if no masking is requested.
        """
        device = positions.device
        S = batch_size * world_size

        # 禁止不同图之间互相注意（block-diagonal）
        mask = torch.full((S, S), float("-inf"), device=device)

        if self.attn_radius is None or self.attn_radius <= 0:
            # 不做局部邻域，只 block 掉跨图的 attention
            for b in range(batch_size):
                i0 = b * world_size
                i1 = (b + 1) * world_size
                mask[i0:i1, i0:i1] = 0.0
            return mask

        radius_sq = float(self.attn_radius) ** 2
        pos = positions.view(batch_size, world_size, -1)  # (B, N, 2)

        for b in range(batch_size):
            pos_b = pos[b]  # (N, 2)
            diff = pos_b.unsqueeze(1) - pos_b.unsqueeze(0)  # (N, N, 2)
            dist_sq = (diff ** 2).sum(-1)

            # allowed: 机器人 i, j 间距在半径内
            allowed = dist_sq <= radius_sq
            block = torch.where(
                allowed,
                torch.zeros_like(dist_sq, dtype=torch.float32, device=device),
                torch.full_like(dist_sq, float("-inf"), dtype=torch.float32, device=device),
            )

            i0 = b * world_size
            i1 = (b + 1) * world_size
            mask[i0:i1, i0:i1] = block

        return mask


    def forward(
        self,
        coverage_maps: torch.Tensor,
        actions_t: torch.Tensor,
        t: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coverage_maps: (B, N, C, H, W)
            actions_t:     (B, N, 2) noisy actions at timestep t
            t:             (B,) or (B, 1) diffusion timesteps
            positions:     (B, N, 2) robot 2D positions (same as LPAC pos)

        Returns:
            eps_hat: (B, N, 2) predicted noise on actions
        """
        B, N, C, H, W = coverage_maps.shape

        # 1) CNN features from local maps
        cnn_in = coverage_maps.view(B * N, C, H, W)
        feat = self.cnn_backbone(cnn_in)               # (B*N, cnn_latent)
        feat = self.cnn_proj(feat)                     # (B*N, d_model)
        feat = feat.view(B, N, self.d_model)           # (B, N, d_model)

        # 2) Action and position embeddings
        a_emb = self.action_mlp(actions_t)             # (B, N, d_model)
        p_emb = self.pos_mlp(positions)                # (B, N, d_model)

        # 3) Time embedding
        t_emb = self._sinusoidal_time_embedding(t)     # (B, time_dim)
        t_emb = self.time_mlp(t_emb)                   # (B, d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)   # (B, N, d_model)

        # 4) Build tokens and attention mask
        tokens = feat + a_emb + p_emb + t_emb          # (B, N, d_model)
        tokens_flat = tokens.view(B * N, self.d_model) # (S, d_model), S = B*N

        attn_mask = self._build_attention_mask(positions, B, N)  # (S, S)

        # Transformer expects (seq_len, batch, dim); here batch=1, seq_len=S
        tokens_seq = tokens_flat.unsqueeze(1)          # (S, 1, d_model)
        h = self.transformer(tokens_seq, mask=attn_mask)  # (S, 1, d_model)
        h = h.squeeze(1).view(B, N, self.d_model)      # (B, N, d_model)

        # 5) Predict noise on actions
        eps_hat = self.noise_head(h)                   # (B, N, 2)
        return eps_hat
