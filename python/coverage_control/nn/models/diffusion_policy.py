import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_backbone import CNNBackBone


import torch
import torch.nn as nn

class GaussianDiffusion(nn.Module):
    """
    DDPM / DDIM-style Gaussian diffusion process for 1D actions (u in R^{N x 2}).

    - q_sample: forward diffusion q(u_t | u_0)
    - ddim_step: reverse DDIM update u_t -> u_{t-1}
    - get_sampling_timesteps: generate a sub-sequence of timesteps for fast DDIM
    """

    def __init__(
        self,
        num_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
        betas: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        device = torch.device(device)

        if betas is None:
            betas = torch.linspace(
                beta_start, beta_end, num_steps, dtype=torch.float32, device=device
            )
        else:
            betas = betas.to(device=device, dtype=torch.float32)
            num_steps = betas.shape[0]

        self.num_steps = int(num_steps)

        # β_t, α_t, ᾱ_t
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device, dtype=torch.float32), alphas_cumprod[:-1]],
            dim=0,
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)          # ᾱ_t
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)  # ᾱ_{t-1}

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    # ---- helpers ----
    def _extract(self, buf: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        从长度为 T 的 buffer 中按 batch t (B,) 取出对应值，并 reshape 成 (B, 1, 1, ...)
        """
        out = buf.gather(0, t)
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    # ---- forward diffusion q(u_t | u_0) ----
    def q_sample(
        self,
        actions_0: torch.Tensor,           # (B, N, 2)
        t: torch.Tensor,                   # (B,)
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample u_t ~ q(u_t | u_0) = N( sqrt(ᾱ_t) u_0, (1 - ᾱ_t) I ).
        """
        if noise is None:
            noise = torch.randn_like(actions_0)

        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, actions_0.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, actions_0.shape
        )

        return sqrt_alpha_bar_t * actions_0 + sqrt_one_minus_alpha_bar_t * noise

    # ---- reverse DDIM step u_t -> u_{t-1} ----
    @torch.no_grad()
    def ddim_step(self, x_t, t, t_prev, eps_hat, eta: float = 0.0):
        alpha_bar_t    = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)

        x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)

        sigma_t = (
            eta
            * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
        )

        noise = torch.randn_like(x_t) if eta > 0.0 else torch.zeros_like(x_t)
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)) * eps_hat

        return torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma_t * noise


    # ---- time subsequence for fast DDIM sampling ----
    def get_sampling_timesteps(
        self,
        num_sampling_steps: int | None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Generate a decreasing sequence of timesteps t_T, ..., t_0 for DDIM.

        - If num_sampling_steps is None or >= num_steps, use all steps T-1,...,0.
        - Otherwise, pick an evenly spaced subsequence.
        """
        if device is None:
            device = self.betas.device

        T = self.num_steps

        if (num_sampling_steps is None) or (num_sampling_steps >= T):
            return torch.arange(T - 1, -1, -1, device=device, dtype=torch.long)

        # 等间隔子序列（包含 0 和 T-1）
        step = (T - 1) / float(num_sampling_steps - 1)
        idx = torch.round(torch.arange(0, T, step, device=device)).long()
        # 从大到小（T-1 ... 0）
        idx, _ = torch.sort(idx, descending=True)
        return idx

    # ---- checkpoint I/O ----
    def to_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "num_steps": torch.tensor(self.num_steps, dtype=torch.int64),
            "betas": self.betas.detach().clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: torch.device | str = "cpu",
    ) -> "GaussianDiffusion":
        device = torch.device(device)
        betas = state_dict.get("betas", None)
        if betas is not None:
            betas = betas.to(device=device, dtype=torch.float32)
            num_steps = int(betas.shape[0])
            return cls(num_steps=num_steps, betas=betas, device=device)
        else:
            num_steps = int(state_dict["num_steps"])
            beta_start = float(state_dict.get("beta_start", 1e-4))
            beta_end = float(state_dict.get("beta_end", 2e-2))
            return cls(
                num_steps=num_steps,
                beta_start=beta_start,
                beta_end=beta_end,
                device=device,
            )




class RotaryMultiHeadAttention(nn.Module):
    """
    Multi-head attention with 2D disentangled RoPE:
      - head_dim split into two planes: x-plane and y-plane (each = head_dim/2)
      - within each plane, apply standard RoPE on (even, odd) pairs
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_base_theta: float = 10_000.0,
        rope_period: float = 1024.0,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim % 4 == 0, "2D disentangled RoPE requires head_dim % 4 == 0"

        self.use_rope = bool(use_rope)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        plane_dim = self.head_dim // 2
        inv_freq = 1.0 / (rope_base_theta ** (torch.arange(0, plane_dim, 2, dtype=torch.float32) / plane_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.rope_scale = (2.0 * math.pi) / float(rope_period)

    def _sin_cos(self, angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # angles: (B, N) -> (B, 1, N, plane_dim/2)
        freqs = torch.einsum("bn,d->bnd", angles.to(self.inv_freq.dtype), self.inv_freq)
        return freqs.sin().unsqueeze(1), freqs.cos().unsqueeze(1)

    @staticmethod
    def _rotate_pairs(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        # x: (B, H, N, plane_dim)
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        return torch.stack([y0, y1], dim=-1).flatten(-2)

    def _apply_rope_2d(self, x: torch.Tensor, pos_2d: torch.Tensor) -> torch.Tensor:
        # x: (B, H, N, head_dim), pos_2d: (B, N, 2)
        plane_dim = x.shape[-1] // 2
        ang_x = pos_2d[..., 0] * self.rope_scale
        ang_y = pos_2d[..., 1] * self.rope_scale

        sin_x, cos_x = self._sin_cos(ang_x)
        sin_y, cos_y = self._sin_cos(ang_y)

        x_plane, y_plane = x.split(plane_dim, dim=-1)
        x_plane = self._rotate_pairs(x_plane, sin_x, cos_x)
        y_plane = self._rotate_pairs(y_plane, sin_y, cos_y)
        return torch.cat([x_plane, y_plane], dim=-1)

    def forward(
        self,
        x: torch.Tensor,                           # (B, Nq, d_model)
        q_positions: torch.Tensor,                 # (B, Nq, 2)
        kv_positions: torch.Tensor,                # (B, Nk, 2)
        attn_mask: Optional[torch.Tensor] = None,  # (B, Nq, Nk), additive 0/-inf
        kv: Optional[torch.Tensor] = None,         # (B, Nk, d_model)
    ) -> torch.Tensor:
        if kv is None:
            kv = x

        B, Nq, _ = x.shape
        Nk = kv.shape[1]

        q = self.q_proj(x)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        H, D = self.num_heads, self.head_dim
        q = q.view(B, Nq, H, D).transpose(1, 2)  # (B,H,Nq,D)
        k = k.view(B, Nk, H, D).transpose(1, 2)  # (B,H,Nk,D)
        v = v.view(B, Nk, H, D).transpose(1, 2)  # (B,H,Nk,D)

        if self.use_rope:
            q = self._apply_rope_2d(q, q_positions)
            k = self._apply_rope_2d(k, kv_positions)

        logits = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # (B,H,Nq,Nk)
        if attn_mask is not None:
            logits = logits + attn_mask.unsqueeze(1)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B,H,Nq,D)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.d_model)
        return self.out_proj(out)


class SpatialEncoderLayer(nn.Module):
    """
    Spatial transformer encoder block with RoPE-based self-attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        rope_base_theta: float = 10_000.0,
        rope_period: float = 1024.0, 
    ) -> None:
        super().__init__()
        self.self_attn = RotaryMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_base_theta=rope_base_theta,
            rope_period=rope_period,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,             # (B, N, d_model)
        positions: torch.Tensor,     # (B, N, 2)
        attn_mask: torch.Tensor,     # (B, N, N)
    ) -> torch.Tensor:
        # Self-attention
        h = self.self_attn(self.norm1(x), positions, positions, attn_mask)
        x = x + self.dropout(h)

        # Feed-forward
        h = self.ff(self.norm2(x))
        x = x + self.dropout(h)
        return x


class SpatialDecoderLayer(nn.Module):
    """
    Pre-LN decoder block:
      x = x + SA(LN(x))
      x = x + CA(LN(x), memory)
      x = x + FFN(LN(x))
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        rope_base_theta: float = 10_000.0,
        rope_period: float = 1024.0,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = RotaryMultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout,
            rope_base_theta=rope_base_theta, rope_period=rope_period,
            use_rope=use_rope,
        )
        self.cross_attn = RotaryMultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout,
            rope_base_theta=rope_base_theta, rope_period=rope_period,
            use_rope=use_rope,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                 # (B, N, d_model)
        positions: torch.Tensor,         # (B, N, 2)
        memory: torch.Tensor,            # (B, N, d_model)
        memory_positions: torch.Tensor,  # (B, N, 2)
        attn_mask: torch.Tensor,         # (B, N, N)
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), positions, positions, attn_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), positions, memory_positions, attn_mask, kv=memory))
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x




class DiffusionPolicy(nn.Module):
    """
    Multi-agent diffusion policy for coverage control.

    Architecture:
      - CNN backbone encodes local coverage maps for each robot.
      - Encoder: spatial transformer over observation tokens (one per robot).
      - Decoder: spatial transformer over noisy action tokens with cross-attention
                 to encoder outputs.

    Conditioning:
      - local coverage maps for each robot
      - noisy actions at diffusion step t
      - 2D robot positions
      - diffusion timestep t
      - (optional) communication graph edge weights for graph-component masks
    """

    def __init__(self, config: Dict):
        """
        Args
        ----
        config: dict with at least:
            config["CNNBackBone"]: CNN backbone config (same structure as LPAC)
            config["DiffusionModel"]:
                {
                    "DModel": int,                 # Transformer hidden dim
                    "NumEncoderLayers": int,       # optional, default = NumLayers
                    "NumDecoderLayers": int,       # optional, default = NumLayers
                    "NumLayers": int,              # legacy, used if above are missing
                    "NumHeads": int,
                    "Dropout": float,
                    "HiddenMultiplier": int,
                    "AttentionRadius": float,      # in world units, optional
                    "TimeEmbeddingDim": int,       # optional, default = DModel
                    "RoPEBaseTheta": float         # optional, default = 1e4
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
        num_enc_layers = diff_cfg.get("NumEncoderLayers", num_layers)
        num_dec_layers = diff_cfg.get("NumDecoderLayers", num_layers)
        num_heads = diff_cfg.get("NumHeads", 4)
        dropout = diff_cfg.get("Dropout", 0.0)
        ff_mult = diff_cfg.get("HiddenMultiplier", 4)
        self.attn_radius: Optional[float] = diff_cfg.get("AttentionRadius", None)
        rope_base_theta: float = diff_cfg.get("RoPEBaseTheta", 10_000.0)
        rope_period: float = float(diff_cfg.get("RoPEPeriod", 1024.0))

        # Action embedding: 2D actions -> d_model
        self.action_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Position embedding (absolute MLP, RoPE handles relative geometry)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Timestep embedding: sinusoidal + MLP
        time_dim = diff_cfg.get("TimeEmbeddingDim", d_model)
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Encoder stack (RoPE only in the first layer)
        self.encoder_layers = nn.ModuleList([
            SpatialEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
                rope_base_theta=rope_base_theta,
                rope_period=rope_period,
            )
            for _ in range(num_enc_layers)
        ])
        # Decoder stack (RoPE only in the first layer)
        self.decoder_layers = nn.ModuleList([
            SpatialDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
                rope_base_theta=rope_base_theta,
                rope_period=rope_period,
                use_rope=(i == 0),
            )
            for i in range(num_dec_layers)
        ])



        for i in range(1, len(self.encoder_layers)):
            self.encoder_layers[i].self_attn.use_rope = False

        for i in range(1, len(self.decoder_layers)):
            self.decoder_layers[i].self_attn.use_rope = False
            self.decoder_layers[i].cross_attn.use_rope = False


        # Final head to predict noise on actions
        self.noise_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2),
        )

        # Buffers for action normalization (set in training script)
        self.register_buffer("actions_mean", torch.zeros(1, 1, 2), persistent=False)
        self.register_buffer("actions_std", torch.ones(1, 1, 2), persistent=False)


    def _sinusoidal_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
        t = t.float()
        device = t.device

        half_dim = self.time_dim // 2
        denom = max(half_dim - 1, 1)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device, dtype=torch.float32) / denom
        )  # (half_dim,)

        args = t[:, None] * freqs[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half_dim)
        if self.time_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


    # ------------------------------------------------------------------
    # Graph-aware attention mask
    # ------------------------------------------------------------------
    def _compute_component_ids(self, edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute connected-component IDs for each robot using the communication graph.

        Args:
            edge_weights: (B, N, N) adjacency weights

        Returns:
            comp_ids: (B, N) integer component IDs per robot
        """
        B, N, _ = edge_weights.shape
        device = edge_weights.device
        comp_ids = torch.empty(B, N, dtype=torch.long, device=device)

        for b in range(B):
            # Treat edges with positive weight as connected
            adj = edge_weights[b] > 0.0  # (N, N), boolean
            visited = [False] * N
            cid = 0

            for i in range(N):
                if visited[i]:
                    continue
                # Depth-first search to mark this component
                stack = [i]
                visited[i] = True
                comp_ids[b, i] = cid
                while stack:
                    u = stack.pop()
                    neighbors = torch.where(adj[u])[0]
                    for v in neighbors.tolist():
                        if not visited[v]:
                            visited[v] = True
                            comp_ids[b, v] = cid
                            stack.append(v)
                cid += 1

        return comp_ids

    def _build_attention_mask(
            self,
            positions: torch.Tensor,           # (B, N, 2)
            edge_weights: Optional[torch.Tensor],  # (B, N, N) or None
        ) -> torch.Tensor:
        """
        Build a graph-aware attention mask over all robots in the batch.

        The mask combines:
        - a distance-based window (radius self.attn_radius), and
        - a graph-component mask based on connected components of the
            communication graph (if edge_weights is provided).

        Returns:
            attn_mask: (B, N, N) with 0 for allowed edges and -inf for disallowed.
        """
        B, N, _ = positions.shape
        device = positions.device

        # Start with all connections allowed
        allowed = torch.ones(B, N, N, dtype=torch.bool, device=device)

        # Distance-based window mask
        if self.attn_radius is not None and self.attn_radius > 0:
            radius_sq = float(self.attn_radius) ** 2
            # pairwise difference: (B, N, 1, 2) - (B, 1, N, 2) -> (B, N, N, 2)
            diff = positions.unsqueeze(2) - positions.unsqueeze(1)   # (B, N, N, 2)
            dist_sq = (diff ** 2).sum(-1)                            # (B, N, N)
            allowed = allowed & (dist_sq <= radius_sq)

        # Graph-component mask: allow attention only within the same component
        if edge_weights is not None:
            comp_ids = self._compute_component_ids(edge_weights)  # (B, N)
            same_comp = comp_ids.unsqueeze(2) == comp_ids.unsqueeze(1)  # (B, N, N)
            allowed = allowed & same_comp

        # Build additive mask with 0 for allowed edges and -inf otherwise
        attn_mask = torch.zeros_like(allowed, dtype=torch.float32)
        attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
        return attn_mask


    # ------------------------------------------------------------------
    # Encoder / Decoder forward passes
    # ------------------------------------------------------------------
    def _encode_observations(
        self,
        coverage_maps: torch.Tensor,      # (B, N, C, H, W)
        positions: torch.Tensor,          # (B, N, 2)
        edge_weights: Optional[torch.Tensor],  # (B, N, N) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observations into a set of memory tokens.

        Returns:
            memory:    (B, N, d_model)
            attn_mask: (B, N, N) attention mask over robot tokens
        """
        B, N, C, H, W = coverage_maps.shape

        # CNN backbone runs per robot
        x = coverage_maps.view(B * N, C, H, W)
        cnn_feat = self.cnn_backbone(x)               # (B*N, cnn_latent)
        cnn_feat = self.cnn_proj(cnn_feat)            # (B*N, d_model)
        cnn_feat = cnn_feat.view(B, N, self.d_model)  # (B, N, d_model)

        # Position embedding (absolute MLP; RoPE handles relative geometry)
        p_emb = self.pos_mlp(positions)               # (B, N, d_model)

        obs_tokens = cnn_feat + p_emb

        # Graph-aware attention mask
        attn_mask = self._build_attention_mask(positions, edge_weights)  # (B, N, N)

        # Pass through encoder layers
        h = obs_tokens
        for layer in self.encoder_layers:
            h = layer(h, positions, attn_mask)

        memory = h  # (B, N, d_model)
        return memory, attn_mask

    def _decode_actions(
        self,
        actions_t: torch.Tensor,          # (B, N, 2)
        positions: torch.Tensor,          # (B, N, 2)
        t: torch.Tensor,                  # (B,)
        memory: torch.Tensor,             # (B, N, d_model)
        memory_positions: torch.Tensor,   # (B, N, 2)
        attn_mask: torch.Tensor,          # (B, N, N)
    ) -> torch.Tensor:
        """
        Decode noisy actions given encoder memory and conditioning.

        Returns:
            eps_hat: (B, N, 2) predicted noise on actions.
        """
        B, N, _ = actions_t.shape

        # Action embedding
        a_emb = self.action_mlp(actions_t)            # (B, N, d_model)

        # Position embedding for decoder (shared module with encoder)
        p_emb = self.pos_mlp(positions)               # (B, N, d_model)

        # Time embedding
        t_emb = self._sinusoidal_time_embedding(t)    # (B, time_dim)
        t_emb = self.time_mlp(t_emb)                  # (B, d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, d_model)

        dec_tokens = a_emb + p_emb + t_emb            # (B, N, d_model)

        # Pass through decoder layers
        h = dec_tokens
        for layer in self.decoder_layers:
            h = layer(h, positions, memory, memory_positions, attn_mask)

        eps_hat = self.noise_head(h)                  # (B, N, 2)
        return eps_hat

    def forward(
        self,
        coverage_maps: torch.Tensor,          # (B, N, C, H, W)
        actions_t: torch.Tensor,              # (B, N, 2)
        t: torch.Tensor,                      # (B,)
        positions: torch.Tensor,              # (B, N, 2)
        edge_weights: Optional[torch.Tensor] = None,  # (B, N, N)
    ) -> torch.Tensor:
        """
        Predict noise on actions at a given diffusion step t.

        This method is used during training.
        """
        memory, attn_mask = self._encode_observations(
            coverage_maps=coverage_maps,
            positions=positions,
            edge_weights=edge_weights,
        )
        eps_hat = self._decode_actions(
            actions_t=actions_t,
            positions=positions,
            t=t,
            memory=memory,
            memory_positions=positions,
            attn_mask=attn_mask,
        )
        return eps_hat

    @torch.no_grad()
    def sample_actions(
        self,
        coverage_maps: torch.Tensor,         # (B, N, C, H, W)
        positions: torch.Tensor,             # (B, N, 2)
        diffusion: GaussianDiffusion,
        edge_weights: torch.Tensor | None = None,
        num_steps: int | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Run the DDIM reverse process to sample actions u_0 from pure noise.

        Returns:
            actions_0: (B, N, 2), still in *normalized* action space.
        """
        self.eval()
        device = coverage_maps.device
        B, N, C, H, W = coverage_maps.shape

        memory, attn_mask = self._encode_observations(
            coverage_maps=coverage_maps,
            positions=positions,
            edge_weights=edge_weights,
        )

        timesteps = diffusion.get_sampling_timesteps(num_steps, device=device)
        x_t = torch.randn(B, N, 2, device=device)

        for t_scalar, t_prev_scalar in zip(timesteps[:-1], timesteps[1:]):
            t_batch      = torch.full((B,), int(t_scalar),      device=device, dtype=torch.long)
            t_prev_batch = torch.full((B,), int(t_prev_scalar), device=device, dtype=torch.long)

            eps_hat = self._decode_actions(
                actions_t=x_t,
                positions=positions,
                t=t_batch,
                memory=memory,
                memory_positions=positions,
                attn_mask=attn_mask,
            )
            x_t = diffusion.ddim_step(x_t=x_t, t=t_batch, t_prev=t_prev_batch, eps_hat=eps_hat, eta=eta)

        return x_t
