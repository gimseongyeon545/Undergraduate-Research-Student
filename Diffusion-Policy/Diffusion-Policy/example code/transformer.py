# models/transformer_joint_gripper_no_dq.py
from typing import Optional
import math
import torch
import torch.nn as nn

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                # [L, D]
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return x + self.pe[:, :x.size(1)]

# -----------------------------
# Lightweight CNN Encoder
# -----------------------------
class ConvEncoder(nn.Module):
    """입력 크기 무관(CxHxW). AdaptiveAvgPool로 벡터화."""
    def __init__(self, in_ch: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(),   # H/2, W/2
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   nn.ReLU(),   # H/4, W/4
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  nn.ReLU(),   # H/8, W/8
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Linear(128, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*K, C, H, W] -> [B*K, d_out]
        h = self.net(x).flatten(1)
        return self.proj(h)

# -----------------------------
# Low-dim q Encoder (7D only)
# -----------------------------
class LowDimEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, in_dim] -> [B, K, d_model]
        return self.net(x)

# -----------------------------
# Transformer (no dq)
# -----------------------------
class TransformerJointGripperNoDQ(nn.Module):
    """
    입력 (모두 정규화된 값):
      - obs_rgb:   [B, K, 3, H, W]   (0~1)
      - obs_depth: [B, K, 1, H, W]   (0~1)
      - obs_q:     [B, K, 7]         ([-1,1])   ← dq 없음

    출력:
      - pred_joint: [B, T, 7]        ([-1,1])   ← 절대 joint 6 + grip(마지막)

    구조:
      (RGB CNN) + (Depth CNN) + (q MLP) → concat → Linear(d_model)
      → PosEnc → TransformerEncoder
      → Learned future queries(T) + PosEnc → TransformerDecoder
      → MLP head → 7D (tanh로 [-1,1])
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        img_feat_dim: int = 128,     # CNN 임베딩 차원
        q_feat_dim: int = 256,       # q 임베딩 차원
        pred_horizon: int = 16,
        out_dim: int = 7,
        use_tanh_out: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.pred_horizon = pred_horizon
        self.out_dim = out_dim
        self.use_tanh_out = use_tanh_out

        # Encoders
        self.rgb_enc   = ConvEncoder(in_ch=3, d_out=img_feat_dim)
        self.depth_enc = ConvEncoder(in_ch=1, d_out=img_feat_dim)
        self.q_enc     = LowDimEncoder(in_dim=7, d_model=q_feat_dim)

        # Fuse per-time features → d_model
        fuse_in = img_feat_dim + img_feat_dim + q_feat_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Positional encodings
        self.pos_enc_enc = PositionalEncoding(d_model, max_len=4096)
        self.pos_enc_dec = PositionalEncoding(d_model, max_len=4096)

        # Learned future queries (length = pred_horizon)
        self.future_queries = nn.Parameter(torch.zeros(pred_horizon, d_model))
        # nn.init.trunc_normal_(self.future_queries, std=0.02)

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim)
        )

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # True인 위치를 차단. 상삼각 마스크.
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _encode_per_timestep(
        self, rgb: torch.Tensor, depth: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        """
        rgb:   [B, K, 3, H, W]
        depth: [B, K, 1, H, W]
        q:     [B, K, 7]
        return: [B, K, d_model]
        """
        B, K = q.shape[:2]

        # CNN 인코딩 (B*K, C, H, W)
        rgb_bk   = rgb.contiguous().view(B*K, 3, *rgb.shape[-2:])
        depth_bk = depth.contiguous().view(B*K, 1, *depth.shape[-2:])
        rgb_feat   = self.rgb_enc(rgb_bk)     # [B*K, img_feat_dim]
        depth_feat = self.depth_enc(depth_bk) # [B*K, img_feat_dim]
        rgb_feat   = rgb_feat.view(B, K, -1)
        depth_feat = depth_feat.view(B, K, -1)

        # q 임베딩
        q_feat = self.q_enc(q)                # [B, K, q_feat_dim]

        # concat → fuse → pos enc
        fused = torch.cat([rgb_feat, depth_feat, q_feat], dim=-1)  # [B, K, fuse_in]
        tokens = self.fuse(fused)                                   # [B, K, d_model]
        tokens = self.pos_enc_enc(tokens)
        return tokens

    def forward(
        self,
        obs_rgb: torch.Tensor,      # [B,K,3,H,W]
        obs_depth: torch.Tensor,    # [B,K,1,H,W]
        obs_q: torch.Tensor,        # [B,K,7]
        pred_horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """
        return:
          pred_joint: [B, T, 7]  in [-1,1]
        """
        B, K = obs_q.shape[:2]
        T = pred_horizon if pred_horizon is not None else self.pred_horizon
        assert T <= self.pred_horizon

        # Encode observation sequence
        memory = self._encode_per_timestep(obs_rgb, obs_depth, obs_q)  # [B,K,D]
        memory = self.encoder(memory)                                   # [B,K,D]

        # Prepare future queries
        qrys = self.future_queries[:T, :].unsqueeze(0).expand(B, T, self.d_model)  # [B,T,D]
        qrys = self.pos_enc_dec(qrys)

        # Decode
        h_dec = self.decoder(tgt=qrys, memory=memory)  # [B,T,D]
        out = self.head(h_dec)  # [B,T,7]
        return torch.tanh(out) if hasattr(self, "use_tanh_out") and self.use_tanh_out else out
