# models/joint_keyframe_regressor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional
from torchvision import models

class ImageNorm(nn.Module):
    """
    입력 rgb0가 [0,1] 범위일 때, ImageNet 통계로 정규화.
    (dataset에서 이미 표준화했다면 disable_imagenet_norm=True로 끄세요)
    """
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std",  std,  persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

class JointKeyframeRegressor(nn.Module):
    """
    입력:  rgb0  — [B, 3, H, W] (0~1)
    출력:  q_hat — [B, 2, 6]  (순서: [grasp, release])
    구조:  ResNet backbone → GAP → MLP → 12 → reshape(2,6)

    Args:
        backbone: "resnet18"|"resnet34"|"resnet50"
        pretrained: torchvision ImageNet 사전학습 가중치 사용
        freeze_backbone: True면 백본 파라미터 고정(학습 X)
        hidden_dim: MLP 중간 차원
        disable_imagenet_norm: True면 내부 정규화 스킵(입력이 이미 표준화된 경우)
        tanh_output: True면 [-1,1]로 출력(정규화 좌표 학습이라면 권장)
    """
    def __init__(
        self,
        backbone: Literal["resnet18","resnet34","resnet50"] = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dim: int = 512,
        disable_imagenet_norm: bool = False,
        tanh_output: bool = True,
    ):
        super().__init__()
        self.tanh_output = tanh_output
        self.norm = None if disable_imagenet_norm else ImageNorm()

        # --- ResNet backbone (fc 제거 후 feature dim 파악)
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # fc 제거하고 conv stem~layer4만 남김
        self.backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
            nn.AdaptiveAvgPool2d((1,1))
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- Head: feat_dim → hidden → 12 → reshape (2,6)
        self.head = nn.Sequential(
            nn.Flatten(),                           # [B, feat_dim, 1,1] → [B, feat_dim]
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 12),              # 2*6
        )

        # 가중치 초기화(MLP만 Xavier)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, rgb0: torch.Tensor) -> torch.Tensor:
        """
        rgb0: [B,3,H,W] in [0,1]
        returns: [B,2,6] (order: [grasp, release])
        """
        if self.norm is not None:
            rgb0 = self.norm(rgb0)
        feat = self.backbone(rgb0)   # [B, feat_dim, 1,1]
        out  = self.head(feat)       # [B, 12]
        if self.tanh_output:
            out = torch.tanh(out)
        out  = out.view(rgb0.size(0), 2, 6)  # [B,2,6]
        return out
