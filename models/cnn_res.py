# model_mahjong.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Conv(3×3)→BN→ReLU→Conv(3×3)→BN + skip"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)  # チャンネル数が変わる場合だけ射影
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class MahjongActorCritic(nn.Module):
    """
    画像特徴 (C=4,H=4,W=16) →
        ResBlock(64) → ResBlock(128) → ResBlock(32) → Flatten
        → FC 1024 （共有）
        → Actor   : FC512×2 → logits (N_action)
        → Critic  : FC512×2 → state-value (1)
    """
    def __init__(self,
                 num_actions: int = 17,
                 extra_feat_dim: int = 0):   # 後段で one-hot 等を連結したい場合
        super().__init__()

        # --- 3-stage Residual CNN ---
        self.stage1 = ResidualBlock(4,   64)   # in=4  out=64
        self.stage2 = ResidualBlock(64, 128)   # in=64 out=128
        self.stage3 = ResidualBlock(128, 32)   # in=128 out=32

        # 32ch × 4 × 16 = 2048 ユニット
        self.flat_dim = 32 * 4 * 16 + extra_feat_dim

        # --- 共有 FC ---
        self.fc_shared = nn.Linear(self.flat_dim, 1024)

        # --- Actor head ---
        self.fc_pi_1 = nn.Linear(1024, 512)
        self.fc_pi_2 = nn.Linear(512, 512)
        self.pi_out  = nn.Linear(512, num_actions)

        # --- Critic head ---
        self.fc_v_1 = nn.Linear(1024, 512)
        self.fc_v_2 = nn.Linear(512, 512)
        self.v_out  = nn.Linear(512, 1)

    # ----------------------------------------------------------
    # img : (B, H, W, C) あるいは (B, C, H, W) どちらでもOK
    # extra_feats : (B, extra_feat_dim) or None
    # ----------------------------------------------------------
    def forward(self, img, extra_feats=None):
        # 画像軸を (B,C,H,W) に揃える
        if img.ndim != 4:
            raise ValueError("img must be 4-D tensor")
        if img.shape[1] != 4:                  # (B,H,W,C) の場合
            img = img.permute(0, 3, 1, 2)      # → (B,C,H,W)

        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.flatten(1)                       # (B, 2048)

        if extra_feats is not None:
            x = torch.cat([x, extra_feats], dim=-1)

        x = F.relu(self.fc_shared(x))

        # --- Actor ---
        a = F.relu(self.fc_pi_1(x))
        a = F.relu(self.fc_pi_2(a))
        logits = self.pi_out(a)

        # --- Critic ---
        v = F.relu(self.fc_v_1(x))
        v = F.relu(self.fc_v_2(v))
        value = self.v_out(v).squeeze(-1)      # (B,)

        return logits, value