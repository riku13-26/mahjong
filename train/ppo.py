#!/usr/bin/env python3
"""
日本麻雀（簡易 1vs1）環境で PPO (Proximal Policy Optimization) を学習する最小実装。

ポイント
---------
* Environment : `env_jpn.mahjong_env.MahjongEnv`
* Model       : `models.cnn_res.MahjongActorCritic`
* Algorithm   : PPO（クリッピング・GAE 付き）

外部 RL ライブラリ（Stable‑Baselines 等）は使わず、
**PPO の中身を読んで理解できる構成** にしてあります。
"""
from __future__ import annotations
import os
import time
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import wandb

wandb.login(key='')

# --- 自作モジュールのインポート ---
from env_jpn.mahjong_env import MahjongEnv
from models.cnn_res import MahjongActorCritic


# --------------------------------------------------------
# ============================================================================
# PPO のハイパーパラメータとその役割
# ----------------------------------------------------------------------------
# total_steps   : 何手分シミュレーションを回すか（1 手 ≒ draw_phase → discard_phase）
# update_every  : ロールアウトを何手ぶん貯めてから 1 回 PPO 更新を行うか
#                 例) total_steps=20_000, update_every=1_024 → 約 19 回更新
# batch_size    : PPO 更新時に update_every サンプルを何件ずつミニバッチに
#                 分割して勾配計算するか (update_every % batch_size == 0 が望ましい)
# epochs        : 上記ミニバッチを何エポック繰り返すか
# ----------------------------------------------------------------------------
# → 1 回の PPO 更新での勾配計算回数
#      = (update_every / batch_size) * epochs
# ============================================================================
@dataclass
class PPOConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    total_steps: int = 2_0000        # ←お好みで
    update_every: int = 1_024        # バッファに貯めるステップ数
    epochs: int = 4                  # policy update 更新エポック
    batch_size: int = 256
    gamma: float = 0.99
    lam: float = 0.95                # GAE
    clip_eps: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    seed: int = 42
    log_interval: int = 1_000
    wandb_project: str = "mahjong-rl"
    wandb_runname: str = "run_ppo_cnn"


# --------------------------------------------------------
# ロールアウトを溜めるバッファ
# --------------------------------------------------------
class RolloutBuffer:
    """
    学習用サンプルを一時的に格納するクラス。

    * store()        : ステップごとに観測・行動などを保存
    * finish_path()  : エピソード or バッファ末端で呼び出し、GAE-Lambda で
                        advantage とリターンを計算
    * get()          : PPO 更新のための dict を返す（advantage は標準化済み）
    """
    def __init__(self, size: int, obs_shape: tuple[int, ...], device: str):
        self.size = size
        self.device = device
        self.obs_buf = torch.zeros((size, *obs_shape), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros(size, dtype=torch.int64, device=device)
        self.logp_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.val_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.adv_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr = 0

    def store(self, obs, act, logp, rew, done, val):
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.logp_buf[idx] = logp
        self.rew_buf[idx] = rew
        self.done_buf[idx] = done
        self.val_buf[idx] = val
        self.ptr += 1

    def is_full(self):
        return self.ptr >= self.size

    # --------------------------------------------------------------------
    # finish_path():
    #   * GAE(λ)  (Generalized Advantage Estimation)
    #       - TD 誤差 δ_t = r_t + γ V(s_{t+1}) − V(s_t)
    #       - advantage A_t = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}
    #         すなわち λ で指数減衰させた “未来の TD 誤差” の総和
    #
    #   * TD 目標 (return) R_t = A_t + V(s_t)
    #       - critic の教師信号として使用
    #
    #   ここではバッファ末端 (ptr) から逆順に計算し、
    #   done=1 のステップではブートストラップを切っている。
    # --------------------------------------------------------------------
    def finish_path(self, last_val: float, gamma: float, lam: float):
        # GAE-Lambda で advantage と TD 目標を計算
        rew = self.rew_buf.cpu().numpy()
        val = self.val_buf.cpu().numpy()
        done = self.done_buf.cpu().numpy()
        adv = np.zeros_like(rew)
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            mask = 1.0 - done[t]
            next_val = last_val if t == self.ptr - 1 else val[t + 1]
            delta = rew[t] + gamma * next_val * mask - val[t]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        self.adv_buf[:self.ptr] = torch.as_tensor(adv, device=self.device)
        self.ret_buf[:self.ptr] = self.adv_buf[:self.ptr] + self.val_buf[:self.ptr]

    def get(self):
        assert self.is_full()
        # advantage を標準化
        adv = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-8)
        dataset = {
            "obs": self.obs_buf,
            "act": self.act_buf,
            "old_logp": self.logp_buf,
            "ret": self.ret_buf,
            "adv": adv,
        }
        return dataset


# --------------------------------------------------------
# 補助関数 : 非合法手を -inf マスクして行動をサンプリング
# --------------------------------------------------------
def select_action(model: nn.Module, obs_img: np.ndarray, legal_actions: list[int], device: str):
    # obs_img : (H,W,C) float32
    obs_t = torch.from_numpy(obs_img).to(device).unsqueeze(0)  # (1,H,W,C)
    logits, value = model(obs_t)
    logits = logits.squeeze(0)

    # 非合法手は -inf にして除外
    mask = torch.full_like(logits, float('-inf'))
    mask[legal_actions] = 0.0
    masked_logits = logits + mask
    dist = torch.distributions.Categorical(logits=masked_logits)
    act = dist.sample()
    logp = dist.log_prob(act)
    return act.item(), logp.item(), value.item()


# --------------------------------------------------------
# PPO の更新処理
# --------------------------------------------------------
def ppo_update(model: nn.Module, optimizer: optim.Optimizer, data: dict, cfg: PPOConfig):
    """
    1 回の PPO 更新を実行する。

    * ratio         = π( a | s ) / π_old( a | s )
    * clip          = ratio を [1−ε, 1+ε] に制限し、過大な更新を抑制
    * pg_loss       = −E[min(ratio * A, clip(ratio) * A)]
    * value_loss    = critic の MSE
    * entropy       = ポリシーの探索性を維持するための正則化項
    * 合計 loss     = pg_loss + vf_coef * value_loss − ent_coef * entropy
    """
    obs = data["obs"]
    act = data["act"]
    old_logp = data["old_logp"]
    ret = data["ret"]
    adv = data["adv"]

    total_pg_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_batches = 0

    for _ in range(cfg.epochs):
        idx = torch.randperm(cfg.update_every, device=cfg.device)
        for start in range(0, cfg.update_every, cfg.batch_size):
            batch_idx = idx[start : start + cfg.batch_size]
            b_obs = obs[batch_idx]
            b_act = act[batch_idx]
            b_old_logp = old_logp[batch_idx]
            b_ret = ret[batch_idx]
            b_adv = adv[batch_idx]

            logits, value = model(b_obs)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(b_act)
            ratio = torch.exp(logp - b_old_logp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * b_adv
            pg_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value, b_ret)

            entropy = dist.entropy().mean()
            loss = pg_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_batches += 1

    avg_pg_loss = total_pg_loss / total_batches if total_batches > 0 else 0.0
    avg_value_loss = total_value_loss / total_batches if total_batches > 0 else 0.0
    avg_entropy = total_entropy / total_batches if total_batches > 0 else 0.0

    return {
        "pg_loss": avg_pg_loss,
        "value_loss": avg_value_loss,
        "entropy": avg_entropy,
    }


# --------------------------------------------------------
# メインの学習ループ
# --------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: PPOConfig):
    # ==============================================================
    # 学習フロー概要
    # --------------------------------------------------------------
    # 1. draw_phase() でツモ → 観測取得
    # 2. select_action(): 合法手をマスクして行動をサンプル
    # 3. discard_phase(): 打牌／ロンを実行、報酬を得る
    # 4. RolloutBuffer.store(): トランジションを保存
    # 5. バッファ満杯 or ゲーム終了で GAE + PPO 更新
    # 6. total_steps 回繰り返す
    # ==============================================================
    set_seed(cfg.seed)

    # --- Weights & Biases 初期化 ---
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_runname,
        config=asdict(cfg),
    )

    env = MahjongEnv(seed=cfg.seed)
    if torch.cuda.is_available():
        print("Using CUDA:", torch.cuda.get_device_name(0))

    model = MahjongActorCritic(num_actions=17).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    wandb.watch(model, log="all", log_freq=100)

    env.reset()  # ← 追加: 山札と手牌を初期化

    # 初回 draw
    obs_dict = env.draw_phase()
    obs_img = env.encode_observation(obs_dict)
    obs_shape = obs_img.shape

    buffer = RolloutBuffer(cfg.update_every, obs_shape, cfg.device)

    global_step = 0
    start_time = time.time()

    while global_step < cfg.total_steps:
        # --- ゲーム終了フラグが立っていたら環境をリセットして続行 ---
        if env.done:
            env.reset()
            obs_dict = env.draw_phase()
            obs_img = env.encode_observation(obs_dict)
            continue

        legal = env._legal_actions(env.current_player)
        action, logp, value = select_action(model, obs_img, legal, cfg.device)

        # ----- discard -----
        next_obs_dict, reward, done, info = env.discard_phase(action)
        next_obs_img = env.encode_observation(next_obs_dict)

        # バッファへ保存
        buffer.store(
            torch.from_numpy(obs_img).float().to(cfg.device),
            action,
            logp,
            reward,
            float(done),
            value,
        )

        global_step += 1
        obs_img = next_obs_img

        # エピソード終了 or 通常ターン終了後に必ず draw_phase() を呼ぶ
        if done:
            env.reset()
            obs_dict = env.draw_phase()
            obs_img = env.encode_observation(obs_dict)
        else:
            obs_dict = env.draw_phase()
            obs_img = env.encode_observation(obs_dict)

        # ----- バッファ満杯で更新 -----
        if buffer.is_full():
            with torch.no_grad():
                _, last_val = model(torch.from_numpy(obs_img).unsqueeze(0).to(cfg.device))
            buffer.finish_path(last_val.item(), cfg.gamma, cfg.lam)
            data = buffer.get()
            metrics = ppo_update(model, optimizer, data, cfg)
            buffer.ptr = 0
            mean_ep_ret = data["ret"].mean().item()
            mean_ep_len = (data["done"] if "done" in data else torch.tensor([])).float().mean().item() if "done" in data else 0.0
            wandb.log(
                {
                    **metrics,
                    "batch_avg_return": mean_ep_ret,
                    "buffer_mean_ep_return": mean_ep_ret,
                    "buffer_mean_ep_length": mean_ep_len,
                },
                step=global_step
            )

        # ----- ログ -----
        if global_step % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"[{global_step:>6}] steps, elapsed {elapsed/60:.1f} min")

    # 保存
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, "checkpoints/ppo_final.pt")
    print("Training finished. Saved to checkpoints/ppo_final.pt")


# --------------------------------------------------------
if __name__ == "__main__":
    cfg = PPOConfig()
    print("PPO Config:", cfg)
    train(cfg)
