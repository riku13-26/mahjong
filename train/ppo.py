#!/usr/bin/env python3
"""
日本麻雀（簡易 1vs1）環境で PPO (Proximal Policy Optimization) を学習する最小実装。

ポイント
---------
* Environment : `env_jpn.mahjong_env.MahjongEnv`
* Model       : `models.cnn_res.MahjongActorCritic`
* Algorithm   : PPO（クリッピング・GAE 付き）

外部 RL ライブラリ（Stable-Baselines 等）は使わず、
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

wandb.login()

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
    total_steps: int = 20000         # 総学習ステップ数ß
    update_every: int = 1024         # バッファに貯めるステップ数
    epochs: int = 4                  # ポリシー更新のエポック数
    batch_size: int = 256            # ミニバッチサイズ
    gamma: float = 0.995              # 割引率
    lam: float = 0.95                # GAE のラムダ値
    clip_eps: float = 0.5            # PPO クリッピング係数
    lr: float = 2.5e-4                 # 学習率
    vf_coef: float = 0.5             # バリュー関数の損失係数
    ent_coef: float = 0.01           # エントロピー正則化係数
    max_grad_norm: float = 0.5       # 勾配クリッピングの最大ノルム
    seed: int = 42                   # ランダムシード
    log_interval: int = 1000         # ログ出力間隔
    wandb_project: str = "mahjong-rl" # Weights & Biases プロジェクト名
    wandb_runname: str = "run_ppo_cnn" # Weights & Biases 実行名


# --------------------------------------------------------
# ロールアウトを溜めるバッファ
# --------------------------------------------------------
class RolloutBuffer:
    """
    学習用サンプルを一時的に格納するクラス。
    
    主な機能:
    * store()        : ステップごとに観測・行動・報酬・価値などを保存
    * finish_path()  : エピソード終了時やバッファが満杯時に呼び出し、
                      GAE-Lambda アルゴリズムでAdvantage推定値とリターンを計算
    * get()          : PPO 更新のための辞書を返す（Advantageは標準化済み）
    """
    def __init__(self, size: int, obs_shape: tuple[int, ...], device: str):
        """
        バッファの初期化
        
        Args:
            size: バッファのサイズ（保存するステップ数）
            obs_shape: 観測の形状（高さ、幅、チャンネル）
            device: 計算デバイス（"cuda" または "cpu"）
        """
        self.size = size
        self.device = device
        # 各種データを保存するテンソルバッファ
        self.obs_buf = torch.zeros((size, *obs_shape), dtype=torch.float32, device=device)  # 観測
        self.act_buf = torch.zeros(size, dtype=torch.int64, device=device)                 # 行動
        self.logp_buf = torch.zeros(size, dtype=torch.float32, device=device)              # 行動の対数確率
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)               # 報酬
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)              # 終了フラグ
        self.val_buf = torch.zeros(size, dtype=torch.float32, device=device)               # 価値関数の推定値
        self.player_buf = torch.zeros(size, dtype=torch.int64, device=device)              # 手番のプレイヤー番号
        self.adv_buf = torch.zeros(size, dtype=torch.float32, device=device)               # Advantage推定値
        self.ret_buf = torch.zeros(size, dtype=torch.float32, device=device)               # リターン（価値の目標値）
        self.ptr = 0  # 現在の書き込み位置

    def store(self, obs, act, logp, rew, done, val, player):
        """
        1ステップ分のデータをバッファに保存
        
        Args:
            obs: 観測（状態）
            act: 選択した行動
            logp: その行動の対数確率
            rew: 得られた報酬
            done: エピソード終了フラグ
            val: 価値関数の推定値
            player: 手番のプレイヤー番号
        """
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.logp_buf[idx] = logp
        self.rew_buf[idx] = rew
        self.done_buf[idx] = done
        self.val_buf[idx] = val
        self.player_buf[idx] = player
        self.ptr += 1
        return idx
    
    def update(self, idx: int, *, rew=None, done=None, val=None):
        if rew  is not None: self.rew_buf[idx]  = rew
        if done is not None: self.done_buf[idx] = done
        if val  is not None: self.val_buf[idx]  = val

    def is_full(self):
        """バッファが満杯かどうかをチェック"""
        return self.ptr >= self.size

    # def finish_path(self, last_val: float, gamma: float, lam: float):
    def finish_path(
        self,
        last_val_per_player: dict[int, float],   # ← 引数を辞書に変更
        gamma: float,
        lam: float,
        ):
        """
        GAE (Generalized Advantage Estimation) を使ってAdvantage推定値とリターンを計算
        
        GAEの仕組み:
        1. TD誤差 δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        2. Advantage A_t = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}
           → λで指数減衰させた「未来のTD誤差」の総和
        3. リターン R_t = A_t + V(s_t)
           → 価値関数の学習目標として使用
        
        Args:
            last_val: 最終状態での価値推定値（ブートストラップ用）
            gamma: 割引率
            lam: GAEのλパラメータ
        """
        # データをCPUに移してnumpy配列に変換（計算効率のため）
        rew  = self.rew_buf[:self.ptr].cpu().numpy()
        val  = self.val_buf[:self.ptr].cpu().numpy()
        done = self.done_buf[:self.ptr].cpu().numpy()
        adv = np.zeros_like(rew)
        pid = self.player_buf[:self.ptr].cpu().numpy()
        
        # 末尾から逆順にAdvantageを計算
        last_gae   = {0: 0.0, 1: 0.0}
        last_value = {0: last_val_per_player.get(0, 0.0),
                      1: last_val_per_player.get(1, 0.0)}

        # for t in reversed(range(self.ptr)):
        #     # エピソード終了時はブートストラップを切る
        #     mask = 1.0 - done[t]
        #     # 次の状態の価値（最終ステップの場合は外部から与えられた値を使用）
        #     next_val = last_val if t == self.ptr - 1 else val[t + 1]
        #     # TD誤差を計算
        #     delta = rew[t] + gamma * next_val * mask - val[t]
        #     # GAEの再帰的計算
        #     last_gae = delta + gamma * lam * mask * last_gae
        #     adv[t] = last_gae

        for t in reversed(range(self.ptr)):      # ← プレイヤーごとに独立して再帰
            p = pid[t]
            mask = 1.0 - done[t]
            delta = rew[t] + gamma * last_value[p] * mask - val[t]
            last_gae[p] = delta + gamma * lam * mask * last_gae[p]
            adv[t] = last_gae[p]
            last_value[p] = val[t]
        
        # 計算結果をGPUに戻す
        self.adv_buf[:self.ptr] = torch.as_tensor(adv, device=self.device)
        self.ret_buf[:self.ptr] = self.adv_buf[:self.ptr] + self.val_buf[:self.ptr]

    def get(self):
        """
        PPO更新用のデータセットを取得
        
        Returns:
            dict: 学習に必要なデータの辞書
                - obs: 観測データ
                - act: 行動データ
                - old_logp: 古いポリシーでの行動の対数確率
                - ret: リターン（価値関数の学習目標）
                - adv: 標準化されたAdvantage推定値
        """
        assert self.is_full(), "バッファが満杯でないとデータを取得できません"
        
        # Advantageを標準化（学習の安定性向上のため）
        # adv = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-8)
        # Advantageをプレイヤー毎に標準化（学習の安定性向上のため）
        adv = self.adv_buf.clone()
        for p in (0, 1):
            mask = self.player_buf == p
            if mask.any():
                sub = adv[mask]
                adv[mask] = (sub - sub.mean()) / (sub.std() + 1e-8)
        
        dataset = {
            "obs": self.obs_buf,
            "act": self.act_buf,
            "old_logp": self.logp_buf,
            "ret": self.ret_buf,
            "adv": adv,
            "player": self.player_buf,
            "done": self.done_buf,
        }
        return dataset


# --------------------------------------------------------
# 行動選択関数（非合法手のマスキング付き）
# --------------------------------------------------------
def select_action(model: nn.Module, obs_img: np.ndarray, legal_actions: list[int], device: str):
    """
    合法手のみから行動を選択する関数
    
    Args:
        model: Actor-Criticモデル
        obs_img: 観測画像 (H, W, C)
        legal_actions: 合法手のリスト
        device: 計算デバイス
    
    Returns:
        tuple: (選択された行動, その行動の対数確率, 価値推定値)
    """
    # 観測をバッチ次元付きのテンソルに変換
    obs_t = torch.from_numpy(obs_img).to(device).unsqueeze(0)  # (1, H, W, C)
    
    # モデルから行動確率と価値を取得
    logits, value = model(obs_t)
    logits = logits.squeeze(0)  # バッチ次元を削除

    # 非合法手を-∞でマスクして除外
    mask = torch.full_like(logits, float('-inf'))
    mask[legal_actions] = 0.0  # 合法手のみ0（マスクなし）
    masked_logits = logits + mask
    
    # 確率分布を作成して行動をサンプリング
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
    total_clip_frac = 0.0
    total_approx_kl = 0.0
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
            
            # Clip fraction: what fraction of the batch was clipped
            clip_frac = torch.mean((torch.abs(ratio - 1) > cfg.clip_eps).float())
            
            # Approximate KL divergence
            approx_kl = 0.5 * torch.mean((logp - b_old_logp) ** 2)

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
            total_clip_frac += clip_frac.item()
            total_approx_kl += approx_kl.item()
            total_batches += 1

    avg_pg_loss = total_pg_loss / total_batches if total_batches > 0 else 0.0
    avg_value_loss = total_value_loss / total_batches if total_batches > 0 else 0.0
    avg_entropy = total_entropy / total_batches if total_batches > 0 else 0.0
    avg_clip_frac = total_clip_frac / total_batches if total_batches > 0 else 0.0
    avg_approx_kl = total_approx_kl / total_batches if total_batches > 0 else 0.0

    return {
        "pg_loss": avg_pg_loss,
        "value_loss": avg_value_loss,
        "entropy": avg_entropy,
        "clip_frac": avg_clip_frac,
        "approx_kl": avg_approx_kl,
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

    wandb.watch(model, log="gradients", log_freq=100)

    env.reset()  # ← 追加: 山札と手牌を初期化

    # 初回 draw
    obs_dict = env.draw_phase()
    obs_img = env.encode_observation(obs_dict)
    obs_shape = obs_img.shape

    buffer = RolloutBuffer(cfg.update_every, obs_shape, cfg.device)

    global_step = 0
    start_time = time.time()
    last_idx = {0: None, 1: None}
    best_mean_return = float('-inf')

    while global_step < cfg.total_steps:
        # --- ゲーム終了フラグが立っていたら環境をリセットして続行 ---
        if env.done:
            env.reset()
            obs_dict = env.draw_phase()
            obs_img = env.encode_observation(obs_dict)
            continue

        # obs_dict = env.draw_phase()

        legal = env._legal_actions(env.current_player)
        action, logp, value = select_action(model, obs_img, legal, cfg.device)

        # ----- discard -----
        next_obs_dict, reward, done, info = env.discard_phase(action)
        next_obs_img = env.encode_observation(next_obs_dict)

        # バッファへ保存
        idx = buffer.store(
            torch.from_numpy(obs_img).float().to(cfg.device),
            action,
            logp,
            reward,
            float(done),
            value,
            env.current_player,
        )
        last_idx[env.current_player] = idx

        # ロンの場合に一つ前の相手の行動をマイナスの報酬にする
        if info.get('ron_win', False):
            loser = 1 - env.current_player
            pen = last_idx[loser]
            if pen is not None:
                buffer.update(pen, rew=buffer.rew_buf[pen] - 1.5, done=1.0)

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

            # draw が返す obs_dict に done / reward が入っている
            # 流局 or ツモ和了
            if obs_dict.get("done", False):
                self_rew = obs_dict['reward']
                opp_rew = obs_dict['opponent_reward']

                # ① 現在プレイヤーの報酬 → 現在の行動(idx)へ
                if idx is not None:
                    buffer.update(idx, rew=buffer.rew_buf[idx] + self_rew, done=1.0)

                # ② 相手プレイヤーの報酬 → 一つ前の相手行動へ
                opp_pid  = 1 - env.current_player
                idx_opp  = last_idx.get(opp_pid)
                if idx_opp is not None:
                    buffer.update(idx_opp, rew=buffer.rew_buf[idx_opp] + opp_rew, done=1.0)

                # 環境をリセット
                env.reset()
                obs_dict = env.draw_phase()
                obs_img = env.encode_observation(obs_dict)


        # ----- バッファ満杯で更新 -----
        if buffer.is_full():
            # --- バッファ終端で両プレイヤー視点の V(s) を推定してブートストラップ ---
            last_val = {}
            with torch.no_grad():
                for p in (0, 1):
                    # プレイヤー p 視点の観測を取得
                    # MahjongEnv には `encode_observation` と対になるユーティリティが想定される。
                    # もし `get_obs_dict(p)` 等が無い場合は適宜置き換えてください。
                    obs_p_dict = env.get_obs_dict(p)           # ← ★環境側の関数名に合わせる
                    obs_p_img  = env.encode_observation(obs_p_dict)
                    _, v_p = model(torch.from_numpy(obs_p_img)
                                         .unsqueeze(0).to(cfg.device))
                    last_val[p] = v_p.item()

            buffer.finish_path(last_val, cfg.gamma, cfg.lam)
                
            data = buffer.get()
            metrics = ppo_update(model, optimizer, data, cfg)
            buffer.ptr = 0
            mean_ep_ret = data["ret"].mean().item()
            mean_ep_len = data["done"].sum().item() / cfg.update_every
            adv_std = data["adv"].std().item()
            wandb.log(
                {
                    **metrics,
                    "batch_avg_return": mean_ep_ret,
                    "buffer_mean_ep_return": mean_ep_ret,
                    "buffer_mean_ep_length": mean_ep_len,
                    "adv_std": adv_std,
                    "clip_frac": metrics["clip_frac"],
                    "approx_kl": metrics["approx_kl"],
                },
                step=global_step
            )
            
            # Save best model based on buffer_mean_ep_return
            if mean_ep_ret > best_mean_return:
                best_mean_return = mean_ep_ret
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "best_return": best_mean_return,
                    "global_step": global_step
                }, "checkpoints/ppo_best.pt")
                print(f"New best model saved! Return: {best_mean_return:.4f} at step {global_step}")

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
