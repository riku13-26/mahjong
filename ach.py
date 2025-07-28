#!/usr/bin/env python3
"""
日本麻雀（簡易 1vs1）環境で Actor‑Critic Hedge (ACH) を学習するサンプル実装。

PPO 実装 (train_ppo.py) との主な相違点
--------------------------------------
1. **ポリシー出力**
   * モデルは行動ごとの **logit y(a|s)** を直接出力。
   * 実際の方策 π は Hedge 係数 η を掛けた soft‑max で求める。
2. **バッファに保存する追加情報**
   * 選択した行動 a_t に対応する旧 logit **y_old** を保存。
3. **ACH 更新 (ach_update)**
   * 二重クリップ：
       - **ratio** = π/π_old → PPO と同じ ε クリップ
       - **logit 差**: Δy = y_new − y_old → 区間 [−l_th, l_th] にクリップ
   * 損失
       L = 
       	− c · η · exp(Δy_clipped) · A
       	+ vf_coef · value_loss
       	− ent_coef · entropy
4. **ハイパーパラメータ** (ACHConfig)
   * hedge_eta      : 情報集合ごとの Hedge 係数 (単一エージェントなので定数扱い)
   * logit_clip_eps : ロジット差クリップ幅 l_th
   * c_regret       : 後悔重み c

このファイルは **依存コード(train_ppo.py と同じモデル & 環境)** を想定している。
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
# ACH のハイパーパラメータ
# Actor-Critic Hedge アルゴリズム用の設定値を管理
# --------------------------------------------------------
@dataclass
class ACHConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 学習デバイス（GPU優先）
    total_steps: int = 20000        # 総学習ステップ数
    update_every: int = 1024        # モデル更新頻度（ステップ数）
    epochs: int = 4                 # 1回の更新での学習エポック数
    batch_size: int = 256           # ミニバッチサイズ
    gamma: float = 0.995            # 割引率（将来報酬の重み）
    lam: float = 0.95               # GAE（Generalized Advantage Estimation）のλ
    clip_eps: float = 0.5           # ratio クリップ幅 (PPO 同様)
    logit_clip_eps: float = 0.25    # ロジット差クリップ幅 l_th（ACH特有）
    hedge_eta: float = 1.0          # Hedge 係数 η(s)（定数近似）
    c_regret: float = 1.0           # 後悔重み c（ACH特有）
    lr: float = 2.5e-4              # 学習率（Adam optimizer用）
    vf_coef: float = 0.5            # 価値関数損失の重み
    ent_coef: float = 0.01          # エントロピー正則化の重み
    max_grad_norm: float = 0.5      # 勾配クリッピングの最大ノルム
    seed: int = 42                  # 乱数シード値
    log_interval: int = 1000        # ログ出力間隔
    wandb_project: str = "mahjong-rl"   # Weights & Biases プロジェクト名
    wandb_runname: str = "run_ach_cnn"  # 実行名

# --------------------------------------------------------
# ロールアウトバッファ (ACH 版)
# 経験データを蓄積し、ACH学習に必要な旧ロジット値も保存
# --------------------------------------------------------
class ACHRolloutBuffer:
    """ ACH 用バッファ: 旧 logit も保存 """
    def __init__(self, size: int, obs_shape: tuple[int, ...], device: str):
        self.size = size        # バッファサイズ
        self.device = device    # 計算デバイス
        # 各種データバッファの初期化
        self.obs_buf   = torch.zeros((size, *obs_shape), dtype=torch.float32, device=device)  # 観測データ
        self.act_buf   = torch.zeros(size, dtype=torch.int64, device=device)    # 実行した行動
        self.logp_buf  = torch.zeros(size, dtype=torch.float32, device=device)  # π_old の log prob（旧方策の対数確率）
        self.logit_buf = torch.zeros(size, dtype=torch.float32, device=device)  # y_old (選択行動のロジット値)
        self.rew_buf   = torch.zeros(size, dtype=torch.float32, device=device)  # 報酬
        self.done_buf  = torch.zeros(size, dtype=torch.float32, device=device)  # エピソード終了フラグ
        self.val_buf   = torch.zeros(size, dtype=torch.float32, device=device)  # 状態価値
        self.adv_buf   = torch.zeros(size, dtype=torch.float32, device=device)  # アドバンテージ
        self.ret_buf   = torch.zeros(size, dtype=torch.float32, device=device)  # リターン（割引累積報酬）
        self.ptr = 0    # 現在の書き込み位置

    def store(self, obs, act, logp, logit, rew, done, val):
        """経験データをバッファに保存"""
        idx = self.ptr
        self.obs_buf[idx]   = obs      # 観測
        self.act_buf[idx]   = act      # 行動
        self.logp_buf[idx]  = logp     # 旧方策の対数確率
        self.logit_buf[idx] = logit    # 選択行動のロジット値
        self.rew_buf[idx]   = rew      # 報酬
        self.done_buf[idx]  = done     # 終了フラグ
        self.val_buf[idx]   = val      # 状態価値
        self.ptr += 1                  # ポインタを進める
        return idx                     # 保存位置のインデックスを返す

    def update(self, idx: int, *, rew=None, done=None, val=None):
        """指定されたインデックスのデータを更新（後から修正する場合に使用）"""
        if rew  is not None: self.rew_buf[idx]  = rew    # 報酬の更新
        if done is not None: self.done_buf[idx] = done   # 終了フラグの更新  
        if val  is not None: self.val_buf[idx]  = val    # 状態価値の更新

    def is_full(self):
        """バッファが満杯かどうかを判定"""
        return self.ptr >= self.size

    def finish_path(self, last_val: float, gamma: float, lam: float):
        """GAE（Generalized Advantage Estimation）を用いてアドバンテージとリターンを計算"""
        # データをnumpy配列に変換
        rew  = self.rew_buf[:self.ptr].cpu().numpy()   # 報酬
        val  = self.val_buf[:self.ptr].cpu().numpy()   # 状態価値
        done = self.done_buf[:self.ptr].cpu().numpy()  # 終了フラグ
        
        # GAEによるアドバンテージ計算
        adv = np.zeros_like(rew)
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            mask = 1.0 - done[t]  # エピソード継続マスク
            next_val = last_val if t == self.ptr - 1 else val[t + 1]  # 次状態の価値
            delta = rew[t] + gamma * next_val * mask - val[t]         # TD誤差
            last_gae = delta + gamma * lam * mask * last_gae          # GAE計算
            adv[t] = last_gae
        
        # アドバンテージとリターンをバッファに保存
        self.adv_buf[:self.ptr] = torch.as_tensor(adv, device=self.device)
        self.ret_buf[:self.ptr] = self.adv_buf[:self.ptr] + self.val_buf[:self.ptr]

    def get(self):
        """学習用データを取得（アドバンテージは正規化済み）"""
        assert self.is_full(), "バッファが満杯でないとデータを取得できません"
        # アドバンテージの正規化（平均0、標準偏差1）
        adv = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-8)
        return {
            "obs": self.obs_buf,        # 観測データ
            "act": self.act_buf,        # 実行した行動
            "old_logp": self.logp_buf,  # 旧方策の対数確率
            "old_logit": self.logit_buf, # 旧ロジット値（ACH用）
            "ret": self.ret_buf,        # リターン（目標値）
            "adv": adv,                 # 正規化済みアドバンテージ
        }

# --------------------------------------------------------
# 行動選択 (logit も返す)
# 方策ネットワークから行動を選択し、ACH学習に必要な情報を取得
# --------------------------------------------------------
@torch.no_grad()
def select_action(model: nn.Module, obs_img: np.ndarray, legal_actions: list[int], device: str):
    """
    観測から行動を選択し、学習に必要な情報を返す
    
    Returns:
        act: 選択された行動ID
        logp: 旧方策での対数確率
        logit_act: 選択行動のロジット値（ACH用）
        value: 状態価値の推定値
    """
    obs_t = torch.from_numpy(obs_img).to(device).unsqueeze(0)
    logits, value = model(obs_t)  # ネットワークからロジットと価値を取得
    logits = logits.squeeze(0)
    
    # 非合法手のマスキング（-∞を設定）
    mask = torch.full_like(logits, float('-inf'))
    mask[legal_actions] = 0.0  # 合法手のみ0（マスクなし）
    masked_logits = logits + mask
    
    # カテゴリカル分布から行動をサンプリング
    dist = torch.distributions.Categorical(logits=masked_logits)
    act = dist.sample()           # 行動をサンプリング
    logp = dist.log_prob(act)     # その行動の対数確率
    logit_act = logits[act]       # 選択行動の生ロジット値
    
    return act.item(), logp.item(), logit_act.item(), value.item()

# --------------------------------------------------------
# ACH 更新
# Actor-Critic Hedge アルゴリズムによるネットワーク更新
# --------------------------------------------------------

def ach_update(model: nn.Module, optimizer: optim.Optimizer, data: dict, cfg: ACHConfig):
    """
    ACHアルゴリズムによるモデル更新
    - PPO風のratio clipping
    - ロジット差分のclipping（ACH特有）
    - 後悔重み付き損失関数
    """
    # バッファからデータを展開
    obs        = data["obs"]        # 観測データ
    act        = data["act"]        # 実行した行動
    old_logp   = data["old_logp"]   # 旧方策の対数確率
    old_logit  = data["old_logit"]  # 旧ロジット値（ACH用）
    ret        = data["ret"]        # リターン（目標値）
    adv        = data["adv"]        # アドバンテージ

    total_loss, total_batches = 0.0, 0
    # 学習メトリクスの初期化
    metrics = {
        "policy_loss": 0.0,   # 方策損失
        "value_loss": 0.0,    # 価値関数損失
        "entropy": 0.0,       # エントロピー
        "clip_frac": 0.0,     # クリップされた割合
        "approx_kl": 0.0,     # 近似KLダイバージェンス
    }

    # 複数エポックの学習ループ
    for _ in range(cfg.epochs):
        # データをランダムシャッフル
        idx = torch.randperm(cfg.update_every, device=cfg.device)
        
        # ミニバッチ単位で処理
        for start in range(0, cfg.update_every, cfg.batch_size):
            batch_idx = idx[start: start + cfg.batch_size]
            # バッチデータの抽出
            b_obs       = obs[batch_idx]       # バッチ観測
            b_act       = act[batch_idx]       # バッチ行動
            b_old_logp  = old_logp[batch_idx]  # バッチ旧対数確率
            b_old_logit = old_logit[batch_idx] # バッチ旧ロジット
            b_ret       = ret[batch_idx]       # バッチリターン
            b_adv       = adv[batch_idx]       # バッチアドバンテージ

            # 現在のネットワークから新しいロジットと価値を取得
            logits, value = model(b_obs)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(b_act)  # 新方策での対数確率
            new_logit = logits.gather(1, b_act.unsqueeze(1)).squeeze(1)  # 選択行動の新ロジット

            # --- ratio & KL divergence ---
            ratio = torch.exp(logp - b_old_logp)  # π_new / π_old
            approx_kl = 0.5 * torch.mean((logp - b_old_logp) ** 2)  # 近似KL

            # --- ロジット差分とクリッピング（ACH特有）---
            logit_diff = new_logit - b_old_logit  # Δy = y_new - y_old
            logit_clipped = torch.clamp(logit_diff, -cfg.logit_clip_eps, cfg.logit_clip_eps)  # クリップ
            w = torch.exp(logit_clipped)  # exp(Δy_clipped) - ACHの重み

            # --- ACH policy loss ---
            # L_policy = -c * η * exp(Δy_clipped) * A
            policy_loss = -(cfg.c_regret * cfg.hedge_eta * w * b_adv).mean()

            # --- 価値関数損失とエントロピー ---
            value_loss = F.mse_loss(value, b_ret)  # MSE損失（価値関数）
            entropy = dist.entropy().mean()        # 方策のエントロピー

            # --- 総損失の計算 ---
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            # --- パラメータ更新 ---
            optimizer.zero_grad()                                           # 勾配をゼロクリア
            loss.backward()                                                 # 逆伝播
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm) # 勾配クリッピング
            optimizer.step()                                                # パラメータ更新

            # --- メトリクスの累積 ---
            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"]     += entropy.item()
            metrics["clip_frac"]   += torch.mean((torch.abs(ratio - 1) > cfg.clip_eps).float()).item()  # クリップ率
            metrics["approx_kl"]   += approx_kl.item()
            total_batches += 1

    # メトリクスの平均化
    for k in metrics:
        metrics[k] /= total_batches
    return metrics

# --------------------------------------------------------
# 学習ループ
# メイン学習処理とユーティリティ関数
# --------------------------------------------------------

def set_seed(seed: int):
    """再現性確保のため各種乱数シードを設定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: ACHConfig):
    """ACHアルゴリズムによるメイン学習ループ"""
    set_seed(cfg.seed)  # 乱数シード固定

    # Weights & Biases初期化（実験管理）
    wandb.init(project=cfg.wandb_project, name=cfg.wandb_runname, config=asdict(cfg))

    # 環境とモデルの初期化
    env = MahjongEnv(seed=cfg.seed)  # 麻雀環境
    if torch.cuda.is_available():
        print("Using CUDA:", torch.cuda.get_device_name(0))

    model = MahjongActorCritic(num_actions=17).to(cfg.device)  # Actor-Criticネットワーク
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)      # オプティマイザ
    wandb.watch(model, log="all", log_freq=100)                # モデル監視

    # 環境の初期化と観測形状の取得
    env.reset()
    obs_dict = env.draw_phase()         # ドロー段階の観測を取得
    obs_img = env.encode_observation(obs_dict)  # 観測をエンコード
    obs_shape = obs_img.shape           # 観測の次元

    # 経験バッファの初期化
    buffer = ACHRolloutBuffer(cfg.update_every, obs_shape, cfg.device)

    # 学習状態の管理変数
    global_step = 0                     # 総ステップ数
    start_time = time.time()            # 学習開始時刻
    last_idx = {0: None, 1: None}       # 各プレイヤーの最後のバッファインデックス
    best_mean_return = float('-inf')    # 最高平均リターン（モデル保存用）

    # メイン学習ループ
    while global_step < cfg.total_steps:
        # エピソード終了時の処理
        if env.done:
            env.reset()
            obs_dict = env.draw_phase()
            obs_img = env.encode_observation(obs_dict)
            continue

        # 行動選択（現在プレイヤーの合法手から）
        legal = env._legal_actions(env.current_player)
        action, logp, logit_act, value = select_action(model, obs_img, legal, cfg.device)

        # 環境で行動実行
        next_obs_dict, reward, done, info = env.discard_phase(action)
        next_obs_img = env.encode_observation(next_obs_dict)

        # 経験をバッファに保存
        idx = buffer.store(
            torch.from_numpy(obs_img).float().to(cfg.device),  # 観測
            action,                                            # 行動
            logp,                                              # 対数確率
            logit_act,                                         # ロジット値
            reward,                                            # 報酬
            float(done),                                       # 終了フラグ
            value,                                             # 状態価値
        )
        last_idx[env.current_player] = idx  # プレイヤーの最後のインデックス記録

        # ロン勝利時の敗者ペナルティ
        if info.get('ron_win', False):
            loser = 1 - env.current_player  # 相手プレイヤー
            pen = last_idx[loser]           # 敗者の最後のインデックス
            if pen is not None:
                buffer.update(pen, rew=buffer.rew_buf[pen] - 1.5)  # ペナルティ追加

        global_step += 1     # ステップ数増加
        obs_img = next_obs_img  # 観測を更新

        # エピソード処理
        if done:
            # 終了時の処理
            env.reset()
            obs_dict = env.draw_phase()
            obs_img = env.encode_observation(obs_dict)
        else:
            # 継続時の処理
            obs_dict = env.draw_phase()
            obs_img = env.encode_observation(obs_dict)
            
            # ゲーム終了判定（ドロー段階で判明する場合）
            if obs_dict.get("done", False):
                self_rew = obs_dict['reward']          # 自分の報酬
                opp_rew  = obs_dict['opponent_reward'] # 相手の報酬
                
                # 自分の報酬を追加し、エピソード終了をマーク
                if idx is not None:
                    buffer.update(idx, rew=buffer.rew_buf[idx] + self_rew, done=1.0)
                
                # 相手の報酬も更新
                opp_pid = 1 - env.current_player
                idx_opp = last_idx.get(opp_pid)
                if idx_opp is not None:
                    buffer.update(idx_opp, rew=buffer.rew_buf[idx_opp] + opp_rew)
                
                # 環境をリセット
                env.reset()
                obs_dict = env.draw_phase()
                obs_img = env.encode_observation(obs_dict)

        # バッファが満杯になったら学習実行
        if buffer.is_full():
            # 最後の状態価値を取得（GAE計算用）
            with torch.no_grad():
                _, last_val = model(torch.from_numpy(obs_img).unsqueeze(0).to(cfg.device))
            
            # GAEによるアドバンテージ計算
            buffer.finish_path(last_val.item(), cfg.gamma, cfg.lam)
            
            # 学習データ取得とACH更新
            data = buffer.get()
            metrics = ach_update(model, optimizer, data, cfg)
            buffer.ptr = 0  # バッファリセット
            
            # 統計情報の計算
            mean_ep_ret = data["ret"].mean().item()           # 平均リターン
            mean_ep_len = data["done"].sum().item() / cfg.update_every  # 平均エピソード長
            adv_std = data["adv"].std().item()                # アドバンテージの標準偏差
            
            # Weights & Biasesにログ出力
            wandb.log({
                **metrics,
                "batch_avg_return": mean_ep_ret,
                "buffer_mean_ep_return": mean_ep_ret,
                "buffer_mean_ep_length": mean_ep_len,
                "adv_std": adv_std,
            }, step=global_step)
            
            # 最高性能モデルの保存
            if mean_ep_ret > best_mean_return:
                best_mean_return = mean_ep_ret
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "best_return": best_mean_return,
                    "global_step": global_step,
                }, "checkpoints/ach_best.pt")
                print(f"New best model saved! Return: {best_mean_return:.4f} at step {global_step}")

        # 定期的な進捗ログ出力
        if global_step % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"[{global_step:>6}] steps, elapsed {elapsed/60:.1f} min")

    # 学習終了時の最終モデル保存
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, "checkpoints/ach_final.pt")
    print("Training finished. Saved to checkpoints/ach_final.pt")

# --------------------------------------------------------
# メイン実行部
# --------------------------------------------------------
if __name__ == "__main__":
    cfg = ACHConfig()           # 設定を初期化
    print("ACH Config:", cfg)   # 設定内容を表示
    train(cfg)                  # 学習開始
