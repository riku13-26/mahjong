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
from train.evaluation import evaluate_vs_rule_based

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
    clip_eps: float = 0.2                 # PPO比率クリップ（0.1–0.2 推奨）
    logit_threshold: float = 6.0          # Logit Thresholding の l_th
    use_eta_in_policy: bool = True        # π=softmax(η·y_tilde) を使う

# --------------------------------------------------------
# ロールアウトバッファ (ACH 版)
# 経験データを蓄積し、ACH学習に必要な旧ロジット値も保存
# --------------------------------------------------------
class ACHRolloutBuffer:
    """ ACH 用バッファ: 旧 logit も保存 """
    def __init__(self, size: int, obs_shape: tuple[int, ...], device, num_actions: str):
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
        self.legal_mask_buf = torch.zeros((size, num_actions), dtype=torch.bool, device=device)

    def store(self, obs, act, logp, logit, rew, done, val, legal_mask):
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
        self.legal_mask_buf[idx] = torch.as_tensor(legal_mask, device=self.device)
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
            "legal_mask": self.legal_mask_buf, # 合法手マスク
        }

# --------------------------------------------------------
# 行動選択 (logit も返す)
# 方策ネットワークから行動を選択し、ACH学習に必要な情報を取得
# --------------------------------------------------------
@torch.no_grad()
def select_action(model: nn.Module, obs_img: np.ndarray, legal_actions: list[int], device: str, cfg: ACHConfig):
    """
    - logits を中心化→±l_th へクリップ（Logit Thresholding）
    - （必要なら）η を掛け、合法手以外を -inf マスク
    - Categorical からサンプル
    - y_old（η 前の前処理後ロジットの選択成分）と合法手マスクを返す
    """
    obs_t = torch.from_numpy(obs_img).to(device).unsqueeze(0)
    logits, value = model(obs_t)  # [1, A], [1, 1]

    # η 前の Logit Thresholding（中心化→±l_th）
    y_pre = threshold_logits(logits, cfg.logit_threshold).squeeze(0)  # [A]

    # 分布用ロジット（必要なら η スケーリング）
    y_for_dist = cfg.hedge_eta * y_pre if cfg.use_eta_in_policy else y_pre

    # 合法手マスク
    mask = torch.full_like(y_for_dist, float('-inf'))
    mask[legal_actions] = 0.0
    dist = torch.distributions.Categorical(logits=y_for_dist + mask)

    act = dist.sample()
    logp = dist.log_prob(act)

    # y_old は η 前の前処理後ロジットの選択成分（現在は未使用だが互換のため保持）
    y_old_act = y_pre[act]

    # 合法手ブールを保存（学習時に同じ正規化を再現するため）
    legal_mask = torch.zeros_like(y_pre, dtype=torch.bool)
    legal_mask[legal_actions] = True

    return act.item(), logp.item(), y_old_act.item(), value.item(), legal_mask.cpu().numpy()

# --------------------------------------------------------
# ACH 更新
# Actor-Critic Hedge アルゴリズムによるネットワーク更新
# --------------------------------------------------------
def ach_update(model: nn.Module, optimizer: optim.Optimizer, data: dict, cfg: ACHConfig):
    """
    二重クリップ：
      (1) PPO 比率クリップ
      (2) Logit Thresholding（平均差し引き→±l_th）
    """
    # --- 展開 ---
    obs        = data["obs"]          # [N, ...]
    act        = data["act"]          # [N]
    old_logp   = data["old_logp"]     # [N]
    # old_logit は今回は未使用（互換のため data には残しておいてOK）
    ret        = data["ret"]          # [N]
    adv        = data["adv"]          # [N]
    legal_mask = data["legal_mask"]   # [N, A] bool

    N = obs.size(0)

    # --- メトリクス ---
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "ratio_clip_frac": 0.0,       # ratio がクリップ域を超えた割合
        "approx_kl_quad": 0.0,        # 0.5 * E[(logp - old_logp)^2]
        "approx_kl_old_new": 0.0,     # E[old_logp - logp]
        "logit_thresh_hit": 0.0,      # |y_pre| が l_th に当たった比率（合法手のみ）
        "illegal_prob_sum": 0.0,      # 非合法手に割り当てられた確率合計（理想は 0）
    }
    total_batches = 0

    for _ in range(cfg.epochs):
        idx = torch.randperm(N, device=cfg.device)
        for start in range(0, N, cfg.batch_size):
            b = idx[start:start+cfg.batch_size]
            b_obs, b_act = obs[b], act[b]
            b_old_logp   = old_logp[b]
            b_ret, b_adv = ret[b], adv[b]
            b_legal_mask = legal_mask[b]  # [B, A] bool

            # --- Logit Thresholding（η 前） ---
            logits, value = model(b_obs)         # logits:[B,A], value:[B] or [B,1]
            y_pre = threshold_logits(logits, cfg.logit_threshold)

            # 分布用ロジット：η を掛けてから合法手以外を -inf
            y_for_dist = cfg.hedge_eta * y_pre if cfg.use_eta_in_policy else y_pre
            minus_inf  = torch.full_like(y_for_dist, float('-inf'))
            y_masked   = torch.where(b_legal_mask, y_for_dist, minus_inf)

            # --- 方策分布と確率比 ---
            dist = torch.distributions.Categorical(logits=y_masked)
            logp = dist.log_prob(b_act)          # 新 logπ
            ratio = torch.exp(logp - b_old_logp) # π_new / π_old

            # --- PPO 比率クリップ（Clip #1）---
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * b_adv
            ppo_obj = torch.min(surr1, surr2)

            # --- 損失（Δy 重みなし） ---
            policy_loss = -(cfg.c_regret * ppo_obj).mean()
            value       = value.squeeze(-1) if value.ndim == 2 else value
            value_loss  = F.mse_loss(value, b_ret)
            entropy     = dist.entropy().mean()

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # --- メトリクス更新 ---
            with torch.no_grad():
                clip_frac = ((ratio < 1.0 - cfg.clip_eps) | (ratio > 1.0 + cfg.clip_eps)).float().mean().item()
                kl_quad   = (0.5 * (logp - b_old_logp).pow(2).mean()).item()
                kl_on     = (b_old_logp - logp).mean().item()

                legal_per_row = b_legal_mask.sum(dim=1).clamp_min(1)
                # しきい値命中は合法手だけでカウント
                thresh_hits = ((y_pre.abs() >= (cfg.logit_threshold - 1e-6)) & b_legal_mask).sum(dim=1)
                logit_thresh_hit = (thresh_hits.float() / legal_per_row.float()).mean().item()

                probs = dist.probs
                illegal_prob_sum = (probs * (~b_legal_mask).float()).sum(dim=1).mean().item()

            metrics["policy_loss"]       += policy_loss.item()
            metrics["value_loss"]        += value_loss.item()
            metrics["entropy"]           += entropy.item()
            metrics["ratio_clip_frac"]   += clip_frac
            metrics["approx_kl_quad"]    += kl_quad
            metrics["approx_kl_old_new"] += kl_on
            metrics["logit_thresh_hit"]  += logit_thresh_hit
            metrics["illegal_prob_sum"]  += illegal_prob_sum

            total_batches += 1

    for k in metrics:
        metrics[k] /= max(total_batches, 1)
    return metrics


def threshold_logits(y: torch.Tensor, lth: float) -> torch.Tensor:
    """各サンプルでロジット平均を引いてから ±lth にクリップ"""
    centered = y - y.mean(dim=-1, keepdim=True)           # 中心化: y - ȳ
    return torch.clamp(centered, -lth, lth)               # しきい値クリップ

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
    buffer = ACHRolloutBuffer(cfg.update_every, obs_shape, cfg.device, num_actions=17)


    # 学習状態の管理変数
    global_step = 0                     # 総ステップ数
    start_time = time.time()            # 学習開始時刻
    last_idx = {0: None, 1: None}       # 各プレイヤーの最後のバッファインデックス
    best_mean_return = float('-inf')    # 最高平均リターン（モデル保存用）
    best_win_rate = 0.0
    last_eval_step = 0

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
        action, logp, y_old_act, value, legal_mask = select_action(
            model, obs_img, legal, cfg.device, cfg
        )

        # 環境で行動実行
        next_obs_dict, reward, done, info = env.discard_phase(action)
        next_obs_img = env.encode_observation(next_obs_dict)

        # 経験をバッファに保存
        idx = buffer.store(
            torch.from_numpy(obs_img).float().to(cfg.device),
            action,
            logp,
            y_old_act,     # ← η 前の前処理後ロジット（命名は logit_act でも可）
            reward,
            float(done),
            value,
            legal_mask,    # ← 追加
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
            adv_std = data["adv"].std().item()                # アドバンテージの標準偏差
            
            # Weights & Biasesにログ出力
            wandb.log({
                **metrics,
                "mean_ep_return": mean_ep_ret,
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

            # --- ルールベースAIとの対戦評価 ---
            if cfg.enable_evaluation and (global_step - last_eval_step) >= cfg.eval_interval:
                print(f"Evaluating model at step {global_step}...")
                eval_start = time.time()
                
                try:
                    eval_result = evaluate_vs_rule_based(
                        model, cfg.device, cfg.eval_games, seed=cfg.seed + global_step
                    )
                    
                    eval_time = time.time() - eval_start
                    win_rate = eval_result["model_win_rate"]
                    
                    # WandBにログ
                    wandb.log({
                        "eval/win_rate_vs_rule_based": win_rate,
                        "eval/model_wins_as_player0": eval_result["model_wins_as_player0"],
                        "eval/model_wins_as_player1": eval_result["model_wins_as_player1"],
                        "eval/total_model_wins": eval_result["total_model_wins"],
                        "eval/avg_game_length": eval_result["avg_game_length"],
                        "eval/evaluation_time": eval_time,
                    }, step=global_step)
                    
                    print(f"Evaluation completed: Win rate {win_rate:.3f} ({eval_result['total_model_wins']}/{cfg.eval_games} wins) in {eval_time:.1f}s")
                    
                    # 最高勝率のモデルを保存
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        os.makedirs("checkpoints", exist_ok=True)
                        torch.save({
                            "model_state": model.state_dict(),
                            "config": asdict(cfg),
                            "best_win_rate": best_win_rate,
                            "global_step": global_step,
                            "eval_result": eval_result
                        }, "checkpoints/ppo_best_winrate.pt")
                        print(f"New best win rate model saved! Win rate: {best_win_rate:.3f} at step {global_step}")
                    
                    last_eval_step = global_step
                    
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    # 評価に失敗してもトレーニングは続行

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
