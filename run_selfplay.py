#!/usr/bin/env python3
"""
保存済みモデル (checkpoints/ppo_final.pt) で自己対局を 1 局だけ可視化。
"""

import time
import torch
import numpy as np

from env_jpn.mahjong_env import MahjongEnv
from models.cnn_res import MahjongActorCritic
from typing import List, Tuple

CKPT_PATH = "checkpoints/ppo_final.pt"
SLEEP_SEC = 0.3        # 画面更新間隔
TILE_NAMES: List[str] = [
    "1萬", "2萬", "3萬", "4萬", "5萬", "6萬", "7萬", "8萬", "9萬",
    "東", "南", "西", "北", "白", "發", "中",
]

# --------------------------- ヘルパ ---------------------------
def greedy_action(model, obs_img, legal, device="cpu"):
    """ソフトマックス無しで合法手の最大ロジットを選択（推論モード）"""
    obs_t = torch.from_numpy(obs_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(obs_t)
    logits = logits.squeeze(0).cpu().numpy()
    logits[~np.isin(np.arange(len(logits)), legal)] = -np.inf
    return int(np.argmax(logits))

# --------------------------- main ---------------------------
def main():
    # --- モデル読込み ---
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = MahjongActorCritic(num_actions=17)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    env = MahjongEnv(seed=0)
    env.reset()
    obs = env.draw_phase()
    obs_img = env.encode_observation(obs)

    print("=== Self-Play Start ===")
    env.render()
    time.sleep(SLEEP_SEC)

    while True:
        legal = env._legal_actions(env.current_player)
        act = greedy_action(model, obs_img, legal)

        obs, reward, done, info = env.discard_phase(act)
        env.render()
        # Action を牌名に変換（0-15 が牌、16 がロン）
        act_str = TILE_NAMES[act] if act < len(TILE_NAMES) else "ロン"
        print(f"Player {1 - env.current_player} → Action {act_str} | Reward {reward}")
        time.sleep(SLEEP_SEC)

        if done:
            print("=== Game End ===", info)
            break

        # 次ターンの draw
        obs = env.draw_phase()
        obs_img = env.encode_observation(obs)

if __name__ == "__main__":
    main()