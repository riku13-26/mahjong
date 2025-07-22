"""
multi_match_stats.py
====================
ランダム Bot（ロン優先）同士を複数局プレイし、
得られた報酬パターンを集計するツール。

実行例
------
$ python multi_match_stats.py           # デフォルト 100 局
$ python multi_match_stats.py 1000      # 1000 局
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from typing import Dict, List, Tuple

from mahjong_env import MahjongEnv

# 牌 ID → 可読表記（デバッグ用：表示でしか使わない）
TILE_NAMES: List[str] = [
    "1萬","2萬","3萬","4萬","5萬","6萬","7萬","8萬","9萬",
    "東","南","西","北","白","發","中",
]

# ------------------------------------------------------------
# 1 局シミュレーション（ランダム＋ロン優先）
# ------------------------------------------------------------
def play_one_episode(seed: int | None = None) -> Tuple[float, float]:
    env = MahjongEnv(seed)
    env.reset()

    reward_tot = [0.0, 0.0]

    while not env.done:
        # Draw phase
        obs = env.draw_phase()
        if env.done:
            # ツモ和了 or 流局
            pid = env.current_player
            reward_tot[pid]     += obs.get("reward", 0.0)
            reward_tot[1 - pid] += obs.get("opponent_reward", 0.0)
            break

        # Discard phase
        pid = env.current_player
        legal = env._legal_actions(pid)
        action = 16 if 16 in legal else random.choice(legal)

        _, r_self, done, info = env.discard_phase(action)
        reward_tot[pid]     += r_self
        reward_tot[1 - pid] += info.get("opponent_reward", 0.0)

    return reward_tot[0], reward_tot[1]

# ------------------------------------------------------------
# メイン：複数エピソードを回して集計
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ランダム Bot 同士の対局を複数回実行し、報酬パターンを集計"
    )
    parser.add_argument("episodes", nargs="?", type=int, default=100,
                        help="対局回数 (既定: 100)")
    args = parser.parse_args()

    counter: Counter[Tuple[float, float]] = Counter()

    for ep in range(1, args.episodes + 1):
        rewards = play_one_episode(seed=random.randint(0, 2**32 - 1))
        counter[rewards] += 1
        if ep % max(1, args.episodes // 10) == 0:
            print(f"[{ep}/{args.episodes}] 進行中…")

    # 結果表示
    print("\n========== 報酬パターン集計 ==========")
    for pattern, cnt in counter.most_common():
        r0, r1 = pattern
        print(f"P0={r0:+.1f}, P1={r1:+.1f} : {cnt} 回")
    print("======================================")

if __name__ == "__main__":
    main()
