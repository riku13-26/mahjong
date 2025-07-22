"""play_human_vs_env.py
============================================================
人間 (P0) vs. ランダム Bot (P1) デモ  
------------------------------------------------------------
MahjongEnv（draw_phase → discard_phase 版）で 1 局だけプレイする CLI。

* あなたは **先手 P0**、Bot は P1。
* 各ターンは «ツモ → 打牌» の 2 フェーズ。
* 双方の手牌・テンパイ状態を表示しながら進行。
* ゲーム終了時に **最終手牌／捨て牌／報酬** をまとめて表示。

実行
----
```bash
$ python play_human_vs_env.py
```
"""
from __future__ import annotations

import random
from typing import List, Tuple

from mahjong_env import MahjongEnv
from utils import TILE_TYPES, COPIES_PER_TILE, tiles_to_vector, check_win_vector

# ------------------------------------------------------------
# 牌 ID ↔ 文字列
# ------------------------------------------------------------
TILE_NAMES: List[str] = [
    "1萬", "2萬", "3萬", "4萬", "5萬", "6萬", "7萬", "8萬", "9萬",
    "東", "南", "西", "北", "白", "發", "中",
]

def tiles_to_str(ids: List[int]) -> str:
    return " ".join(TILE_NAMES[i] for i in sorted(ids))

# ------------------------------------------------------------
# テンパイ判定
# ------------------------------------------------------------

def is_tenpai(hand: List[int]) -> bool:
    vec = tiles_to_vector(hand)
    for t in TILE_TYPES:
        if vec[t] >= COPIES_PER_TILE:
            continue
        vec[t] += 1
        if check_win_vector(vec):
            vec[t] -= 1
            return True
        vec[t] -= 1
    return False

# ------------------------------------------------------------
# 共通表示
# ------------------------------------------------------------

def show_state(env: MahjongEnv) -> None:
    for pid in (0, 1):
        label = "あなた" if pid == 0 else "Bot "
        mark = "[*テンパイ*]" if is_tenpai(env.hands[pid]) else ""
        print(f"{label} 手牌 : {tiles_to_str(env.hands[pid])} {mark}")
        print(f"{label} 捨て牌: {tiles_to_str(env.discards[pid])}")
    print(f"残り山: {len(env.wall)} | ターン: {env.turn_count}\n")

# ------------------------------------------------------------
# 打牌フェーズ
# ------------------------------------------------------------

def human_discard(env: MahjongEnv) -> Tuple[float, float]:
    pid = env.current_player
    legal = env._legal_actions(pid)
    print("合法手:", ", ".join(f"{a}:{'ロン' if a==16 else TILE_NAMES[a]}" for a in legal))
    while True:
        try:
            act = int(input("打つ牌 ID (16=ロン): "))
            if act in legal:
                break
            print("[!] 非合法手です。再入力してください。")
        except ValueError:
            print("[!] 数字を入力してください。")
    _, r_self, _, info = env.discard_phase(act)
    r_opp = info.get("opponent_reward", 0.0)
    print("あなたの打牌 →", "ロン" if act == 16 else TILE_NAMES[act])
    if r_self or env.done:
        print(f"報酬={r_self} | 終局={env.done}")
    return r_self, r_opp


def bot_discard(env: MahjongEnv) -> Tuple[float, float]:
    pid = env.current_player
    legal = env._legal_actions(pid)
    act = random.choice(legal)
    _, r_bot, _, info = env.discard_phase(act)
    r_you = info.get("opponent_reward", 0.0)
    print("Bot 打牌 →", "ロン" if act == 16 else TILE_NAMES[act])
    if r_bot or env.done:
        print(f"Bot報酬={r_bot} | 終局={env.done}")
    return r_you, r_bot  # 第一戻り値＝あなたの視点

# ------------------------------------------------------------
# 終局結果
# ------------------------------------------------------------

def show_final(env: MahjongEnv, reward_p0: float, reward_p1: float) -> None:
    print("\n================ 最終結果 ================")
    print("--- プレイヤー0 (あなた) ---")
    print("手牌 :", tiles_to_str(env.hands[0]))
    print("捨て牌:", tiles_to_str(env.discards[0]))
    print(f"報酬 : {reward_p0:+.1f}")
    print("\n--- プレイヤー1 (Bot) ---")
    print("手牌 :", tiles_to_str(env.hands[1]))
    print("捨て牌:", tiles_to_str(env.discards[1]))
    print(f"報酬 : {reward_p1:+.1f}")
    print("========================================")

# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------

def main(seed: int | None = 42):
    env = MahjongEnv(seed=seed)
    env.reset()

    reward_p0 = reward_p1 = 0.0  # 最終報酬格納

    while not env.done:
        # --- Draw Phase ---
        obs = env.draw_phase()
        if env.done:
            # draw_phase 内で終了（ツモ和了/流局）
            reward_p0 = obs.get("reward", 0.0) if env.current_player == 0 else obs.get("reward", 0.0)*-1
            reward_p1 = -reward_p0
            break

        show_state(env)
        tensor = env.encode_observation(obs)
        print(tensor[0])

        # --- Discard Phase ---
        if env.current_player == 0:
            r0, r1 = human_discard(env)
        else:
            r0, r1 = bot_discard(env)
        reward_p0 += r0
        reward_p1 += r1

        show_state(env)
        tensor = env.encode_observation(obs)
        print(tensor)

    # 終局表示
    show_final(env, reward_p0, reward_p1)


if __name__ == "__main__":
    main()
