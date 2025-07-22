"""mahjong_env.py
============================================================
二人用・簡易日本麻雀環境（Draw Phase → Discard Phase）
------------------------------------------------------------
この改訂版では 1 ターンを **2 つのフェーズ** に分割し、
エージェントがツモ牌を見てから打牌を選択できるようにした。

* ``draw_phase()``  : 自動で 1 枚ツモ → 観測返却（和了/流局なら終了）
* ``discard_phase(action)`` : 打牌 or ロン → 次状態返却

Gym の ``step`` を分割した形だが、呼び出し側の学習ループは
```python
obs = env.draw_phase()
act = agent.select_action(obs)
next_obs, r, done, info = env.discard_phase(act)
```
の 2 行で済む。
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import numpy as np

from env_jpn.utils import (
    TILE_TYPES,
    COPIES_PER_TILE,
    REWARD_TABLE,
    init_wall,
    tiles_to_vector,
    check_win_vector,
)

TILE_NAMES: List[str] = [
    "1萬", "2萬", "3萬", "4萬", "5萬", "6萬", "7萬", "8萬", "9萬",
    "東", "南", "西", "北", "白", "發", "中",
]

class MahjongEnv:
    """Draw → Discard の 2 フェーズ制簡易 2 人麻雀環境。"""

    # ----------------------------
    # コンストラクタ
    # ----------------------------
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        # 内部状態
        self.wall: List[int] = []
        self.hands: List[List[int]] = []
        self.discards: List[List[int]] = []
        self.current_player: int = 0
        self.turn_count: int = 0
        self.done: bool = False
        self.waiting_discard: bool = False  # True = draw 済みで打牌待ち

    # ------------------------------------------------------------------
    # リセット
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.wall = init_wall(self.rng)
        self.hands = [[], []]
        self.discards = [[], []]
        for _ in range(13):
            for pid in range(2):
                self.hands[pid].append(self.wall.pop())
        self.current_player = 0
        self.turn_count = 0
        self.done = False
        self.waiting_discard = False
        # ツモ前観測
        return self._build_observation(tsumo_tile=None)

    # ------------------------------------------------------------------
    # フェーズ 1: ツモ
    # ------------------------------------------------------------------
    def draw_phase(self) -> Dict[str, Any]:
        """現在手番プレイヤーが 1 枚ツモし、観測を返す。"""
        if self.done:
            raise RuntimeError("ゲーム終了済みです reset() を呼んでください。")
        if self.waiting_discard:
            raise RuntimeError("既に draw 済みです discard_phase() を呼んでください。")

        pid = self.current_player
        tsumo_tile: int | None = None
        reward_self = reward_opp = 0.0
        info: Dict[str, Any] = {}

        # 山が残っていればツモ
        if self.wall:
            tsumo_tile = self.wall.pop()
            self.hands[pid].append(tsumo_tile)
            # ツモ和了チェック
            if check_win_vector(tiles_to_vector(self.hands[pid])):
                self.done = True
                reward_self = REWARD_TABLE["tsumo"]["self"]
                reward_opp = REWARD_TABLE["tsumo"]["opp"]
                info["tsumo_win"] = True
        else:
            # 流局処理
            self.done = True
            tenpai_self = self._is_tenpai(pid)
            tenpai_opp = self._is_tenpai(1 - pid)
            if tenpai_self and tenpai_opp:
                pass  # 報酬 0 のまま
            elif tenpai_self:
                reward_self = REWARD_TABLE["ryu_self_tenpai"]["self"]
                reward_opp = REWARD_TABLE["ryu_self_tenpai"]["opp"]
            elif tenpai_opp:
                reward_self = REWARD_TABLE["ryu_opp_tenpai"]["self"]
                reward_opp = REWARD_TABLE["ryu_opp_tenpai"]["opp"]
            info["ryukyoku"] = True

        # 終了した場合は opponent_reward を info に詰めて返す
        if self.done:
            obs_term = self._build_observation(tsumo_tile)
            obs_term["done"] = True
            obs_term["reward"] = reward_self
            obs_term["opponent_reward"] = reward_opp
            info["opponent_reward"] = reward_opp
            self.waiting_discard = False
            return obs_term

        # 続行：打牌待ち状態へ
        self.waiting_discard = True
        return self._build_observation(tsumo_tile)

    # ------------------------------------------------------------------
    # フェーズ 2: 打牌 / ロン
    # ------------------------------------------------------------------
    def discard_phase(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """draw_phase 後に呼び出し、打牌/ロンを処理する。"""
        if self.done:
            raise RuntimeError("ゲーム終了済みです reset() を呼んでください。")
        if not self.waiting_discard:
            raise RuntimeError("まず draw_phase() を呼んでください。")

        pid = self.current_player
        reward_self = reward_opp = 0.0
        info: Dict[str, Any] = {}

        legal = self._legal_actions(pid)
        if action not in legal:
            self.done = True
            obs = self._build_observation(tsumo_tile=None)
            return obs, -1.0, True, {"illegal_action": True}

        if action == 16:
            # ロン宣言
            self.done = True
            reward_self = REWARD_TABLE["ron"]["self"]
            reward_opp = REWARD_TABLE["ron"]["opp"]
            info["ron_win"] = True
        else:
            # 通常打牌
            self.hands[pid].remove(action)
            self.discards[pid].append(action)
            # 相手ロンチェック
            opp = 1 - pid
            if check_win_vector(tiles_to_vector(self.hands[opp] + [action])):
                self.done = True
                reward_self = REWARD_TABLE["ron"]["opp"]  # 放銃側
                reward_opp = REWARD_TABLE["ron"]["self"]
                info["opponent_ron"] = True

        # ターン交代
        if not self.done:
            self.current_player = 1 - self.current_player
            self.turn_count += 1
            self.waiting_discard = False  # 次ターン draw 待ち
        else:
            info["opponent_reward"] = reward_opp
            self.waiting_discard = False

        obs = self._build_observation(tsumo_tile=None)
        return obs, reward_self, self.done, info

    # ------------------------------------------------------------------
    # 内部ヘルパ
    # ------------------------------------------------------------------
    def _build_observation(self, tsumo_tile: int | None) -> Dict[str, Any]:
        pid = self.current_player
        opp = 1 - pid
        hand_vec = tiles_to_vector(self.hands[pid])
        opp_vec = tiles_to_vector(self.hands[opp])
        tsumo_vec = np.zeros(len(TILE_TYPES), dtype=np.float32)
        if tsumo_tile is not None:
            tsumo_vec[tsumo_tile] = 1.0
        return {
            "hand_vec": hand_vec,
            "opp_hand_vec": opp_vec,
            "my_discard": self.discards[pid],
            "opp_discard": self.discards[opp],
            "tsumo": tsumo_vec,
            "tenpai": int(self._is_tenpai(pid)),
            "remain": len(self.wall),
            "turn": self.turn_count,
        }

    def _legal_actions(self, player: int) -> List[int]:
        legal = sorted(set(self.hands[player]))
        opp = 1 - player
        if self.discards[opp]:
            last = self.discards[opp][-1]
            if check_win_vector(tiles_to_vector(self.hands[player] + [last])):
                legal.append(16)
        return legal

    def _is_tenpai(self, player: int) -> bool:
        vec = tiles_to_vector(self.hands[player])
        for t in TILE_TYPES:
            if vec[t] >= COPIES_PER_TILE:
                continue
            vec[t] += 1
            if check_win_vector(vec):
                vec[t] -= 1
                return True
            vec[t] -= 1
        return False

    # ------------------------------------------------------------------
    # 画像エンコードユーティリティ
    # ------------------------------------------------------------------
    @staticmethod
    def _counts_to_grid(counts: np.ndarray) -> np.ndarray:
        """
        16 種類の牌を「それぞれ何枚持っているか」を示すベクトル
        （長さ 16, 各要素 0〜4）を、画像のような 0/1 グリッドに変換する。

        - 横方向（列 0〜15）: 牌の種類
        - 縦方向（行 0〜3） : その牌を何枚持っているか

        例）counts[0] == 2 なら
            一萬を 2 枚持っているので grid[0, 0] と grid[1, 0] を 1 にする
        """
        grid = np.zeros((4, len(TILE_TYPES)), dtype=np.float32)
        for t in range(len(TILE_TYPES)):
            c = int(counts[t])
            if c > 0:
                grid[:c, t] = 1.0
        return grid

    def encode_observation(
        self,
        obs: Dict[str, Any],
        open_hand: bool = False,
    ) -> np.ndarray:
        """
        観測 `obs` を CNN へ渡せる 3‑次元テンソル
        **(高さ 4, 幅 16, チャンネル 4)** に変換する。

        チャンネルの意味
        0: 手番プレイヤーの手牌
        1: 手番プレイヤーの捨て牌
        2: 相手の手牌（`open_hand=True` の時だけ表示）
        3: 相手の捨て牌

        戻り値は 0/1 の `float32` 配列。
        """
        # ----- チャンネル 0: 自分の手牌 -----
        hand_grid = self._counts_to_grid(obs["hand_vec"])

        # ----- チャンネル 1: 自分の捨て牌 -----
        my_disc_vec = np.zeros(len(TILE_TYPES), dtype=np.int32)
        for t in obs["my_discard"]:
            my_disc_vec[t] += 1
        my_disc_grid = self._counts_to_grid(my_disc_vec)

        # ----- チャンネル 2: 相手の手牌（デフォルトは非表示） -----
        if open_hand:
            opp_hand_vec = obs["opp_hand_vec"]
        else:
            opp_hand_vec = np.zeros(len(TILE_TYPES), dtype=np.int32)
        opp_hand_grid = self._counts_to_grid(opp_hand_vec)

        # ----- チャンネル 3: 相手の捨て牌 -----
        opp_disc_vec = np.zeros(len(TILE_TYPES), dtype=np.int32)
        for t in obs["opp_discard"]:
            opp_disc_vec[t] += 1
        opp_disc_grid = self._counts_to_grid(opp_disc_vec)

        # Stack channels along the last axis: (H, W, C)
        tensor = np.stack(
            [hand_grid, my_disc_grid, opp_hand_grid, opp_disc_grid],
            axis=-1,
        ).astype(np.float32)

        return tensor

    # ------------------------------------------------------------------
    # デバッグ表示（人間可読）
    # ------------------------------------------------------------------
    def render(self) -> None:
        """現在の内部状態をコンソールに表示する。デバッグ用。"""
        def to_str(lst: List[int]) -> str:
            """牌 ID リストを昇順ソートし、可読な牌名（TILE_NAMES）で返す。"""
            return " ".join(TILE_NAMES[t] for t in sorted(lst)) if lst else "(なし)"

        print("========== 麻雀環境 ==========")
        print(f"残り山: {len(self.wall)} 枚 | ターン: {self.turn_count}")
        for pid in range(2):
            turn_mark = "← 手番" if pid == self.current_player else "        "
            hand_str = to_str(self.hands[pid])
            discard_str = to_str(self.discards[pid])
            tenpai_mark = "[*テンパイ*]" if self._is_tenpai(pid) else ""
            print(f"P{pid}{turn_mark} 手牌({len(self.hands[pid])}) {tenpai_mark}: {hand_str}")
            print(f"      捨て牌: {discard_str}")
        print("==============================")