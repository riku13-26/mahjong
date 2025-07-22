"""utils.py
================================================
簡易二人日本麻雀環境で共通利用するユーティリティ集。
各種定数・牌操作関数・和了判定ロジックなどを提供する。
※ コードには日本語で丁寧なコメントを付与している。
"""
from __future__ import annotations

import itertools
import random
from typing import List

# ------------------------------------------------------------
# 牌に関する定数定義
# ------------------------------------------------------------
# 0〜8  : 萬子 1〜9
# 9〜15 : 字牌（東=9, 南=10, 西=11, 北=12, 白=13, 發=14, 中=15）
TILE_TYPES: List[int] = list(range(16))
COPIES_PER_TILE: int = 4  # 各牌の枚数
TOTAL_TILES: int = len(TILE_TYPES) * COPIES_PER_TILE  # 64 枚

# 1 局で最大何枚まで捨てられるか（2 人×最大 18 巡を想定）
MAX_DISCARDS: int = 18

# 報酬テーブル（終局時に参照）
REWARD_TABLE = {
    "tsumo": {"self": 1.0, "opp": -0.5},
    "ron":   {"self": 1.5, "opp": -1.5},
    "ryu_self_tenpai": {"self": 0.2, "opp": -0.2},
    "ryu_opp_tenpai": {"self": -0.2, "opp": 0.2},
    "ryu_noten":  {"self": 0.0, "opp":  0.0},
}

# ------------------------------------------------------------
# 山（ウォール）生成
# ------------------------------------------------------------

def init_wall(rng: random.Random | None = None) -> List[int]:
    """64 枚の牌をシャッフルして山（リスト）を生成する。

    Parameters
    ----------
    rng : random.Random | None
        乱数生成器。`None` の場合は内部で生成する。

    Returns
    -------
    List[int]
        シャッフル済みの牌 ID リスト（先頭が山の頂点）。
    """
    if rng is None:
        rng = random.Random()

    wall: List[int] = []
    for tile_id in TILE_TYPES:
        wall.extend([tile_id] * COPIES_PER_TILE)
    rng.shuffle(wall)
    return wall

# ------------------------------------------------------------
# 牌 ⇔ ベクトル変換
# ------------------------------------------------------------

def tiles_to_vector(tiles: List[int]) -> List[int]:
    """牌 ID リスト → 16 次元枚数ベクトルに変換する。"""
    vec = [0] * len(TILE_TYPES)
    for tid in tiles:
        vec[tid] += 1
    return vec


def vector_to_tiles(vec: List[int]) -> List[int]:
    """16 次元枚数ベクトル → 牌 ID リストに変換する。"""
    tiles: List[int] = []
    for tid, count in enumerate(vec):
        tiles.extend([tid] * count)
    return tiles

# ------------------------------------------------------------
# 和了（ホーラ）判定
# ------------------------------------------------------------

def check_win_vector(vec: List[int]) -> bool:
    """非常に単純化した和了形判定（順子・刻子のみ）。

    - 七対子・国士・チートイなど複雑な役は無視。
    - 順子（連続 3 枚）と 刻子（同一 3 枚）を 4 セット揃え、
      残り 2 枚が同一牌の対子であれば和了とする。
    - 字牌は刻子と対子のみとし、順子は成立しない。

    参考実装のため厳密さよりシンプルさを優先。
    """
    tiles_left = vec.copy()

    # 対子を探してみる
    for i in range(len(TILE_TYPES)):
        if tiles_left[i] >= 2:
            # 対子を一旦抜いて残り 12 枚を確認
            tiles_left[i] -= 2
            if _can_form_melds(tiles_left):
                return True
            tiles_left[i] += 2  # 戻す
    return False


def _can_form_melds(vec: List[int]) -> bool:
    """残り 12 枚がすべて順子/刻子で構成できるかを再帰的に判定。"""
    # ベースケース：すべて 0 枚なら OK
    if sum(vec) == 0:
        return True

    # 最初に見つかった牌 ID
    first = next(i for i, c in enumerate(vec) if c > 0)

    # 1) 刻子を作るケース
    if vec[first] >= 3:
        vec[first] -= 3
        if _can_form_melds(vec):
            vec[first] += 3
            return True
        vec[first] += 3

    # 2) 順子を作るケース（萬子のみ）
    if first <= 6:  # 6 = 8‑2 で 7,8,9 は順子の先頭になれない
        if vec[first + 1] > 0 and vec[first + 2] > 0:
            vec[first] -= 1
            vec[first + 1] -= 1
            vec[first + 2] -= 1
            if _can_form_melds(vec):
                vec[first] += 1
                vec[first + 1] += 1
                vec[first + 2] += 1
                return True
            vec[first] += 1
            vec[first + 1] += 1
            vec[first + 2] += 1

    return False

# # ------------------------------------------------------------
# # 捨て牌パディング
# # ------------------------------------------------------------

# def pad_discards(discards: List[int], max_len: int = MAX_DISCARDS) -> List[int]:
#     """捨て牌リストを固定長にパディングしてベクトル化。"""
#     padded = discards + [-1] * (max_len - len(discards))  # -1 は「空」を表す
#     # -1 → 16 次元ワンホット 0 ベクトルとするために 17 番目の次元を用意してもよい
#     vec = []
#     for tid in padded:
#         one_hot = [0] * len(TILE_TYPES)
#         if tid != -1:
#             one_hot[tid] = 1
#         vec.extend(one_hot)
#     return vec
