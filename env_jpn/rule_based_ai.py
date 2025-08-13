"""rule_based_ai.py
簡易な麻雀ルールベースAIの実装

基本戦略:
1. ロンできる場合は積極的にロンする
2. テンパイを目指しつつ、安全な牌を優先的に捨てる  
3. 危険そうな牌は避ける
4. 字牌や端牌を優先的に捨てる
"""
from __future__ import annotations
import random
from typing import Dict, Any, List

from env_jpn.utils import (
    TILE_TYPES, 
    tiles_to_vector, 
    check_win_vector
)


class RuleBasedAI:
    """簡易麻雀ルールベースAI"""
    
    def __init__(self, seed: int = None):
        """
        Args:
            seed: ランダムシードの値
        """
        self.rng = random.Random(seed)
        
    def select_action(self, obs: Dict[str, Any], legal_actions: List[int]) -> int:
        """
        観測情報と合法手から行動を選択する
        
        Args:
            obs: 環境からの観測情報 
            legal_actions: 合法手のリスト
            
        Returns:
            選択した行動（牌IDまたは16=ロン）
        """
        # ロンが可能な場合は積極的にロンする
        if 16 in legal_actions:
            return 16
            
        # 手牌をベクトルに変換
        hand_vec = obs["hand_vec"]
        current_tiles = []
        for tile_id, count in enumerate(hand_vec):
            current_tiles.extend([tile_id] * count)
        
        # 各合法手について評価スコアを計算
        action_scores = []
        for action in legal_actions:
            if action == 16:
                continue  # ロンは上で処理済み
                
            score = self._evaluate_discard(action, current_tiles, obs)
            action_scores.append((action, score))
        
        # スコアが最も高い行動を選択（同じスコアの場合はランダム）
        if not action_scores:
            return self.rng.choice(legal_actions)
            
        best_score = max(score for _, score in action_scores)
        best_actions = [action for action, score in action_scores if score == best_score]
        return self.rng.choice(best_actions)
    
    def _evaluate_discard(self, discard_tile: int, hand_tiles: List[int], obs: Dict[str, Any]) -> float:
        """
        特定の牌を捨てることの評価スコアを計算
        
        Args:
            discard_tile: 捨てる牌のID
            hand_tiles: 現在の手牌
            obs: 観測情報
            
        Returns:
            評価スコア（高いほど良い）
        """
        score = 0.0
        
        # 1. テンパイ維持/向上を評価
        remaining_tiles = hand_tiles.copy()
        remaining_tiles.remove(discard_tile)
        
        # 捨てた後にテンパイになるかチェック
        tenpai_after = self._is_tenpai(remaining_tiles)
        if tenpai_after:
            score += 100.0  # テンパイ維持は高評価
            
        # 2. 牌の価値を評価
        tile_value = self._get_tile_value(discard_tile, hand_tiles)
        score -= tile_value  # 価値の高い牌を捨てるのはマイナス
        
        # 3. 安全度を評価（相手の捨て牌を参考に）
        safety_score = self._get_safety_score(discard_tile, obs)
        score += safety_score
        
        return score
    
    def _is_tenpai(self, tiles: List[int]) -> bool:
        """手牌がテンパイかどうかを判定"""
        hand_vec = tiles_to_vector(tiles)
        
        # 各牌を1枚加えて和了形になるかチェック  
        for tile_id in TILE_TYPES:
            if hand_vec[tile_id] >= 4:  # 既に4枚持っている場合はスキップ
                continue
            hand_vec[tile_id] += 1
            if check_win_vector(hand_vec):
                hand_vec[tile_id] -= 1
                return True
            hand_vec[tile_id] -= 1
        return False
    
    def _get_tile_value(self, tile_id: int, hand_tiles: List[int]) -> float:
        """牌の価値を評価（高いほど手牌に重要）"""
        hand_vec = tiles_to_vector(hand_tiles)
        
        # 字牌は基本的に価値が低い
        if tile_id >= 9:  # 字牌
            # ただし対子や刻子になっている場合は価値あり
            if hand_vec[tile_id] >= 2:
                return 10.0
            else:
                return 5.0  # 単独字牌は価値低い
        
        # 萬子（数牌）の価値
        value = 20.0
        
        # 既に持っている枚数が多いほど価値が高い（対子・刻子狙い）
        value += hand_vec[tile_id] * 5.0
        
        # 順子の可能性を評価
        if tile_id <= 6:  # 1-7萬は順子の先頭になり得る
            if hand_vec[tile_id + 1] > 0 or hand_vec[tile_id + 2] > 0:
                value += 10.0
        if 1 <= tile_id <= 7:  # 2-8萬は順子の中間になり得る
            if hand_vec[tile_id - 1] > 0 or hand_vec[tile_id + 1] > 0:
                value += 10.0
        if tile_id >= 2:  # 3-9萬は順子の末尾になり得る
            if hand_vec[tile_id - 1] > 0 or hand_vec[tile_id - 2] > 0:
                value += 10.0
                
        return value
    
    def _get_safety_score(self, tile_id: int, obs: Dict[str, Any]) -> float:
        """牌の安全度を評価（高いほど安全）"""
        score = 0.0
        
        # 相手の捨て牌をチェック
        opp_discards = obs.get("opp_discard", [])
        
        # 相手が既に捨てている牌は比較的安全
        if tile_id in opp_discards:
            score += 20.0
            
        # 字牌は比較的安全
        if tile_id >= 9:
            score += 10.0
            
        # 端牌（1萬、9萬）も比較的安全
        if tile_id in [0, 8]:
            score += 15.0
        
        return score