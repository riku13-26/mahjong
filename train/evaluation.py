"""evaluation.py
学習中のモデルとルールベースAIの対戦評価機能
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Dict, Any, Tuple
import time

from env_jpn.mahjong_env import MahjongEnv
from env_jpn.rule_based_ai import RuleBasedAI


def select_action_for_model(model, obs_img: np.ndarray, legal_actions: list[int], device: str):
    """
    学習モデルの行動選択（train/ppo.pyから移植）
    
    Args:
        model: Actor-Criticモデル
        obs_img: 観測画像 (H, W, C)
        legal_actions: 合法手のリスト
        device: 計算デバイス
    
    Returns:
        選択された行動
    """
    # 観測をバッチ次元付きのテンソルに変換
    obs_t = torch.from_numpy(obs_img).to(device).unsqueeze(0)  # (1, H, W, C)
    
    # モデルから行動確率と価値を取得
    with torch.no_grad():
        logits, _ = model(obs_t)
        logits = logits.squeeze(0)  # バッチ次元を削除

        # 非合法手を-∞でマスクして除外
        mask = torch.full_like(logits, float('-inf'))
        mask[legal_actions] = 0.0  # 合法手のみ0（マスクなし）
        masked_logits = logits + mask
        
        # 確率分布を作成して行動をサンプリング
        dist = torch.distributions.Categorical(logits=masked_logits)
        act = dist.sample()
    
    return act.item()


def evaluate_vs_rule_based(model, device: str, num_games: int = 100, seed: int = None, debug: bool = False) -> Dict[str, float]:
    """
    学習モデルとルールベースAIを対戦させて勝率を評価
    
    Args:
        model: 学習済みモデル
        device: 計算デバイス 
        num_games: 対戦回数
        seed: ランダムシード
        
    Returns:
        評価結果の辞書
        - model_win_rate: モデルの勝率
        - model_wins_as_player0: モデルが先手で勝った回数
        - model_wins_as_player1: モデルが後手で勝った回数
        - total_games: 総ゲーム数
        - avg_game_length: 平均ゲーム長
    """
    model.eval()
    
    rule_ai = RuleBasedAI(seed=seed)
    
    model_wins_as_p0 = 0  # モデルがプレイヤー0（先手）で勝った回数
    model_wins_as_p1 = 0  # モデルがプレイヤー1（後手）で勝った回数
    total_turns = 0
    
    # 半分ずつプレイヤー位置を入れ替えて対戦
    for game_idx in range(num_games):
        env = MahjongEnv(seed=seed + game_idx if seed else None)
        env.reset()
        
        # モデルがプレイヤー0か1かを決める（奇数偶数で分ける）
        model_is_player0 = (game_idx % 2 == 0)
        
        # ゲーム実行
        game_result = _play_single_game(
            env, model, rule_ai, device, model_is_player0, debug
        )
        
        if debug:
            print(f"Game {game_idx}: Winner={game_result['winner']}, ModelIsP0={model_is_player0}, Turns={game_result['turns']}")
        
        # 結果集計
        if game_result["winner"] is not None:
            if model_is_player0 and game_result["winner"] == 0:
                model_wins_as_p0 += 1
                if debug: print(f"  Model won as player 0")
            elif not model_is_player0 and game_result["winner"] == 1:
                model_wins_as_p1 += 1
                if debug: print(f"  Model won as player 1")
            elif debug:
                print(f"  AI won (winner={game_result['winner']}, model_is_p0={model_is_player0})")
        elif debug:
            print(f"  Draw/Timeout")
                
        total_turns += game_result["turns"]
    
    # 統計計算
    total_model_wins = model_wins_as_p0 + model_wins_as_p1
    model_win_rate = total_model_wins / num_games if num_games > 0 else 0.0
    avg_game_length = total_turns / num_games if num_games > 0 else 0.0
    
    model.train()  # 評価後は訓練モードに戻す
    
    return {
        "model_win_rate": model_win_rate,
        "model_wins_as_player0": model_wins_as_p0,
        "model_wins_as_player1": model_wins_as_p1,
        "total_model_wins": total_model_wins,
        "total_games": num_games,
        "avg_game_length": avg_game_length,
    }


def _play_single_game(
    env: MahjongEnv, 
    model, 
    rule_ai: RuleBasedAI, 
    device: str,
    model_is_player0: bool,
    debug: bool = False
) -> Dict[str, Any]:
    """
    1ゲームを実行する
    
    Args:
        env: 麻雀環境
        model: 学習モデル
        rule_ai: ルールベースAI
        device: 計算デバイス
        model_is_player0: モデルがプレイヤー0かどうか
        
    Returns:
        ゲーム結果
    """
    turns = 0
    winner = None
    max_turns = 300  # 無限ループを防ぐための制限
    
    # 初回draw
    obs_dict = env.draw_phase()
    
    while not env.done and turns < max_turns:
        turns += 1
        
        # 現在のプレイヤーを決定
        current_player = env.current_player
        is_model_turn = (model_is_player0 and current_player == 0) or \
                       (not model_is_player0 and current_player == 1)
        
        # 行動選択
        legal_actions = env._legal_actions(current_player)
        
        if is_model_turn:
            # モデルの番
            obs_img = env.encode_observation(obs_dict)
            action = select_action_for_model(model, obs_img, legal_actions, device)
        else:
            # ルールベースAIの番
            action = rule_ai.select_action(obs_dict, legal_actions)
        
        # 行動実行
        next_obs_dict, reward, done, info = env.discard_phase(action)
        
        # 勝者判定
        if done:
            if debug:
                print(f"  Game ended after discard_phase: info={info}")
            if info.get('tsumo_win', False) or info.get('ron_win', False):
                winner = current_player
                if debug:
                    print(f"    Winner: Player {winner} ({'Model' if (model_is_player0 and winner == 0) or (not model_is_player0 and winner == 1) else 'AI'})")
            # 流局の場合はwinnerはNoneのまま
            break
            
        # 次のターン
        if not done:
            obs_dict = env.draw_phase()
            
            # draw_phaseで終了した場合の処理
            if obs_dict.get("done", False):
                if debug:
                    print(f"  Game ended after draw_phase: reward={obs_dict.get('reward', 0)}, current_player={env.current_player}")
                # draw_phaseで終了した場合、現在のプレイヤーがツモ和了した可能性がある
                # ツモ和了かどうかは報酬で判定（正の報酬なら勝利）
                if obs_dict.get("reward", 0) > 0:
                    winner = env.current_player
                    if debug:
                        print(f"    Winner: Player {winner} ({'Model' if (model_is_player0 and winner == 0) or (not model_is_player0 and winner == 1) else 'AI'}) via tsumo")
                # 流局の場合はwinnerはNoneのまま
                break
        else:
            break
    
    # ターン制限に達した場合
    if turns >= max_turns and debug:
        print(f"  Game timeout after {turns} turns")
    
    return {
        "winner": winner,
        "turns": turns,
    }


def quick_evaluate(model, device: str, num_games: int = 20) -> float:
    """
    高速評価（少ないゲーム数で勝率をチェック）
    
    Args:
        model: 学習モデル
        device: 計算デバイス
        num_games: 評価ゲーム数
        
    Returns:
        モデルの勝率
    """
    result = evaluate_vs_rule_based(model, device, num_games)
    return result["model_win_rate"]


def test_evaluation_debug(model, device: str, num_games: int = 3):
    """
    デバッグモードで評価をテストする
    """
    print(f"Testing evaluation with debug mode...")
    result = evaluate_vs_rule_based(model, device, num_games, debug=True)
    print(f"\nTest Results:")
    print(f"  Model win rate: {result['model_win_rate']:.3f}")
    print(f"  Model wins as P0: {result['model_wins_as_player0']}")
    print(f"  Model wins as P1: {result['model_wins_as_player1']}")
    print(f"  Total games: {result['total_games']}")
    print(f"  Avg game length: {result['avg_game_length']:.1f}")
    return result
