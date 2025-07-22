import matplotlib.pyplot as plt
import numpy as np
import torch

from mahjong_env import MahjongEnv  # パスは環境に合わせて調整してください

# 1. 環境初期化
env = MahjongEnv(seed=42)
obs = env.reset()

# 2. 1 ターンだけ進めて観測を取得
obs = env.draw_phase()                 # ツモ
action = env._legal_actions(env.current_player)[0]  # 適当に先頭の合法手を選択
obs, _, _, _ = env.discard_phase(action)            # 打牌

print(obs)

# 3. 画像エンコード
tensor = env.encode_observation(obs)   # (H=4, W=16, C=4)
img_t = torch.from_numpy(tensor).permute(2, 0, 1)  # (C, H, W)
print(img_t.shape)
print(img_t)

# 4. 可視化
titles = [
    "Ch0 : my_hand",
    "Ch1 : my_discords",
    "Ch2 : oppo_hand (only zero)",
    "Ch3 : oppo_discords",
]

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    axes[i].imshow(tensor[:, :, i], cmap="gray", vmin=0, vmax=1)
    axes[i].set_title(titles[i], fontsize=9)
    axes[i].set_xticks(range(16))
    axes[i].set_yticks(range(4))
    axes[i].invert_yaxis()          # 最上段＝0 枚目を上に
plt.tight_layout()
plt.show()