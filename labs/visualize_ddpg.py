"""
Lab10 DDPG 训练结果可视化脚本

用法：
    python labs/visualize_ddpg.py

功能：
    1. 加载训练好的 Actor 模型 ddpg_hopper_actor_class.pth
    2. 在 Hopper-v4 环境上运行训练好的策略
    3. 输出平均 Reward 和 Length
    4. 保存可视化结果为图片（默认 labs/ddpg_hopper_visualization.png）

可选：
    如果安装了 imageio，可以同时保存 GIF 动画。
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gymnasium as gym


class Actor(nn.Module):
    """DDPG Actor 网络，与训练时使用的结构保持一致。"""

    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)


def evaluate(actor, env, device, episodes=10, render=False):
    """运行多个测试回合，返回平均 reward 和 length。"""
    total_reward = 0.0
    total_length = 0

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = actor(state_tensor).cpu().numpy().flatten()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1

            if render and ep == 0:
                # 仅收集第一个 episode 的帧用于可视化
                env.unwrapped.render()

        total_reward += ep_reward
        total_length += ep_length
        print(f"Test Episode {ep + 1}/{episodes} | Reward: {ep_reward:.1f} | Length: {ep_length}")

    return total_reward / episodes, total_length / episodes


def collect_frames(actor, env, device, max_steps=1000, seed=0):
    """用训练好的策略收集一回合的渲染帧。"""
    state, _ = env.reset(seed=seed)
    frames = []

    for step in range(max_steps):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor(state_tensor).cpu().numpy().flatten()

        state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render().copy())

        if terminated or truncated:
            break

    return frames


def save_visualization(frames, save_path, num_show=8):
    """将关键帧保存为一张图片。"""
    fig, axes = plt.subplots(1, num_show, figsize=(16, 3))
    step = max(1, len(frames) // num_show)

    for i, ax in enumerate(axes):
        idx = min(i * step, len(frames) - 1)
        ax.imshow(frames[idx])
        ax.axis("off")
        ax.set_title(f"Step {idx}")

    plt.suptitle("Trained DDPG on Hopper-v4")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"可视化图片已保存: {save_path}")


def save_gif(frames, save_path, duration=0.04):
    """尝试保存 GIF 动画，需要 imageio。"""
    try:
        import imageio
        imageio.mimsave(save_path, frames, duration=duration)
        print(f"GIF 动画已保存: {save_path}")
    except ImportError:
        print("未安装 imageio，跳过 GIF 保存。可运行: pip install imageio")


def main():
    parser = argparse.ArgumentParser(description="DDPG Hopper-v4 可视化脚本")
    parser.add_argument(
        "--model",
        type=str,
        default="ddpg_hopper_actor_class.pth",
        help="训练好的 Actor 模型路径",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="测试回合数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="可视化回合的随机种子",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="labs/ddpg_hopper_visualization.png",
        help="可视化图片保存路径",
    )
    parser.add_argument(
        "--gif",
        type=str,
        default="labs/ddpg_hopper_visualization.gif",
        help="GIF 动画保存路径",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="不保存 GIF",
    )
    args = parser.parse_args()

    # 自动切换工作目录到项目根目录（脚本在 labs/ 下运行时也适用）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建环境（用于评估，不需要渲染）
    env_eval = gym.make("Hopper-v4")
    obs_dim = env_eval.observation_space.shape[0]
    act_dim = env_eval.action_space.shape[0]
    max_action = float(env_eval.action_space.high[0])

    # 加载模型
    actor = Actor(obs_dim, act_dim, max_action).to(device)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"找不到模型文件: {args.model}，请先完成训练。")
    actor.load_state_dict(torch.load(args.model, map_location=device))
    actor.eval()
    print(f"已加载模型: {args.model}")

    # 评估
    print(f"\n开始评估 {args.episodes} 个回合...")
    avg_reward, avg_length = evaluate(actor, env_eval, device, episodes=args.episodes)
    print(f"\n平均 Reward: {avg_reward:.1f}")
    print(f"平均 Length: {avg_length:.1f}")
    env_eval.close()

    # 收集可视化帧
    print("\n开始收集可视化帧...")
    env_vis = gym.make("Hopper-v4", render_mode="rgb_array")
    frames = collect_frames(actor, env_vis, device, seed=args.seed)
    env_vis.close()
    print(f"共收集 {len(frames)} 帧")

    # 保存可视化结果
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_visualization(frames, args.output)

    if not args.no_gif:
        save_gif(frames, args.gif)

    print("\n可视化完成！")


if __name__ == "__main__":
    main()
