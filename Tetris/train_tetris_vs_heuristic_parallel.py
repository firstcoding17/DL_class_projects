# train_tetris_vs_heuristic_parallel.py

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tetris_env_vs_heuristic import TetrisVsHeuristicEnv


class PolicyNet(nn.Module):
    """
    state_dim = 214 (board 200 + current 7 + hold 7)
    n_actions = 41 (10*4 + 1(hold))
    """
    def __init__(self, state_dim=214, n_actions=41, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)  # logits


def train_parallel(
    total_episodes=10000,
    num_envs=8,
    gamma=0.99,
    lr=1e-3,
    save_path="tetris_vs_heuristic_hold_policy_parallel.pth",
    log_interval=10,
):
    """
    total_episodes: 전체 학습할 에피소드 수
    num_envs: 한 번에 함께 돌릴 환경 개수 (배치 크기 느낌)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 멀티 환경 생성
    envs = [TetrisVsHeuristicEnv(seed=i) for i in range(num_envs)]
    state_dim = 214
    n_actions = envs[0].n_actions

    policy = PolicyNet(state_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_returns_log = []
    episode_lengths_log = []

    # total_episodes를 num_envs씩 잘라서 학습
    episode_idx = 0
    batch_idx = 0

    while episode_idx < total_episodes:
        batch_idx += 1

        # 이번 배치에서 실제로 돌릴 에피소드 수 (마지막 배치는 num_envs보다 작을 수 있음)
        batch_size = min(num_envs, total_episodes - episode_idx)

        # 배치 버퍼
        all_log_probs = []
        all_returns = []
        batch_ep_returns = []
        batch_ep_lengths = []

        # 각 env로 에피소드 1개씩 돌리기
        for env_id in range(batch_size):
            env = envs[env_id]
            state = env.reset()

            log_probs = []
            rewards = []

            done = False
            steps = 0

            while not done:
                s_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
                logits = policy(s_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_state, reward, done, info = env.step(action.item())

                log_probs.append(log_prob)
                rewards.append(reward)

                state = next_state
                steps += 1

            ep_return = float(sum(rewards))
            ep_len = steps

            # Return 계산 (REINFORCE)
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            # 버퍼에 누적
            all_log_probs.extend(log_probs)
            all_returns.extend(returns)

            batch_ep_returns.append(ep_return)
            batch_ep_lengths.append(ep_len)

            # 전체 로그에도 저장
            episode_returns_log.append(ep_return)
            episode_lengths_log.append(ep_len)

            episode_idx += 1
            if episode_idx >= total_episodes:
                break

        # 텐서화 + normalize
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32).to(device)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        loss = 0.0
        for log_prob, Gt in zip(all_log_probs, returns_tensor):
            loss += -log_prob * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 로그 출력
        if batch_idx % log_interval == 0 or episode_idx == batch_size:
            mean_return = float(np.mean(batch_ep_returns))
            mean_len = float(np.mean(batch_ep_lengths))
            print(
                f"[Batch {batch_idx}] Episodes: {episode_idx}/{total_episodes}, "
                f"MeanReturn: {mean_return:.2f}, MeanLen: {mean_len:.1f}, Loss: {loss.item():.4f}"
            )

        # 중간 모델 저장
        if batch_idx % max(20, log_interval) == 0:
            torch.save(policy.state_dict(), save_path)
            print(f"  -> Saved model to {save_path}")

    # 로그 저장 (원하면 시각화)
    np.savez(
        "tetris_vs_heuristic_hold_parallel_logs.npz",
        returns=np.array(episode_returns_log, dtype=np.float32),
        lengths=np.array(episode_lengths_log, dtype=np.int32),
    )
    torch.save(policy.state_dict(), save_path)
    print(f"Training finished. Final model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000, help="전체 에피소드 수")
    parser.add_argument("--num_envs", type=int, default=8, help="병렬 환경 수")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="tetris_vs_heuristic_hold_policy_parallel.pth")
    args = parser.parse_args()

    train_parallel(
        total_episodes=args.episodes,
        num_envs=args.num_envs,
        gamma=args.gamma,
        lr=args.lr,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
