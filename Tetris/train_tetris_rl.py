# train_tetris_rl.py
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tetris_env_rl import TetrisEnv


class PolicyNet(nn.Module):
    def __init__(self, state_dim=207, n_actions=40, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        # x: (batch, state_dim)
        return self.net(x)  # logits


def train(num_episodes=500, gamma=0.99, lr=1e-3, save_path="tetris_policy.pth", log_interval=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TetrisEnv()
    state_dim = 207
    n_actions = env.n_actions

    policy = PolicyNet(state_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            s_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)  # (1, state_dim)
            logits = policy(s_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        # 에피소드 끝 → 리턴 계산
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # 안정화를 위해 normalize
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 정책 경사 loss
        loss = 0.0
        for log_prob, Gt in zip(log_probs, returns):
            loss += -log_prob * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 로깅
        ep_return = sum(rewards)
        if episode % log_interval == 0 or episode == 1:
            print(f"[Episode {episode}/{num_episodes}] "
                  f"Return: {ep_return:.2f}, Steps: {len(rewards)}, Loss: {loss.item():.4f}")

        # 중간중간 저장
        if episode % max(50, log_interval) == 0:
            torch.save(policy.state_dict(), save_path)
            print(f"  -> Saved model to {save_path}")

    # 마지막 한 번 더 저장
    torch.save(policy.state_dict(), save_path)
    print(f"Training finished. Final model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500, help="학습 에피소드 수")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="tetris_policy.pth")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
