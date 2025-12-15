# train_tetris_vs_heuristic.py
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tetris_env_vs_heuristic import TetrisVsHeuristicEnv


class PolicyNet(nn.Module):
    """
    state_dim = 214 (board 200 + current 7 + hold 7)
    n_actions = 41 (width=10 → 10*4 + 1(hold))
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


def train_vs_heuristic(
    num_episodes=5000,
    gamma=0.99,
    lr=1e-3,
    save_path="tetris_vs_heuristic_hold_policy.pth",
    log_interval=50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TetrisVsHeuristicEnv()
    state_dim = 214
    n_actions = env.n_actions

    policy = PolicyNet(state_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_returns = []
    episode_lengths = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        log_probs = []
        rewards = []

        done = False
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

        ep_return = float(sum(rewards))
        ep_len = len(rewards)
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)

        # Gt 계산 (discount)
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for log_prob, Gt in zip(log_probs, returns):
            loss += -log_prob * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % log_interval == 0 or episode == 1:
            print(
                f"[Episode {episode}/{num_episodes}] "
                f"Return: {ep_return:.2f}, Steps: {ep_len}, Loss: {loss.item():.4f}"
            )

        if episode % max(200, log_interval) == 0:
            torch.save(policy.state_dict(), save_path)
            print(f"  -> Saved model to {save_path}")

    # 로그 저장 (원하면 시각화에 사용)
    np.savez(
        "tetris_vs_heuristic_hold_logs.npz",
        returns=np.array(episode_returns, dtype=np.float32),
        lengths=np.array(episode_lengths, dtype=np.int32),
    )
    torch.save(policy.state_dict(), save_path)
    print(f"Training finished. Final model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="tetris_vs_heuristic_hold_policy.pth")
    args = parser.parse_args()

    train_vs_heuristic(
        num_episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
