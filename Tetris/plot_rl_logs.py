# plot_rl_logs.py
import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.load("tetris_vs_heuristic_logs.npz")
    returns = data["returns"]
    losses = data["losses"]
    lengths = data["lengths"]

    # 1) Return 그래프
    plt.figure()
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode Returns (vs Heuristic)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("returns_vs_heuristic.png")

    # 2) Loss 그래프
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Policy Loss (vs Heuristic)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_vs_heuristic.png")

    # 3) Episode length 그래프 (옵션)
    plt.figure()
    plt.plot(lengths)
    plt.xlabel("Episode")
    plt.ylabel("Steps per Episode")
    plt.title("Episode Length (vs Heuristic)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("length_vs_heuristic.png")

    print("Saved plots: returns_vs_heuristic.png, loss_vs_heuristic.png, length_vs_heuristic.png")


if __name__ == "__main__":
    main()
