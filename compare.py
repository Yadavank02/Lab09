import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lab09_starter as S


def load_all_results():
    with open("results/dqn_standard.json") as f:
        dqn = json.load(f)
    with open("results/dqn_ablations.json") as f:
        abl = json.load(f)
    with open("results/policy_gradient.json") as f:
        pg = json.load(f)

    return {
        "DQN (Full)": dqn,
        "DQN (No Target)": abl["dqn_no_target"],
        "DQN (No Buffer)": abl["dqn_no_buffer"],
        "DQN (Buffer 1k)": abl["dqn_buffer_1000"],
        "REINFORCE": pg["reinforce"],
        "PPO": pg["ppo"],
    }


def ranges_do_not_overlap(vals_a, vals_b):
    """Example decision rule: non-overlapping per-seed min-max ranges."""
    return (min(vals_a) > max(vals_b)) or (min(vals_b) > max(vals_a))


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    all_agents = load_all_results()

    print("=" * 95)
    print(f"{'Agent':<20} | {'Seed 0':>8} | {'Seed 1':>8} | {'Seed 2':>8} | {'Mean':>8} | {'Steps->150 (S0,S1,S2)':<22} | {'Avg Time'}")
    print("-" * 95)

    for name, data in all_agents.items():
        rets = [data[str(s)]["final_return"] for s in [0, 1, 2]]
        s150 = [str(data[str(s)]["steps_to_150"]) for s in [0, 1, 2]]
        times = [data[str(s)]["wall_time"] for s in [0, 1, 2]]
        print(
            f"{name:<20} | {rets[0]:>8.2f} | {rets[1]:>8.2f} | {rets[2]:>8.2f} | "
            f"{np.mean(rets):>8.2f} | {', '.join(s150):<22} | {np.mean(times):>6.2f}s"
        )
    print("=" * 95)

    # Pairwise comparison check against DQN (Full)
    base_rets = [all_agents["DQN (Full)"][str(s)]["final_return"] for s in [0, 1, 2]]
    print("\nDecision Rule Check (Non-overlapping per-seed final return ranges vs DQN Full):")
    for name, data in all_agents.items():
        if name == "DQN (Full)":
            continue
        other_rets = [data[str(s)]["final_return"] for s in [0, 1, 2]]
        diff = ranges_do_not_overlap(base_rets, other_rets)
        print(
            f"  DQN (Full) [{min(base_rets):.1f}, {max(base_rets):.1f}] vs "
            f"{name:<18} [{min(other_rets):.1f}, {max(other_rets):.1f}] -> "
            f"{'PASS (Real Difference)' if diff else 'FAIL (Overlapping Ranges)'}"
        )

    # Figure 1: Learning curves of all 6 agents (2x3 grid, every seed drawn, mean in bold)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    colors = ["#1C3D6E", "#B22222", "#D97706", "#0D9488", "#6D28D9", "#15803D"]
    for ax, (name, data), color in zip(axes.flat, all_agents.items(), colors):
        curves = {int(s): np.array(data[s]["curve"]) for s in ["0", "1", "2"]}
        S.plot_curves(ax, curves, name, color=color)

    fig.suptitle("CartPole Learning Curves Across All Six Agents (3 Seeds Each, 20k Steps)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/all_agents_learning_curves.png", dpi=150)
    plt.close(fig)
    print("\nSaved figures/all_agents_learning_curves.png")

    # Figure 2: Probe Q-values over training vs theoretical bound of 100
    q_agents = [
        ("DQN (Full)", "#1C3D6E"),
        ("DQN (No Target)", "#B22222"),
        ("DQN (No Buffer)", "#D97706"),
        ("DQN (Buffer 1k)", "#0D9488"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, (name, color) in zip(axes, q_agents):
        data = all_agents[name]
        xs = data["0"]["q_probe_steps"]
        mat = np.array([data[str(s)]["q_probe_means"] for s in [0, 1, 2]])
        for row in mat:
            ax.plot(xs, row, color=color, alpha=0.35, lw=1)
        ax.plot(xs, mat.mean(axis=0), color=color, lw=2.2, label="Mean max Q")
        ax.axhline(100.0, color="black", linestyle="--", lw=1.2, label="True Bound (100)")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean max Q on 500 probe states")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Probe Q-Value Growth vs. Theoretical Upper Bound (100)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/q_values_vs_bound.png", dpi=150)
    plt.close(fig)
    print("Saved figures/q_values_vs_bound.png")