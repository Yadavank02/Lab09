import os
import json
from dqn_runs import train_dqn


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    configs = {
        "dqn_no_target": {
            "label": "DQN (No Target Network)",
            "kwargs": {"use_target_net": False, "use_replay_buffer": True, "buffer_capacity": 20000},
        },
        "dqn_no_buffer": {
            "label": "DQN (No Replay Buffer)",
            "kwargs": {"use_target_net": True, "use_replay_buffer": False},
        },
        "dqn_buffer_1000": {
            "label": "DQN (Buffer = 1,000)",
            "kwargs": {"use_target_net": True, "use_replay_buffer": True, "buffer_capacity": 1000},
        },
    }

    all_ablation_results = {}
    for key, cfg in configs.items():
        print(f"\nRunning Piece 3: {cfg['label']}...")
        all_ablation_results[key] = {}
        for seed in [0, 1, 2]:
            res = train_dqn(seed=seed, **cfg["kwargs"])
            all_ablation_results[key][str(seed)] = res
            final_probe_q = res["q_probe_means"][-1]
            print(
                f"  Seed {seed}: final_return = {res['final_return']:.2f}, "
                f"steps_to_150 = {res['steps_to_150']}, "
                f"final_mean_max_Q = {final_probe_q:.2f}, "
                f"max_|Q|_ever = {res['max_abs_q_seen']:.2f} (bound=100)"
            )

    with open("results/dqn_ablations.json", "w") as f:
        json.dump(all_ablation_results, f, indent=2)
    print("\nSaved results/dqn_ablations.json")