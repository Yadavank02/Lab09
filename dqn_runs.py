import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import lab09_starter as S
from checks import compute_dqn_target, huber_loss_manual


class ReplayBuffer:
    def __init__(self, capacity=20000, obs_dim=4):
        self.capacity = capacity
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, term):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s2
        self.terminated[self.ptr] = term
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return self._get_batch(idx)

    def sample_recent(self, batch_size):
        """Returns the most recent `batch_size` transitions in chronological order."""
        assert self.size >= batch_size
        indices = [(self.ptr - batch_size + i) % self.capacity for i in range(batch_size)]
        return self._get_batch(np.array(indices))

    def _get_batch(self, idx):
        return (
            torch.tensor(self.states[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.int64),
            torch.tensor(self.rewards[idx], dtype=torch.float32),
            torch.tensor(self.next_states[idx], dtype=torch.float32),
            torch.tensor(self.terminated[idx], dtype=torch.bool),
        )


def linear_epsilon(step, eps_start=1.0, eps_end=0.05, decay_steps=8000):
    if step >= decay_steps:
        return eps_end
    return eps_start + (eps_end - eps_start) * (step / decay_steps)


def train_dqn(
    seed,
    total_steps=20000,
    buffer_capacity=20000,
    use_target_net=True,
    use_replay_buffer=True,
    target_update_freq=250,
    batch_size=64,
    gamma=0.99,
    lr=0.001,
    grad_clip=10.0,
):
    S.set_seed(seed)
    rng = np.random.default_rng(seed)
    env = S.CartPole(seed=seed)

    q_net = S.QNet(obs_dim=4, n_actions=2)
    target_net = S.QNet(obs_dim=4, n_actions=2)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

    buf_cap = buffer_capacity if use_replay_buffer else batch_size
    start_learning = 500 if use_replay_buffer else batch_size
    buffer = ReplayBuffer(capacity=buf_cap, obs_dim=4)

    probes = S.probe_states()
    q_probe_steps = []
    q_probe_means = []
    max_abs_q_seen = 0.0

    episodes = []
    s = env.reset()
    ep_return = 0.0
    t0 = time.time()

    for step in range(1, total_steps + 1):
        eps = linear_epsilon(step - 1)
        if rng.random() < eps:
            action = int(rng.integers(0, 2))
        else:
            with torch.no_grad():
                q_vals = q_net(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
                action = int(q_vals.argmax(dim=1).item())

        s2, r, term, trunc = env.step(action)
        ep_return += r
        # Store ONLY true termination (never truncation)
        buffer.add(s, action, r, s2, term)

        if term or trunc:
            episodes.append((step, ep_return))
            s = env.reset()
            ep_return = 0.0
        else:
            s = s2

        # Learning step
        if step >= start_learning and buffer.size >= batch_size:
            if use_replay_buffer:
                b_s, b_a, b_r, b_s2, b_term = buffer.sample(batch_size, rng)
            else:
                b_s, b_a, b_r, b_s2, b_term = buffer.sample_recent(batch_size)

            with torch.no_grad():
                ref_net = target_net if use_target_net else q_net
                next_q_vals = ref_net(b_s2)
                targets = compute_dqn_target(b_r, next_q_vals, gamma, b_term)

            q_pred = q_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
            loss = huber_loss_manual(q_pred, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
            optimizer.step()

        if use_target_net and step % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Measure Q-values on the 500 fixed probe states every 1,000 steps
        if step % 1000 == 0:
            with torch.no_grad():
                pq = q_net(probes)
                mean_max_q = float(pq.max(dim=1).values.mean().item())
                abs_max_q = float(pq.abs().max().item())
                max_abs_q_seen = max(max_abs_q_seen, abs_max_q)
                q_probe_steps.append(step)
                q_probe_means.append(mean_max_q)

    wall_time = time.time() - t0
    curve = S.smooth_curve(episodes)
    fin_ret = S.final_return(episodes)
    step_150 = S.steps_to_reach(curve, 150)

    return {
        "seed": seed,
        "episodes": episodes,
        "curve": curve.tolist(),
        "final_return": fin_ret,
        "steps_to_150": step_150,
        "q_probe_steps": q_probe_steps,
        "q_probe_means": q_probe_means,
        "max_abs_q_seen": max_abs_q_seen,
        "wall_time": wall_time,
    }


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    dqn_results = {}
    print("Running Piece 2: Standard DQN (Seeds 0, 1, 2)...")
    for seed in [0, 1, 2]:
        res = train_dqn(seed=seed)
        dqn_results[str(seed)] = res
        print(
            f"  Seed {seed}: final_return = {res['final_return']:.2f}, "
            f"steps_to_150 = {res['steps_to_150']}, "
            f"max_|Q| = {res['max_abs_q_seen']:.2f}, "
            f"time = {res['wall_time']:.2f}s"
        )

    with open("results/dqn_standard.json", "w") as f:
        json.dump(dqn_results, f, indent=2)
    print("Saved results/dqn_standard.json")