import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import lab09_starter as S
from checks import compute_returns_to_go, ppo_clipped_objective


def train_reinforce(
    seed,
    total_steps=20000,
    episodes_per_update=8,
    gamma=0.99,
    lr_policy=0.003,
    lr_value=0.01,
):
    S.set_seed(seed)
    env = S.CartPole(seed=seed)

    policy_net = S.mlp(4, 2)
    value_net = S.mlp(4, 1)
    opt_policy = torch.optim.Adam(policy_net.parameters(), lr=lr_policy)
    opt_value = torch.optim.Adam(value_net.parameters(), lr=lr_value)

    episodes = []
    step = 0
    t0 = time.time()

    while step < total_steps:
        batch_states = []
        batch_actions = []
        batch_returns = []
        eps_collected = 0

        while eps_collected < episodes_per_update and step < total_steps:
            s = env.reset()
            ep_states, ep_actions, ep_rewards = [], [], []
            ep_ret = 0.0

            while step < total_steps:
                s_t = torch.tensor(s, dtype=torch.float32)
                logits = policy_net(s_t)
                dist = Categorical(logits=logits)
                a = dist.sample()

                s2, r, term, trunc = env.step(int(a.item()))
                step += 1
                ep_ret += r

                ep_states.append(s)
                ep_actions.append(int(a.item()))
                ep_rewards.append(r)

                if term or trunc:
                    episodes.append((step, ep_ret))
                    break
                s = s2

            # Compute return-to-go for collected episode steps
            rtg = compute_returns_to_go(ep_rewards, gamma)
            batch_states.extend(ep_states)
            batch_actions.extend(ep_actions)
            batch_returns.extend(rtg.tolist())
            eps_collected += 1

        if len(batch_states) == 0:
            break

        states_t = torch.tensor(np.array(batch_states), dtype=torch.float32)
        actions_t = torch.tensor(batch_actions, dtype=torch.int64)
        returns_t = torch.tensor(batch_returns, dtype=torch.float32)

        # Compute advantages BEFORE updating value function
        with torch.no_grad():
            values_pred = value_net(states_t).squeeze(-1)
            advantages = returns_t - values_pred
            if advantages.numel() > 1:
                advantages = advantages / (advantages.std(unbiased=False) + 1e-8)

        # Policy update
        logits = policy_net(states_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        policy_loss = -(log_probs * advantages).mean()

        opt_policy.zero_grad()
        policy_loss.backward()
        opt_policy.step()

        # Value baseline update
        values_new = value_net(states_t).squeeze(-1)
        value_loss = F.mse_loss(values_new, returns_t)

        opt_value.zero_grad()
        value_loss.backward()
        opt_value.step()

    wall_time = time.time() - t0
    curve = S.smooth_curve(episodes)
    return {
        "seed": seed,
        "episodes": episodes,
        "curve": curve.tolist(),
        "final_return": S.final_return(episodes),
        "steps_to_150": S.steps_to_reach(curve, 150),
        "wall_time": wall_time,
    }


def train_ppo(
    seed,
    total_steps=20000,
    rollout_steps=1024,
    epochs=6,
    minibatch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    lr=0.0003,
    vf_coef=0.5,
    max_grad_norm=0.5,
):
    S.set_seed(seed)
    rng = np.random.default_rng(seed)
    env = S.CartPole(seed=seed)

    policy_net = S.mlp(4, 2)
    value_net = S.mlp(4, 1)
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()), lr=lr
    )

    episodes = []
    step = 0
    s = env.reset()
    ep_ret = 0.0
    t0 = time.time()

    while step < total_steps:
        n_collect = min(rollout_steps, total_steps - step)
        states = np.zeros((n_collect, 4), dtype=np.float32)
        actions = np.zeros(n_collect, dtype=np.int64)
        rewards = np.zeros(n_collect, dtype=np.float32)
        terminated_flags = np.zeros(n_collect, dtype=np.float32)
        done_flags = np.zeros(n_collect, dtype=np.float32)
        old_log_probs = np.zeros(n_collect, dtype=np.float32)
        values = np.zeros(n_collect, dtype=np.float32)

        for i in range(n_collect):
            states[i] = s
            s_t = torch.tensor(s, dtype=torch.float32)
            with torch.no_grad():
                dist = Categorical(logits=policy_net(s_t))
                a = dist.sample()
                lp = dist.log_prob(a)
                v = value_net(s_t).squeeze(-1)

            actions[i] = int(a.item())
            old_log_probs[i] = float(lp.item())
            values[i] = float(v.item())

            s2, r, term, trunc = env.step(actions[i])
            step += 1
            ep_ret += r

            rewards[i] = r
            # True termination prevents bootstrap; time-limit truncation still bootstraps
            terminated_flags[i] = 1.0 if term else 0.0
            done_flags[i] = 1.0 if (term or trunc) else 0.0

            if term or trunc:
                episodes.append((step, ep_ret))
                s = env.reset()
                ep_ret = 0.0
            else:
                s = s2

        # Bootstrap value for final state in rollout
        with torch.no_grad():
            last_value = float(value_net(torch.tensor(s, dtype=torch.float32)).item())

        # Compute GAE advantages BEFORE policy/value updates
        advantages = np.zeros(n_collect, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n_collect)):
            next_val = last_value if t == n_collect - 1 else values[t + 1]
            # If episode ended on step t (either term or trunc), next step is a fresh reset,
            # EXCEPT when truncated at t=200 where we want the value of the final state s2.
            # Standard handling: if terminated, next_non_terminal=0; if truncated, bootstrap using s2.
            next_non_terminal = 1.0 - terminated_flags[t]
            if done_flags[t] == 1.0 and terminated_flags[t] == 0.0:
                # Truncated at 200 steps: bootstrap from s2 (which was already stepped before reset)
                # Note: in our loop s was reset to env.reset(), so for exactness on truncated steps:
                next_val = values[t]  # approximate continuation value at boundary
            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            reset_mask = 1.0 - done_flags[t]
            gae = delta + gamma * gae_lambda * reset_mask * gae
            advantages[t] = gae

        returns = advantages + values

        # Normalize advantages within rollout
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64)
        old_lp_t = torch.tensor(old_log_probs, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)

        for _ in range(epochs):
            perm = rng.permutation(n_collect)
            for start in range(0, n_collect, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]
                dist = Categorical(logits=policy_net(states_t[mb_idx]))
                new_lp = dist.log_prob(actions_t[mb_idx])
                ratios = torch.exp(new_lp - old_lp_t[mb_idx])

                p_loss = -ppo_clipped_objective(ratios, adv_t[mb_idx], clip_eps=clip_eps)
                v_pred = value_net(states_t[mb_idx]).squeeze(-1)
                v_loss = F.mse_loss(v_pred, ret_t[mb_idx])

                loss = p_loss + vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(policy_net.parameters()) + list(value_net.parameters()),
                    max_grad_norm,
                )
                optimizer.step()

    wall_time = time.time() - t0
    curve = S.smooth_curve(episodes)
    return {
        "seed": seed,
        "episodes": episodes,
        "curve": curve.tolist(),
        "final_return": S.final_return(episodes),
        "steps_to_150": S.steps_to_reach(curve, 150),
        "wall_time": wall_time,
    }


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    pg_results = {"reinforce": {}, "ppo": {}}

    print("Running Piece 4: REINFORCE (with baseline)...")
    for seed in [0, 1, 2]:
        res = train_reinforce(seed=seed)
        pg_results["reinforce"][str(seed)] = res
        print(
            f"  Seed {seed}: final_return = {res['final_return']:.2f}, "
            f"steps_to_150 = {res['steps_to_150']}, "
            f"wall_time = {res['wall_time']:.2f}s"
        )

    print("\nRunning Piece 4: PPO...")
    for seed in [0, 1, 2]:
        res = train_ppo(seed=seed)
        pg_results["ppo"][str(seed)] = res
        print(
            f"  Seed {seed}: final_return = {res['final_return']:.2f}, "
            f"steps_to_150 = {res['steps_to_150']}, "
            f"wall_time = {res['wall_time']:.2f}s"
        )

    with open("results/policy_gradient.json", "w") as f:
        json.dump(pg_results, f, indent=2)
    print("\nSaved results/policy_gradient.json")