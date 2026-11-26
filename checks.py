import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lab09_starter as S


def compute_dqn_target(reward, next_q_values, gamma, terminated):
    """
    Computes y = r + gamma * max_a' Q_target(s', a') * (1 - terminated).
    IMPORTANT: terminated must be True ONLY for true failure termination,
    NOT for 200-step time-limit truncation.
    """
    with torch.no_grad():
        max_next_q = next_q_values.max(dim=-1).values
        not_done = 1.0 - terminated.float()
        return reward + gamma * max_next_q * not_done


def huber_loss_manual(pred, target, delta=1.0):
    """Manual Huber (Smooth L1) loss matching PyTorch's reduction='mean'."""
    err = pred - target
    abs_err = torch.abs(err)
    quadratic = torch.minimum(abs_err, torch.tensor(delta, device=err.device))
    linear = abs_err - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss.mean()


def ppo_clipped_objective(ratios, advantages, clip_eps=0.2):
    """
    PPO clipped surrogate objective (to be MAXIMIZED; negate for gradient descent).
    L^CLIP = E[ min(r_t * A_t, clip(r_t, 1 - eps, 1 + eps) * A_t) ]
    """
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return torch.min(unclipped, clipped).mean()


def compute_returns_to_go(rewards, gamma):
    """Computes discounted return-to-go G_t = sum_{k=t}^T gamma^{k-t} r_k."""
    rtg = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        rtg[t] = running
    return rtg


def run_fixed_point_check(use_target_net=True, updates=3000, target_refresh=50):
    """
    Single state, single action, r=1, s'=s, never terminal, gamma=0.9 -> true Q* = 10.0.
    """
    S.set_seed(0)
    q_net = S.QNet(obs_dim=4, n_actions=2)
    target_net = S.QNet(obs_dim=4, n_actions=2)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=0.01)
    state = torch.tensor([[0.1, -0.2, 0.05, 0.1]], dtype=torch.float32)
    reward = torch.tensor([1.0], dtype=torch.float32)
    terminated = torch.tensor([False])

    history = []
    for step in range(1, updates + 1):
        with torch.no_grad():
            ref_net = target_net if use_target_net else q_net
            next_q = ref_net(state)
            target = compute_dqn_target(reward, next_q, gamma=0.9, terminated=terminated)

        pred = q_net(state)[:, 0]
        loss = huber_loss_manual(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if use_target_net and step % target_refresh == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step in {1, 100, 500, 1000, 2000, 3000}:
            history.append((step, float(pred.item())))

    return float(q_net(state)[0, 0].item()), history


if __name__ == "__main__":
    print("=" * 60)
    print("PIECE 1 VERIFICATION CHECKS")
    print("=" * 60)

    # 1. DQN Target checks (r=1, max_next_q=7, gamma=0.9)
    r = torch.tensor([1.0])
    next_q = torch.tensor([[2.0, 7.0]])
    y_term = compute_dqn_target(r, next_q, 0.9, torch.tensor([True])).item()
    y_nonterm = compute_dqn_target(r, next_q, 0.9, torch.tensor([False])).item()
    # Time-limit truncation is NOT a true terminal state -> terminated=False
    y_trunc = compute_dqn_target(r, next_q, 0.9, torch.tensor([False])).item()

    print("\n1. DQN Targets (r=1, max_next_Q=7, gamma=0.9):")
    print(f"   - True terminal transition:      {y_term:.4f} (expected 1.0000)")
    print(f"   - Non-terminal transition:       {y_nonterm:.4f} (expected 7.3000)")
    print(f"   - Time-limit truncated (t=200):  {y_trunc:.4f} (expected 7.3000 - MUST bootstrap)")

    # 2. Huber loss check at errors 0.5, 3.0, -3.0
    print("\n2. Huber Loss (delta=1.0) vs Formula vs PyTorch:")
    for err_val in [0.5, 3.0, -3.0]:
        pred = torch.tensor([err_val])
        targ = torch.tensor([0.0])
        manual = huber_loss_manual(pred, targ).item()
        torch_val = F.huber_loss(pred, targ, delta=1.0).item()
        expected = 0.5 * (err_val ** 2) if abs(err_val) <= 1.0 else (abs(err_val) - 0.5)
        print(f"   - Error {err_val:>4.1f}: manual={manual:.4f}, torch={torch_val:.4f}, formula={expected:.4f}")

    # 3. Fixed-point check (true value = 1 / (1 - 0.9) = 10.0)
    print("\n3. Fixed-Point Check (3,000 updates, Adam lr=0.01, true Q*=10.0):")
    val_with_target, hist_with = run_fixed_point_check(use_target_net=True)
    val_no_target, hist_no = run_fixed_point_check(use_target_net=False)
    print(f"   - With target net (refresh=50): final Q = {val_with_target:.4f} | trace: {hist_with}")
    print(f"   - Without target net:           final Q = {val_no_target:.4f} | trace: {hist_no}")

    # 4. PPO clipped objective check
    print("\n4. PPO Clipped Objective (eps=0.2):")
    cases = [
        (1.3, 1.0, 1.2),    # min(1.3*1, 1.2*1) = 1.2
        (1.3, -1.0, -1.3),  # min(1.3*-1, 1.2*-1) = min(-1.3, -1.2) = -1.3
        (0.7, 1.0, 0.7),    # min(0.7*1, 0.8*1) = 0.7
        (0.7, -1.0, -0.8),  # min(0.7*-1, 0.8*-1) = min(-0.7, -0.8) = -0.8
    ]
    for ratio, adv, hand in cases:
        val = ppo_clipped_objective(torch.tensor([ratio]), torch.tensor([adv])).item()
        print(f"   - ratio={ratio:.1f}, A={adv:>4.1f}: code={val:>7.4f}, hand={hand:>7.4f}")

    # 5. Return-to-go check
    print("\n5. Return-to-Go (rewards=[1, 1, 1], gamma=0.5):")
    rtg = compute_returns_to_go([1.0, 1.0, 1.0], gamma=0.5)
    print(f"   - Result: {rtg.tolist()} (expected [1.75, 1.5, 1.0])")