"""
train_ppo.py

PPO training script for active tactile shape search.

It uses:
  - TactileExplorer (motion_controller.TactileExplorer)
  - TactileSearchEnv (tactile_env.TactileSearchEnv)
  - ShapeClassifier (classifier.ShapeClassifier)
  - SearchManager (search_manager.SearchManager)

Usage (basic):
    mjpython train_ppo.py

This will:
  - ask for a target label (sphere/cube/cylinder/cone)
  - train a PPO policy for that target
  - save model as ppo_policy_<target>.pth
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from motion_controller import TactileExplorer
from classifier import ShapeClassifier
from search_manager import SearchManager
from tactile_env import TactileSearchEnv


# -------------------------------------------------------------
# PPO Actor-Critic Network
# -------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)  # [B]
        return logits, value

    def act(self, state):
        """
        state: torch.Tensor [state_dim]
        Returns: action (int), log_prob (tensor), value (tensor)
        """
        logits, value = self.forward(state.unsqueeze(0))  # [1, A], [1]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze(0), value.squeeze(0)

    def evaluate_actions(self, states, actions):
        """
        states: [B, state_dim]
        actions: [B]
        Returns: log_probs, entropy, values
        """
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


# -------------------------------------------------------------
# PPO Training Loop
# -------------------------------------------------------------
def train_ppo(
    env,
    num_episodes=1000,       # increase later if needed
    rollout_batch_size=8,   # episodes per PPO update
    gamma=0.99,
    clip_eps=0.2,
    lr=3e-4,
    k_epochs=4,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5,
    device="cpu",
    save_path="ppo__1000_policy.pth"
):
    device = torch.device(device)

    # Use env-defined dimensions (from TactileSearchEnv)
    policy = ActorCritic(env.state_dim, env.num_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []

    def compute_returns(rewards, dones, gamma):
        """
        Compute discounted returns over a flattened rollout.
        Each 'done' resets the return for the next episode.
        """
        returns = []
        G = 0.0
        # Walk backwards through the rollout
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                G = 0.0
            G = r + gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        return np.array(returns, dtype=np.float32)

    total_steps = 0

    for ep in range(num_episodes):
        # Collect rollouts for 'rollout_batch_size' episodes
        all_states = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_dones = []

        for _ in range(rollout_batch_size):
            state = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                s_t = torch.tensor(state, dtype=torch.float32, device=device)
                action, log_prob, _ = policy.act(s_t)

                next_state, reward, done, info = env.step(action)

                all_states.append(state)
                all_actions.append(action)
                all_log_probs.append(log_prob.detach().cpu().numpy())
                all_rewards.append(reward)
                all_dones.append(done)

                state = next_state
                ep_reward += reward
                total_steps += 1

            episode_rewards.append(ep_reward)

        # Convert collected rollouts to tensors
        states = torch.tensor(np.array(all_states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(all_actions), dtype=torch.int64, device=device)
        old_log_probs = torch.tensor(np.array(all_log_probs), dtype=torch.float32, device=device)
        rewards_np = np.array(all_rewards, dtype=np.float32)
        dones_np = np.array(all_dones, dtype=bool)

        # Compute discounted returns
        returns_np = compute_returns(rewards_np, dones_np, gamma)
        returns = torch.tensor(returns_np, dtype=torch.float32, device=device)

        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # PPO update (multiple epochs over the same rollout)
        for _ in range(k_epochs):
            log_probs, entropy, values = policy.evaluate_actions(states, actions)
            advantages = returns - values.detach()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

        avg_reward = np.mean(episode_rewards[-rollout_batch_size:])
        print(
            f"[EP {ep+1:03d}/{num_episodes}] "
            f"avg_reward={avg_reward:.3f}  "
            f"total_steps={total_steps}"
        )

    # Save trained policy
    torch.save(policy.state_dict(), save_path)
    print(f"\n✅ PPO policy saved to: {save_path}")


# -------------------------------------------------------------
# Script entry point
# -------------------------------------------------------------
def main():
    print("Available target objects: sphere, cube, cylinder, cone")
    target = input("Enter target object for PPO training: ").strip().lower()

    if target not in ["sphere", "cube", "cylinder", "cone"]:
        print("Invalid target object.")
        return

    # ----- Classifier (frozen during PPO training) -----
    classifier = ShapeClassifier(
        weight_path="best_tactile_classifier_convnet.pth",
        device="cpu"
    )

    # ----- SearchManager (needed by TactileExplorer, but not used in PPO logic) -----
    CONF_THRESHOLD = 0.90
    search_mgr = SearchManager(
        classifier=classifier,
        target_label=target,
        conf_threshold=CONF_THRESHOLD
    )

    # ----- TactileExplorer (MuJoCo controller; PPO uses perform_single_touch) -----
    explorer = TactileExplorer(search_mgr)

    # ----- PPO Environment (per-object confirm/reject + step penalty) -----
    env = TactileSearchEnv(
        motion_controller=explorer,
        classifier=classifier,
        target_label=target,
        max_steps=20,        # safety cap; conceptually agent decides earlier
        step_penalty=0.02,   # small cost per probe
        positive_ratio=0.5,  # half episodes are true target, half distractors
    )

    # ----- Train PPO -----
    save_name = f"100_ppo_policy_{target}.pth"
    train_ppo(
        env=env,
        num_episodes=100,
        rollout_batch_size=4,
        gamma=0.99,
        clip_eps=0.2,
        lr=3e-4,
        k_epochs=4,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        save_path=save_name
    )


if __name__ == "__main__":
    main()
