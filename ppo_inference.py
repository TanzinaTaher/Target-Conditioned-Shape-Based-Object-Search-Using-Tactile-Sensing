"""
ppo_inference.py

Run a trained PPO policy for per-shape tactile pose selection.

- Loads ppo_policy_<target>.pth
- Uses TactileSearchEnv (same as in training)
- Runs one episode (~max_touches steps)
- Prints classifier probabilities after each touch
"""

import os
import numpy as np
import torch

from motion_controller import TactileExplorer
from classifier import ShapeClassifier
from search_manager import SearchManager
from tactile_env import TactileSearchEnv
from train_ppo import ActorCritic  # reuse the same network architecture


def run_ppo_inference(target, model_path, max_touches=8, device="cpu"):
    device = torch.device(device)

    # ----- Classifier -----
    classifier = ShapeClassifier(
        weight_path="best_tactile_classifier_convnet.pth",
        device=str(device)
    )

    # ----- SearchManager (needed for TactileExplorer compatibility) -----
    CONF_THRESHOLD = 0.90
    search_mgr = SearchManager(
        classifier=classifier,
        target_label=target,
        conf_threshold=CONF_THRESHOLD
    )

    # ----- TactileExplorer (MuJoCo controller + tactile images) -----
    explorer = TactileExplorer(search_mgr)

    # ----- Environment (same as in training) -----
    env = TactileSearchEnv(
        motion_controller=explorer,
        classifier=classifier,
        target_label=target,
        max_touches=max_touches
    )

    # ----- Build policy and load weights -----
    # We need a dummy reset to know state_dim
    state = env.reset()
    state_dim = state.shape[0]
    num_actions = env.num_actions

    policy = ActorCritic(state_dim, num_actions).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    print(f"\nRunning PPO inference for target='{target}' with model: {model_path}")
    print(f"max_touches = {max_touches}\n")

    # ----- Run a single episode -----
    done = False
    step_idx = 0

    while not done:
        step_idx += 1
        s_t = torch.tensor(state, dtype=torch.float32, device=device)

        # You can sample (stochastic) or take argmax for greedy behaviour.
        with torch.no_grad():
            logits, _ = policy(s_t.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            # For deterministic:
            # action = torch.argmax(logits, dim=-1).item()

        next_state, reward, done, info = env.step(action)

        probs = info["probs"]
        p_correct = info["p_correct"]

        print(
            f"[touch {info['touch_count']:02d}] "
            f"action={action}  "
            f"probs={np.round(probs, 3)}  "
            f"p_correct={p_correct:.3f}  "
            f"reward={reward:.3f}"
        )

        state = next_state

    print("\nEpisode finished.\n")


def main():
    print("Available target objects: sphere, cube, cylinder, cone")
    target = input("Enter target object for PPO inference: ").strip().lower()

    if target not in ["sphere", "cube", "cylinder", "cone"]:
        print("Invalid target object.")
        return

    # Default model name follows train_ppo.py
    default_model_path = f"ppo_policy_{target}.pth"
    model_path = input(f"Enter PPO model path [{default_model_path}]: ").strip()
    if model_path == "":
        model_path = default_model_path

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    run_ppo_inference(target, model_path, max_touches=8, device="cpu")


if __name__ == "__main__":
    main()
