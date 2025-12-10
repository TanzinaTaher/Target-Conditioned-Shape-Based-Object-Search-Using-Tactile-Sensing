# main.py
import sys
import torch

from motion_controller import TactileExplorer
from classifier import ShapeClassifier
from search_manager import SearchManager
from tactile_env import TactileSearchEnv

VALID_OBJECTS = ["sphere", "cube", "cylinder", "cone"]
DEFAULT_CONF_THRESHOLD = 0.90


# -------------------------------------------------------------
# MANUAL SEARCH (fixed policy)
# -------------------------------------------------------------
def run_manual_search(target):
    """
    Standard full-pipeline tactile search (manual exploration).
    """
    classifier = ShapeClassifier(
        weight_path="best_tactile_classifier_convnet.pth",
        device="cpu"
    )

    search_mgr = SearchManager(
        classifier=classifier,
        target_label=target,
        conf_threshold=DEFAULT_CONF_THRESHOLD
    )

    explorer = TactileExplorer(search_mgr)
    explorer.run_manual()   # IMPORTANT: updated


# -------------------------------------------------------------
# PPO INFERENCE (use a trained PPO policy)
# -------------------------------------------------------------
def run_ppo_inference(target, model_path):
    """
    Run tactile search using a trained PPO policy.
    PPO selects the next tactile pose each step.
    """
    from train_ppo import ActorCritic   # import the policy network

    # Classifier + Search Manager
    classifier = ShapeClassifier(
        weight_path="best_tactile_classifier_convnet.pth", device="cpu"
    )

    search_mgr = SearchManager(
        classifier=classifier,
        target_label=target,
        conf_threshold=DEFAULT_CONF_THRESHOLD
    )

    # Motion controller (single-touch interface)
    explorer = TactileExplorer(search_mgr)

    # PPO environment
    env = TactileSearchEnv(
        motion_controller=explorer,
        classifier=classifier,
        target_label=target,
        max_touches=12
    )

    # Load trained PPO model
    policy = ActorCritic(env.state_dim, env.num_actions)
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    # Run inference
    state = env.reset(object_label=target)
    done = False

    while not done:
        s_t = torch.tensor(state, dtype=torch.float32)
        logits, _ = policy(s_t.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        action = int(dist.sample().item())

        next_state, reward, done, info = env.step(action)
        state = next_state

    print("\n==== PPO Inference Finished ====")
    print(f"Final info: {info}")


# -------------------------------------------------------------
# PPO TRAINING ENTRY (calls train_ppo.py)
# -------------------------------------------------------------
def run_ppo_training(target):
    """
    Calls the PPO training script programmatically.
    """
    from train_ppo import train_ppo

    # Classifier
    classifier = ShapeClassifier(
        weight_path="best_tactile_classifier_convnet.pth",
        device="cpu"
    )

    # Search Manager
    search_mgr = SearchManager(
        classifier=classifier,
        target_label=target,
        conf_threshold=DEFAULT_CONF_THRESHOLD
    )

    # Motion controller
    explorer = TactileExplorer(search_mgr)

    # PPO Env
    env = TactileSearchEnv(
        motion_controller=explorer,
        classifier=classifier,
        target_label=target,
        max_touches=12
    )

    model_path = f"ppo_policy_{target}.pth"

    # Train PPO
    train_ppo(
        env=env,
        num_episodes=200,
        rollout_batch_size=4,
        save_path=model_path,
    )


# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------
def main():
    print("Modes:")
    print("  manual     → fixed exploration policy")
    print("  ppo        → run PPO inference")
    print("  train_ppo  → train PPO agent")
    print()

    mode = input("Select mode (manual/ppo/train_ppo): ").strip().lower()

    if mode not in ["manual", "ppo", "train_ppo"]:
        print("Invalid mode.")
        sys.exit(1)

    print("\nAvailable target objects:", ", ".join(VALID_OBJECTS))
    target = input("Enter target object: ").strip().lower()

    if target not in VALID_OBJECTS:
        print("Invalid object.")
        sys.exit(1)

    # ---------------------------
    # MANUAL MODE
    # ---------------------------
    if mode == "manual":
        run_manual_search(target)
        return

    # ---------------------------
    # PPO INFERENCE
    # ---------------------------
    if mode == "ppo":
        model_path = input("Path to trained PPO model: ").strip()
        run_ppo_inference(target, model_path)
        return

    # ---------------------------
    # PPO TRAINING
    # ---------------------------
    if mode == "train_ppo":
        run_ppo_training(target)
        return


if __name__ == "__main__":
    main()
