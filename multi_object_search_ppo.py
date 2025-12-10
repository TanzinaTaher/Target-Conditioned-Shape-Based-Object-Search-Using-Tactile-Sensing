"""
multi_object_search_ppo.py

Multi-object tactile search using a pre-trained PPO policy with
confirm / reject actions.

- 4 objects: sphere, cube, cylinder, cone
- Random order per run, with the target object always placed last (worst-case).
- For each object, we run ONE TactileSearchEnv episode:
    * PPO chooses between 6 exploration poses, CONFIRM, or REJECT.
    * Exploration steps generate tactile images and incremental beliefs.
    * CONFIRM / REJECT end the episode.
- We show MuJoCo viewer so the touches are visible.
- At the end, we summarize P(target | images) for each object and
  report whether PPO correctly identified the target object.

Usage:
    mjpython multi_object_search_ppo.py
"""

import os
import random
import numpy as np
import torch
import mujoco

from motion_controller import TactileExplorer
from classifier import ShapeClassifier
from search_manager import SearchManager
from tactile_env import TactileSearchEnv
from train_ppo import ActorCritic  # reuse the same network architecture


def run_multi_object_search(
    target_label: str,
    model_path: str,
    device: str = "cpu",
):
    device = torch.device(device)

    # ----- Classifier -----
    classifier = ShapeClassifier(
        weight_path="best_tactile_classifier_convnet.pth",
        device=str(device),
    )

    class_names = list(classifier.class_names)
    num_classes = len(class_names)
    target_index = class_names.index(target_label)

    # ----- SearchManager -----
    CONF_THRESHOLD = 0.90
    search_mgr = SearchManager(
        classifier=classifier,
        target_label=target_label,
        conf_threshold=CONF_THRESHOLD,
    )

    # ----- TactileExplorer -----
    explorer = TactileExplorer(search_mgr)

    # ----- PPO Environment -----
    # Values should match training (max_steps, step_penalty, positive_ratio),
    # but here we override object_label explicitly, so positive_ratio is unused.
    env = TactileSearchEnv(
        motion_controller=explorer,
        classifier=classifier,
        target_label=target_label,
        max_steps=20,
        step_penalty=0.02,
        positive_ratio=0.5,
    )

    num_actions = env.num_actions

    # ----- PPO Policy -----
    dummy_state = env.reset(object_label=target_label)
    state_dim = dummy_state.shape[0]

    policy = ActorCritic(state_dim, num_actions).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    # ----- Object order: random non-targets + target last -----
    all_objects = ["sphere", "cube", "cylinder", "cone"]
    assert target_label in all_objects, "Target must be one of the 4 known classes."

    non_targets = [o for o in all_objects if o != target_label]
    random.shuffle(non_targets)
    obj_sequence = non_targets + [target_label]

    print("\n=== Multi-object PPO Search ===")
    print(f"Target class: {target_label}")
    print(f"Object visit order (unknown to classifier/PPO): {obj_sequence}\n")

    # To store final P(target) for each object
    object_beliefs = {}
    predicted_target_obj = None

    # ---------------------------------------------------------
    # Launch MuJoCo viewer so PPO touches are visible
    # ---------------------------------------------------------
    with mujoco.viewer.launch_passive(explorer.model, explorer.data) as viewer:
        explorer.viewer = viewer

        # -----------------------------------------------------
        # Loop over each object in the sequence
        # -----------------------------------------------------
        for obj_label in obj_sequence:
            print(f"\n--- Exploring object: {obj_label} ---")

            # Reset env for THIS object (one episode per object)
            state = env.reset(object_label=obj_label)
            last_probs = np.ones(num_classes, dtype=np.float32) / float(num_classes)

            while True:
                # PPO chooses action from current state
                s_t = torch.tensor(state, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action, _, _ = policy.act(s_t)

                next_state, reward, done, info = env.step(action)
                probs = info["probs"]
                last_probs = probs

                # Decode predicted class from probs
                pred_idx = int(np.argmax(probs))
                pred_label = class_names[pred_idx]
                pred_conf = float(probs[pred_idx])
                p_target = float(probs[target_index])

                step_id = info["step_count"]

                if action < env.num_pose_actions:
                    # Exploration pose
                    print(
                        f"[step {step_id:02d}] action={action} (pose) "
                        f"pred={pred_label} ({pred_conf:.2f}) "
                        f"P(target={target_label})={p_target:.3f} "
                        f"reward={reward:.3f}"
                    )
                elif action == env.CONFIRM_ACTION:
                    print(
                        f"[step {step_id:02d}] action=CONFIRM  "
                        f"is_current_target={info['is_current_target']} "
                        f"reward={reward:.3f}"
                    )
                elif action == env.REJECT_ACTION:
                    print(
                        f"[step {step_id:02d}] action=REJECT   "
                        f"is_current_target={info['is_current_target']} "
                        f"reward={reward:.3f}"
                    )
                else:
                    print(
                        f"[step {step_id:02d}] action={action} (UNKNOWN) reward={reward:.3f}"
                    )

                state = next_state

                if done:
                    # Episode for this object finished
                    p_target_final = float(last_probs[target_index])
                    object_beliefs[obj_label] = p_target_final

                    print(
                        f"Final belief for object {obj_label}: "
                        f"P(target={target_label} | images) = {p_target_final:.3f}"
                    )

                    # If agent CONFIRMED this object, treat that as global search decision
                    if action == env.CONFIRM_ACTION:
                        predicted_target_obj = obj_label
                    # If REJECT or timeout, we just move to the next object

                    break  # break inner while

            # If some object was confirmed as target, stop searching others
            if predicted_target_obj is not None:
                break

        explorer.viewer = None

    # ---------------------------------------------------------
    # Summary: which object looks like the target?
    # ---------------------------------------------------------
    print("\n=== Summary of beliefs (P(target | object)) ===")
    for obj_label in all_objects:
        if obj_label in object_beliefs:
            p_val = object_beliefs[obj_label]
            print(f"  {obj_label:8s} -> {p_val:.3f}")
        else:
            print(f"  {obj_label:8s} -> (not visited)")

    # If PPO never explicitly confirmed any object, fall back to argmax belief
    if predicted_target_obj is None and len(object_beliefs) > 0:
        print("\nPPO never issued CONFIRM; using argmax of P(target | object) as fallback.")
        predicted_target_obj = max(object_beliefs.items(), key=lambda kv: kv[1])[0]

    print(f"\nPredicted target object: {predicted_target_obj}")
    print(f"True target object     : {target_label}")

    if predicted_target_obj == target_label:
        print("✅ Multi-object search SUCCESS: correct target identified.")
    else:
        print("❌ Multi-object search FAILURE: wrong target identified.")


def main():
    print("Available target objects: sphere, cube, cylinder, cone")
    target = input("Enter target object for PPO multi-object search: ").strip().lower()

    if target not in ["sphere", "cube", "cylinder", "cone"]:
        print("Invalid target object.")
        return

    default_model_path = f"ppo_policy_{target}.pth"
    model_path = input(f"Enter PPO model path [{default_model_path}]: ").strip()
    if model_path == "":
        model_path = default_model_path

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    run_multi_object_search(
        target_label=target,
        model_path=model_path,
        device="cpu",
    )


if __name__ == "__main__":
    main()
