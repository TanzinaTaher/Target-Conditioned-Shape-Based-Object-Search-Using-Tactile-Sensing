import numpy as np
import math
import time


class TactileSearchEnv:
    """
    PPO training environment for target-conditioned tactile search
    on a SINGLE object.

    Actions:
        0..5 : probe poses
        6    : confirm (object IS target)
        7    : reject  (object is NOT target)
    """

    def __init__(
        self,
        motion_controller,
        classifier,
        target_label,
        max_steps=20,
        step_penalty=0.02,
        positive_ratio=0.25,
        min_decision_steps=2,
        delta_scale=0.5,
    ):
        self.mc = motion_controller
        self.clf = classifier
        self.target_label = target_label

        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.positive_ratio = positive_ratio
        self.min_decision_steps = min_decision_steps
        self.delta_scale = delta_scale

        # Actions: 6 poses + confirm + reject
        self.num_pose_actions = 6
        self.CONFIRM_ACTION = 6
        self.REJECT_ACTION = 7
        self.num_actions = 8

        # Class setup
        self.class_names = list(self.clf.class_names)
        self.num_classes = len(self.class_names)
        self.target_index = self.class_names.index(self.target_label)

        # STATE = [probs(4), action_onehot(8), step_fraction(1)]
        self.state_dim = self.num_classes + self.num_actions + 1

        # Episode vars
        self.step_count = 0
        self.prev_probs = np.ones(self.num_classes, dtype=np.float32) / self.num_classes

        self.current_object_label = None
        self.is_current_target = False

        self.search_manager = getattr(self.mc, "search_manager", None)

        self.last_force = 0.0
        self.last_xy = (0.0, 0.0)

    # ================================================================
    # RESET EPISODE
    # ================================================================
    def reset(self, object_label=None):

        # Choose object for this episode
        if object_label is not None:
            self.current_object_label = object_label
            self.is_current_target = (object_label == self.target_label)
        else:
            if np.random.rand() < self.positive_ratio:
                obj = self.target_label
                self.is_current_target = True
            else:
                candidates = [c for c in self.class_names if c != self.target_label]
                obj = np.random.choice(candidates)
                self.is_current_target = False
            self.current_object_label = obj

        # Reset MuJoCo + SearchManager
        self.mc._park_all_objects()
        self.mc._set_active_object(self.current_object_label)

        if self.search_manager is not None:
            if hasattr(self.search_manager, "set_current_object"):
                self.search_manager.set_current_object(self.current_object_label)
            if hasattr(self.search_manager, "reset_current_object_state"):
                self.search_manager.reset_current_object_state()

            # initialize fused probs to uniform
            self.search_manager.last_probs = (
                np.ones(self.num_classes, dtype=np.float32) / self.num_classes
            )

        # Episode vars
        self.step_count = 0
        self.prev_probs = np.ones(self.num_classes, dtype=np.float32) / self.num_classes
        self.last_force = 0.0
        self.last_xy = (0.0, 0.0)

        return self._build_state(self.prev_probs, last_action=-1)

    # ================================================================
    # STEP
    # ================================================================
    def step(self, action):
        self.step_count += 1
        done = False
        reward = 0.0

        # Start from last fused probabilities if available
        if self.search_manager is not None and self.search_manager.last_probs is not None:
            probs = self.search_manager.last_probs.astype(np.float32)
        else:
            probs = self.prev_probs.copy()

        prev_p_target = float(probs[self.target_index])

        # ------------------------------------------------------
        # (A) POSE ACTIONS (0..5)
        # ------------------------------------------------------
        if 0 <= action < self.num_pose_actions:

            img_path, force_val, (x, y) = self.mc.perform_single_touch(action)

            if img_path is not None and self.search_manager is not None:
                self.search_manager.add_image(img_path)
                self.search_manager.classify_incremental()

                # Use fused Bayesian probabilities
                probs = self.search_manager.last_probs.astype(np.float32)

            else:
                force_val = self.last_force
                x, y = self.last_xy

            # Reward shaping
            new_p_target = float(probs[self.target_index])
            delta_p = new_p_target - prev_p_target

            reward = -self.step_penalty + self.delta_scale * delta_p
            done = False

            # Timeout penalty
            if self.step_count >= self.max_steps:
                reward -= 1.0
                done = True

        # ------------------------------------------------------
        # (B) BLOCK EARLY DECISIONS
        # ------------------------------------------------------
        elif action in (self.CONFIRM_ACTION, self.REJECT_ACTION) and (
            self.step_count < self.min_decision_steps
        ):
            reward = -self.step_penalty
            done = False
            force_val = self.last_force
            x, y = self.last_xy

        # ------------------------------------------------------
        # (C) CONFIRM
        # ------------------------------------------------------
        elif action == self.CONFIRM_ACTION:
            reward = 1.0 if self.is_current_target else -1.0
            done = True
            force_val = self.last_force
            x, y = self.last_xy

        # ------------------------------------------------------
        # (D) REJECT
        # ------------------------------------------------------
        elif action == self.REJECT_ACTION:
            reward = 1.0 if (not self.is_current_target) else -1.0
            done = True
            force_val = self.last_force
            x, y = self.last_xy

        else:
            raise ValueError(f"Invalid action index: {action}")

        # ------------------------------------------------------
        # Update internal memory
        # ------------------------------------------------------
        # 🔥 CRITICAL FIX — store the fused probs for next step
        if self.search_manager is not None and self.search_manager.last_probs is not None:
            self.prev_probs = self.search_manager.last_probs.astype(np.float32)
        else:
            self.prev_probs = probs

        self.last_force = float(force_val)
        self.last_xy = (float(x), float(y))

        state = self._build_state(self.prev_probs, last_action=action)

        info = {
            "probs": probs,
            "step_count": self.step_count,
            "last_action": action,
            "is_current_target": self.is_current_target,
            "current_object_label": self.current_object_label,
        }

        return state, float(reward), bool(done), info

    # ================================================================
    # BUILD STATE VECTOR
    # ================================================================
    def _build_state(self, probs, last_action):

        probs = probs.astype(np.float32)

        # One-hot encode last action
        action_onehot = np.zeros(self.num_actions, dtype=np.float32)
        if 0 <= last_action < self.num_actions:
            action_onehot[last_action] = 1.0

        step_frac = np.array(
            [self.step_count / float(self.max_steps)],
            dtype=np.float32,
        )

        state = np.concatenate([probs, action_onehot, step_frac], axis=0)
        return state.astype(np.float32)
