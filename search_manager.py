# search_manager.py
'''import numpy as np


class SearchManager:
    def __init__(self,
                 classifier,
                 target_label,
                 conf_threshold,
                 min_images_for_decision: int = 3,
                 consec_target_required: int = 2,
                 consec_skip_required: int = 2,
                 skip_margin: float = 0.35):
        """
        Manage incremental classification + skip / stop logic.

        Args:
            classifier: ShapeClassifier instance
            target_label: string name of target class
            conf_threshold: base confidence threshold (e.g., 0.9)
            min_images_for_decision: minimum images before declaring found/skip
            consec_target_required: how many consecutive high-confidence
                target predictions are needed to declare 'found'
            consec_skip_required: how many consecutive high-confidence
                non-target predictions (with margin) needed to 'skip'
            skip_margin: P(pred) - P(target) must exceed this to skip
        """
        self.classifier = classifier
        self.target_label = target_label
        self.conf_threshold = conf_threshold

        # decision parameters
        self.min_images = min_images_for_decision
        self.consec_target_required = consec_target_required
        self.consec_skip_required = consec_skip_required
        self.skip_margin = skip_margin

        # per-object accumulators
        self.current_images = []     # image paths
        self.current_probs = []      # per-image probability vectors
        self.current_object_label = None
        self.log_prob_sum = None     # accumulated log-probs (for Bayesian fusion)

        # running decision state
        self.target_streak = 0
        self.skip_streak = 0
        self.last_combined_probs = None

    # ---------------------------------------------------------
    # Called when object changes
    # ---------------------------------------------------------
    def set_current_object(self, label: str):
        self.current_object_label = label
        self.current_images = []
        self.current_probs = []
        self.log_prob_sum = None
        self.target_streak = 0
        self.skip_streak = 0
        self.last_combined_probs = None

    # ---------------------------------------------------------
    # Add a new image path
    # ---------------------------------------------------------
    def add_image(self, img_path: str):
        self.current_images.append(img_path)

    # ---------------------------------------------------------
    # Incremental classification
    # ---------------------------------------------------------
    def classify_incremental(self):
        """
        Classify using all images seen so far.

        Returns:
            is_target (bool): whether we should STOP and declare target found
            pred_label (str): current most likely label
            pred_conf (float): its confidence (from fused probs)
        """
        if len(self.current_images) == 0:
            return False, None, 0.0

        last_image = self.current_images[-1]

        # model prediction on the latest image
        pred, conf, probs = self.classifier.predict_image(last_image)

        probs = np.asarray(probs, dtype=np.float64)
        self.current_probs.append(probs)

        # ---- Bayesian-style accumulation over images ----
        eps = 1e-8
        logp = np.log(probs + eps)

        if self.log_prob_sum is None:
            self.log_prob_sum = logp
        else:
            self.log_prob_sum += logp

        # back to probability simplex
        max_log = np.max(self.log_prob_sum)
        exp_logits = np.exp(self.log_prob_sum - max_log)
        combined_probs = exp_logits / np.sum(exp_logits)

        self.last_combined_probs = combined_probs

        pred_idx = int(np.argmax(combined_probs))
        pred_label = self.classifier.class_names[pred_idx]
        pred_conf = float(combined_probs[pred_idx])

        # Target index and its probability
        target_idx = self.classifier.class_names.index(self.target_label)
        target_conf = float(combined_probs[target_idx])

        # ---- update target streak ----
        if pred_label == self.target_label and target_conf >= self.conf_threshold:
            self.target_streak += 1
        else:
            self.target_streak = 0

        # ---- update skip streak ----
        # candidate to skip if some non-target class is clearly above target
        if pred_label != self.target_label:
            margin = pred_conf - target_conf
            if (pred_conf >= self.conf_threshold) and (margin >= self.skip_margin):
                self.skip_streak += 1
            else:
                self.skip_streak = 0
        else:
            # if current best is target, don't accumulate skip evidence
            self.skip_streak = 0

        print(
            f"[INCREMENTAL] {len(self.current_images)} imgs → {pred_label} ({pred_conf:.2f}) | "
            f"P(target={self.target_label})={target_conf:.2f}, "
            f"target_streak={self.target_streak}, skip_streak={self.skip_streak}"
        )

        # ---- stopping condition (found target) ----
        is_target = (
            len(self.current_images) >= self.min_images
            and self.target_streak >= self.consec_target_required
        )

        return is_target, pred_label, pred_conf

    # ---------------------------------------------------------
    # Separate API for skip decision
    # ---------------------------------------------------------
    def should_skip_object(self):
        """
        Return True if we should skip current object (high confidence non-target).

        Uses fused probabilities + streak logic + minimum number of images.
        """
        if len(self.current_images) < self.min_images:
            return False

        return self.skip_streak >= self.consec_skip_required
'''

# search_manager.py
import numpy as np


class SearchManager:
    """
    Handles incremental tactile classification using Bayesian fusion +
    minimum-probe decision logic for both manual and PPO-based search.
    """

    def __init__(self,
                 classifier,
                 target_label,
                 conf_threshold=0.90,
                 min_probes=3,
                 target_streak_required=2,
                 skip_streak_required=2,
                 skip_margin=0.35):

        self.classifier = classifier
        self.target_label = target_label
        self.conf_threshold = conf_threshold

        # decision parameters
        self.min_probes = min_probes
        self.target_streak_required = target_streak_required
        self.skip_streak_required = skip_streak_required
        self.skip_margin = skip_margin

        # per-object state
        self.reset_state()

    # -----------------------------------------------------------
    def reset_state(self):
        """Reset buffers when switching to a new object."""
        self.image_paths = []
        self.log_prob_sum = None
        self.target_streak = 0
        self.skip_streak = 0
        self.last_probs = None

    # -----------------------------------------------------------
    def set_current_object(self, obj_name: str):
        """Called by controller whenever robot moves to a new object."""
        self.current_object = obj_name
        self.reset_state()

    # -----------------------------------------------------------
    def add_image(self, img_path: str):
        """Store the newly generated tactile image path."""
        self.image_paths.append(img_path)

    # -----------------------------------------------------------
    def classify_incremental(self):
        """
        Perform classifier inference on the most recent image,
        fuse all predictions (Bayesian), and update stopping criteria.

        Returns:
            is_target (bool)
            pred_label (str)
            pred_conf (float)
        """
        if len(self.image_paths) == 0:
            return False, None, 0.0

        # Latest image
        last_img = self.image_paths[-1]

        pred_label, pred_conf, probs = self.classifier.predict_image(last_img)
        probs = np.asarray(probs, dtype=np.float64)

        # =============================
        # 1. Bayesian fusion
        # =============================
        eps = 1e-8
        logp = np.log(probs + eps)

        if self.log_prob_sum is None:
            self.log_prob_sum = logp
        else:
            self.log_prob_sum += logp

        # normalized probabilities
        max_log = np.max(self.log_prob_sum)  # avoids overflow
        unnorm = np.exp(self.log_prob_sum - max_log)
        fused = unnorm / unnorm.sum()

        self.last_probs = fused

        # fused prediction
        pred_idx = int(np.argmax(fused))
        fused_label = self.classifier.class_names[pred_idx]
        fused_conf = float(fused[pred_idx])

        # target probability
        target_idx = self.classifier.class_names.index(self.target_label)
        target_conf = float(fused[target_idx])

        # =============================
        # 2. Update streaks
        # =============================
        # target streak (for confirming)
        if fused_label == self.target_label and target_conf >= self.conf_threshold:
            self.target_streak += 1
        else:
            self.target_streak = 0

        # skip streak (for rejecting)
        if fused_label != self.target_label:
            margin = fused_conf - target_conf
            if fused_conf >= self.conf_threshold and margin >= self.skip_margin:
                self.skip_streak += 1
            else:
                self.skip_streak = 0
        else:
            self.skip_streak = 0

        # Debug print
        print(
            f"[FUSED] {len(self.image_paths)} imgs → {fused_label} ({fused_conf:.2f}) | "
            f"P(target={self.target_label})={target_conf:.2f}, "
            f"target_streak={self.target_streak}, skip_streak={self.skip_streak}"
        )

        # =============================
        # 3. Confirm target?
        # =============================
        enough_data = len(self.image_paths) >= self.min_probes
        is_target = (
            enough_data and self.target_streak >= self.target_streak_required
        )

        return is_target, fused_label, fused_conf

    # -----------------------------------------------------------
    def should_skip_object(self):
        """
        Called *after* classify_incremental().
        Returns True if the system is confident the object is NOT the target.
        """
        if len(self.image_paths) < self.min_probes:
            return False

        return self.skip_streak >= self.skip_streak_required
