import os
import time
import numpy as np
import mujoco
import mujoco.viewer

from utils import get_freejoint_qposadr, CSVLogger
from tactile_image_buffer import TactileImageBuffer


class TactileExplorer:
    """
    Unified tactile controller that supports:
      - Manual fixed-policy search (run_manual)
      - PPO-based single-touch control (perform_single_touch)

    This section only defines:
      - configuration constants
      - object table
      - MuJoCo model loading
      - object indexing helpers
      - basic tactile sensing
    """

    # -------------------------------------------------------------
    # STATIC CONFIG (shared for all modes)
    # -------------------------------------------------------------
    RES = 128
    DAMPING = 0.96
    MAX_CONTACTS_PER_OBJECT = 20

    # Unified contact behaviour for ALL shapes
    GLOBAL_FORCE_THR = 0.0015      # same threshold for all objects
    FORWARD_STEP = 0.0003          # 1 mm per physics step towards sensor
    BACKWARD_STEP = 0.0003         # 1 mm per physics step away from sensor

    # -------------------------------------------------------------
    # OBJECT DEFINITIONS
    # -------------------------------------------------------------
    OBJECTS = [
        {
            "name": "sphere",
            "label": "sphere",
            "BASE_X": 0.10,
            "CONTACT_X": 0.1384,
            "FIX_Z": 0.070,
            "FORWARD_SPD": 1.00,
            "BACK_SPD": 0.90,
            "PAUSE_T": 0.08,
            "ROT_T": 0.15,
            "NUDGE_X": 0.0,
            "FORCE_THR": 0.002,
        },
        {
            "name": "cube",
            "label": "cube",
            "BASE_X": 0.14,
            "CONTACT_X": 0.1472,
            "FIX_Z": 0.070,
            "FORWARD_SPD": 0.40,
            "BACK_SPD": 0.36,
            "PAUSE_T": 0.12,
            "ROT_T": 0.80,
            "NUDGE_X": 0.0005,
            "FORCE_THR": 0.0005,
        },
        {
            "name": "cylinder",
            "label": "cylinder",
            "BASE_X": 0.095,
            "CONTACT_X": 0.1367,
            "FIX_Z": 0.070,
            "FORWARD_SPD": 0.40,
            "BACK_SPD": 0.36,
            "PAUSE_T": 0.10,
            "ROT_T": 0.85,
            "NUDGE_X": 0.0005,
            "FORCE_THR": 0.0005,
        },
        {
            "name": "cone",
            "label": "cone",
            "BASE_X": 0.085,
            "CONTACT_X": 0.1252,
            "FIX_Z": 0.070,
            "FORWARD_SPD": 0.40,
            "BACK_SPD": 0.36,
            "PAUSE_T": 0.10,
            "ROT_T": 0.80,
            "NUDGE_X": 0.0005,
            "FORCE_THR": 0.0005,
        },
    ]

    # -------------------------------------------------------------
    # INIT
    # -------------------------------------------------------------
    def __init__(self, search_manager):
        print("🚀 Tactile Motion Controller initialized.")
        self.search_manager = search_manager

        # -------- Paths --------
        self.SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.MODEL_XML = os.path.join(self.SCRIPT_DIR, "Search_scene.xml")
        self.SEARCH_CSV_DIR = os.path.join(self.SCRIPT_DIR, "search_csv")
        os.makedirs(self.SEARCH_CSV_DIR, exist_ok=True)

        self.csv_path = os.path.join(
            self.SEARCH_CSV_DIR,
            f"tactile_search_raw_{int(time.time())}.csv",
        )

        # -------- MuJoCo model --------
        self.model = mujoco.MjModel.from_xml_path(self.MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.viewer = None   # NEW: handle to a MuJoCo viewer (used by PPO/inference)

        mujoco.mj_forward(self.model, self.data)

        # -------- Logging --------
        self.logger = CSVLogger(self.csv_path)

        # -------- Object indexing + parking --------
        self._prepare_objects()

        # Current active object (used by both manual + PPO)
        self.active_obj = None      # dict from OBJECTS
        self.x_idx = None
        self.y_idx = None
        self.z_idx = None
        self.quat_sl = None
        self.cfg = None             # convenience alias for active_obj config

        # Real-time pseudo tactile image buffer
        self.img_buffer = TactileImageBuffer(
            shape_name=None,   # updated when switching objects
            save_root="dataset"
        )

    # -------------------------------------------------------------
    # Object initialization
    # -------------------------------------------------------------
    def _prepare_objects(self):
        """
        Precompute qpos indices for each freejoint object and park them.
        """
        for obj in self.OBJECTS:
            q = get_freejoint_qposadr(self.model, obj["name"])
            obj["x_idx"] = q
            obj["y_idx"] = q + 1
            obj["z_idx"] = q + 2
            obj["quat_slice"] = slice(q + 3, q + 7)

            # Park all objects far away initially
            self.data.qpos[q:q + 3] = [0.3, 0.0, -0.2]
            self.data.qpos[q + 3:q + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

    def _park_all_objects(self):
        """
        Move all objects away from the sensor (shared helper).
        """
        for obj in self.OBJECTS:
            qi = obj["x_idx"]
            qs = obj["quat_slice"]
            self.data.qpos[qi:qi + 3] = [0.3, 0.0, -0.2]
            self.data.qpos[qs] = [1, 0, 0, 0]

    def _set_active_object(self, label):
        """
        Select which object is currently being used (by label:
        'sphere' / 'cube' / 'cylinder' / 'cone').
        Sets indices, base pose, and resets SearchManager for this object.
        """
        active = None
        for obj in self.OBJECTS:
            if obj["label"] == label:
                active = obj
                break

        if active is None:
            raise ValueError(f"[TactileExplorer] Unknown object label: {label}")

        self.active_obj = active
        self.cfg = active
        self.x_idx = active["x_idx"]
        self.y_idx = active["y_idx"]
        self.z_idx = active["z_idx"]
        self.quat_sl = active["quat_slice"]

        # Put active object at its BASE pose
        self.data.qpos[self.x_idx:self.x_idx + 3] = [
            active["BASE_X"], 0.0, active["FIX_Z"]
        ]
        # Default orientation = 90° around Y (this is what you had originally)
        q0 = np.zeros(4)
        mujoco.mju_euler2Quat(q0, [0, np.pi / 2, 0], "xyz")
        self.data.qpos[self.quat_sl] = q0

        mujoco.mj_forward(self.model, self.data)

        # Update buffer shape name for saving images
        if hasattr(self, "img_buffer") and self.img_buffer is not None:
            self.img_buffer.set_shape(active["label"])

        # NEW: reset SearchManager state for this object
        if hasattr(self, "search_manager") and self.search_manager is not None:
            if hasattr(self.search_manager, "set_current_object"):
                self.search_manager.set_current_object(active["label"])



    # -------------------------------------------------------------
    # Sensor reading
    # -------------------------------------------------------------
    def tactile_force(self):
        """
        Returns scalar magnitude of 3-axis force sensor plus touch scalar (if present).
        """
        fx, fy, fz = self.data.sensordata[0:3]
        touch = self.data.sensordata[3] if self.model.nsensor > 1 else 0.0
        return np.sqrt(fx * fx + fy * fy + fz * fz) + touch

    # -------------------------------------------------------------
    # MANUAL SEARCH MODE  (fixed policy, same behavior as original)
    # -------------------------------------------------------------
    def run_manual(self):
        """
        Full multi-object search:
            sphere → cube → cylinder → cone
        using your existing fixed exploration sequence.

        This replicates your original behavior, but uses the new
        shared object-management and pose-management structure.
        """

        print(f"📁 Logging CSV to: {self.csv_path}")

        # Start with first object (sphere)
        object_order = ["sphere", "cube", "cylinder", "cone"]
        obj_idx = 0

        # Load first object
        self._park_all_objects()
        self._set_active_object(object_order[obj_idx])

        # -------------------------------------------------------------------
        # Manual exploration state
        # -------------------------------------------------------------------
        phase = "forward"
        contact_idx = 0
        start_time = time.time()
        pause_start = None
        rotate_start = None
        contact_x = None

        # Your 4 fixed exploration orientations
        quat_face   = np.zeros(4)
        quat_edge   = np.zeros(4)
        quat_corner = np.zeros(4)
        quat_side   = np.zeros(4)

        mujoco.mju_euler2Quat(quat_face,   [0.0,       0.0,        0.0],       "xyz")
        mujoco.mju_euler2Quat(quat_edge,   [0.0,       0.0,        np.pi/4],  "xyz")
        mujoco.mju_euler2Quat(quat_corner, [np.pi/4.0, 0.0,        np.pi/4],  "xyz")
        mujoco.mju_euler2Quat(quat_side,   [0.0,       np.pi/2,    0.0],       "xyz")

        EXPLORATION = [
            {"quat": quat_face,   "dy":  0.000, "dz":  0.000},
            {"quat": quat_edge,   "dy": +0.010, "dz":  0.000},
            {"quat": quat_corner, "dy": -0.010, "dz":  0.000},
            {"quat": quat_side,   "dy":  0.000, "dz": +0.005},
        ]

        # -------------------------------------------------------------------
        # Viewer loop (unchanged behavior)
        # -------------------------------------------------------------------
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:

            while viewer.is_running():

                t_now = time.time() - start_time
                f_val = self.tactile_force()

                # Enforce object height
                self.data.qpos[self.z_idx] = self.cfg["FIX_Z"]

                # Park inactive objects
                for obj in self.OBJECTS:
                    if obj["label"] != self.active_obj["label"]:
                        qi = obj["x_idx"]
                        qs = obj["quat_slice"]
                        self.data.qpos[qi:qi+3] = [0.3, 0.0, -0.2]
                        self.data.qpos[qs] = [1, 0, 0, 0]

                x = float(self.data.qpos[self.x_idx])
                y = float(self.data.qpos[self.y_idx])
                z = float(self.data.qpos[self.z_idx])

                # =============================================================
                # PHASE: FORWARD
                # =============================================================
                if phase == "forward":
                    self.data.qpos[self.x_idx] -= self.FORWARD_STEP

                    # Detect first contact
                    if f_val > self.GLOBAL_FORCE_THR:
                        contact_x = float(self.data.qpos[self.x_idx])
                        self.data.qpos[self.x_idx] = contact_x
                        phase = "pause"
                        pause_start = time.time()

                # =============================================================
                # PHASE: PAUSE
                # =============================================================
                elif phase == "pause":
                    self.data.qpos[self.x_idx] = contact_x

                    # Tiny wobble
                    self.data.qpos[self.x_idx] -= 0.0002
                    self.data.qpos[self.z_idx] = self.cfg["FIX_Z"] - 0.001 * np.sin(time.time() * 40)

                    if time.time() - pause_start > self.cfg["PAUSE_T"]:
                        phase = "backward"

                # =============================================================
                # PHASE: BACKWARD
                # =============================================================
                elif phase == "backward":
                    self.data.qpos[self.x_idx] += self.BACKWARD_STEP
                    if self.data.qpos[self.x_idx] >= self.cfg["BASE_X"]:
                        self.data.qpos[self.x_idx] = self.cfg["BASE_X"]
                        phase = "rotate"
                        rotate_start = time.time()

                # =============================================================
                # PHASE: ROTATE (fixed 4-pose loop)
                # =============================================================
                elif phase == "rotate":
                    pose = EXPLORATION[contact_idx % 4]

                    self.data.qpos[self.quat_sl] = pose["quat"]
                    self.data.qpos[self.y_idx]   = pose["dy"]
                    self.data.qpos[self.z_idx]   = self.cfg["FIX_Z"] + pose["dz"]

                    # hold object slightly in front
                    self.data.qpos[self.x_idx] = self.cfg["CONTACT_X"] + 0.003

                    # random drift
                    self.data.qpos[self.y_idx] += 0.003 * np.sin(contact_idx)
                    self.data.qpos[self.x_idx] -= 0.002 * np.cos(contact_idx)

                    if time.time() - rotate_start > self.cfg["ROT_T"]:
                        contact_idx += 1
                        phase = "forward"
                        contact_x = None

                # =============================================================
                # DO PHYSICS STEP + VIEW
                # =============================================================
                mujoco.mj_step(self.model, self.data)
                self.data.qvel[:] *= self.DAMPING
                viewer.sync()
                time.sleep(self.model.opt.timestep)

                # =============================================================
                # LOGGING + TACTILE IMAGE GENERATION
                # =============================================================
                self.logger.log(t_now, self.active_obj["label"], x, y, z, f_val, phase, contact_idx)

                # Only generate image during pause & actual contact
                if phase == "pause" and f_val > 0:
                    img_path = self.img_buffer.add_reading({
                        "x": x,
                        "y": y,
                        "force": f_val,
                        "phase": phase
                    })
                    if img_path:
                        print("Generated tactile image:", img_path)

                        # Feed to SearchManager
                        self.search_manager.add_image(img_path)

                        # Incremental classification
                        is_target, pred_label, pred_conf = self.search_manager.classify_incremental()

                        if is_target:
                            print(f"\n🎯 EARLY STOP: target FOUND → {pred_label} (conf={pred_conf:.2f})")
                            return

                        # Skip object if high confidence not target
                        if (pred_conf >= self.search_manager.conf_threshold and
                                pred_label != self.search_manager.target_label):
                            print(f"\n⏭️ SKIP OBJECT: model highly confident it's NOT the target → {pred_label} ({pred_conf:.2f})")
                            contact_idx = self.MAX_CONTACTS_PER_OBJECT

                # =============================================================
                # SWITCH OBJECT
                # =============================================================
                if contact_idx >= self.MAX_CONTACTS_PER_OBJECT and phase == "forward":
                    obj_idx += 1
                    if obj_idx >= len(object_order):
                        print("\n❌ Completed all objects — target NOT FOUND.")
                        print(f"[INFO] CSV saved to {self.csv_path}")
                        return

                    # Change object
                    next_label = object_order[obj_idx]

                    # Park everything and set new active object
                    self._park_all_objects()
                    self._set_active_object(next_label)
                    # _set_active_object now:
                    #   - moves object
                    #   - sets img_buffer shape
                    #   - resets SearchManager state

                    # Reset motion state
                    contact_idx = 0
                    pause_start = None
                    rotate_start = None
                    contact_x = None
                    phase = "forward"

                    # Reset timing
                    start_time = time.time()


        print(f"\n❌ Completed all objects — target NOT FOUND.")
        print(f"[INFO] CSV saved to {self.csv_path}")

    # -------------------------------------------------------------
    # PPO MODE: single-touch API
    # -------------------------------------------------------------
    def perform_single_touch(self, pose_idx: int):
        """
        Execute ONE tactile contact using the given pose index.
        Used by PPO: one call = one action = one touch.

        Returns:
            img_path : str or None   (None if no meaningful contact)
            force_val : float        (last measured force during pause)
            (x, y) : tuple(float,float)  contact position at the end
        """
        # Determine which object is currently being explored.
        # Prefer SearchManager's current_object_label if set,
        # otherwise fall back to the target label.
        label = getattr(self.search_manager, "current_object_label", None)
        if label is None:
            label = self.search_manager.target_label

        # Ensure the correct object is active in the scene
        if self.active_obj is None or self.active_obj["label"] != label:
            self._park_all_objects()
            self._set_active_object(label)

        # Build the shared 6 exploration poses
        poses = self._build_exploration_poses()
        pose = poses[int(pose_idx) % len(poses)]

        # Apply orientation + small offsets
        self.data.qpos[self.quat_sl] = pose["quat"]
        self.data.qpos[self.y_idx]   = pose["dy"]
        self.data.qpos[self.z_idx]   = self.cfg["FIX_Z"] + pose["dz"]

        # Hold slightly in front of sensor
        self.data.qpos[self.x_idx] = self.cfg["CONTACT_X"] + 0.003
        mujoco.mj_forward(self.model, self.data)

        # Execute ONE forward→pause→backward cycle
        img_path, force_val, pos_xy = self._single_contact_cycle()

        return img_path, force_val, pos_xy


    def _build_exploration_poses(self):
        """
        Define the 6 motion primitives that BOTH:
          - manual mode (if you want to reuse later), and
          - PPO mode
        can share.

        These correspond to:
            0: top-down
            1: edge-aligned
            2: corner-aligned
            3: side-swipe
            4: diagonal
            5: reverse-angle
        """
        quat_face   = np.zeros(4)
        quat_edge   = np.zeros(4)
        quat_corner = np.zeros(4)
        quat_side   = np.zeros(4)
        quat_diag   = np.zeros(4)
        quat_rev    = np.zeros(4)

        mujoco.mju_euler2Quat(quat_face,   [0.0,       0.0,        0.0],        "xyz")
        mujoco.mju_euler2Quat(quat_edge,   [0.0,       0.0,        np.pi/4],    "xyz")
        mujoco.mju_euler2Quat(quat_corner, [np.pi/4,   0.0,        np.pi/6],    "xyz")
        mujoco.mju_euler2Quat(quat_side,   [0.0,       np.pi/2,    0.0],        "xyz")
        mujoco.mju_euler2Quat(quat_diag,   [np.pi/6,   np.pi/6,    np.pi/6],    "xyz")
        mujoco.mju_euler2Quat(quat_rev,    [-np.pi/6,  -np.pi/6,   0.0],        "xyz")

        poses = [
            {"quat": quat_face,   "dy":  0.000,  "dz":  0.000},   # 0 top-down
            {"quat": quat_edge,   "dy": +0.010,  "dz":  0.000},   # 1 edge-aligned
            {"quat": quat_corner, "dy": -0.010,  "dz":  0.000},   # 2 corner-aligned
            {"quat": quat_side,   "dy":  0.000,  "dz": +0.005},   # 3 side-swipe
            {"quat": quat_diag,   "dy": +0.005,  "dz": -0.005},   # 4 diagonal
            {"quat": quat_rev,    "dy": -0.005,  "dz": +0.005},   # 5 reverse-angle
        ]
        return poses


    def _single_contact_cycle(self):
        """
        ONE full contact cycle for the currently active object:

            1) Move forward until first contact (force > GLOBAL_FORCE_THR)
            2) Pause at fixed contact_x, wobble slightly, generate ONE tactile image
            3) Move backward to BASE_X

        If self.viewer is None: runs headless (used for PPO training).
        If self.viewer is set: each mj_step calls viewer.sync() so motion is visible.
        """
        contact_x = None
        img_path = None
        force_val = 0.0

        # NEW: optional viewer handle
        viewer = getattr(self, "viewer", None)

        # ----------------------- FORWARD -----------------------
        for _ in range(400):  # forward movement steps
            # keep Z fixed
            self.data.qpos[self.z_idx] = self.cfg["FIX_Z"]
            # step towards sensor
            self.data.qpos[self.x_idx] -= self.FORWARD_STEP

            mujoco.mj_step(self.model, self.data)
            self.data.qvel[:] *= self.DAMPING

            # NEW: update viewer if attached
            if viewer is not None:
                viewer.sync()
                time.sleep(self.model.opt.timestep)

            force_val = self.tactile_force()
            if force_val > self.GLOBAL_FORCE_THR:
                contact_x = float(self.data.qpos[self.x_idx])
                # clamp to contact point
                self.data.qpos[self.x_idx] = contact_x
                mujoco.mj_forward(self.model, self.data)
                break

        # No contact → return "no image"
        if contact_x is None:
            x = float(self.data.qpos[self.x_idx])
            y = float(self.data.qpos[self.y_idx])
            return None, 0.0, (x, y)

        # ----------------------- PAUSE -------------------------
        pause_start = time.time()
        while time.time() - pause_start < self.cfg["PAUSE_T"]:
            # lock at contact_x, add small wobble in Z for richer pressure
            self.data.qpos[self.x_idx] = contact_x
            self.data.qpos[self.z_idx] = (
                self.cfg["FIX_Z"] - 0.001 * np.sin(time.time() * 40)
            )

            mujoco.mj_step(self.model, self.data)
            self.data.qvel[:] *= self.DAMPING

            if viewer is not None:
                viewer.sync()
                time.sleep(self.model.opt.timestep)

            force_val = self.tactile_force()

            # Generate tactile image once during pause
            if img_path is None and force_val > 0:
                x = float(self.data.qpos[self.x_idx])
                y = float(self.data.qpos[self.y_idx])

                img_path = self.img_buffer.add_reading(
                    {
                        "x": x,
                        "y": y,
                        "force": force_val,
                        "phase": "pause",
                    }
                )

        # ----------------------- BACKWARD ----------------------
        for _ in range(400):
            self.data.qpos[self.z_idx] = self.cfg["FIX_Z"]
            self.data.qpos[self.x_idx] += self.BACKWARD_STEP

            mujoco.mj_step(self.model, self.data)
            self.data.qvel[:] *= self.DAMPING

            if viewer is not None:
                viewer.sync()
                time.sleep(self.model.opt.timestep)

            if self.data.qpos[self.x_idx] >= self.cfg["BASE_X"]:
                self.data.qpos[self.x_idx] = self.cfg["BASE_X"]
                break

        # Final contact position (after backward)
        x_final = float(self.data.qpos[self.x_idx])
        y_final = float(self.data.qpos[self.y_idx])

        # Might still be None if buffer refused to save
        return img_path, force_val, (x_final, y_final)
