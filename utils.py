import mujoco
import numpy as np
import csv
import os


def get_freejoint_qposadr(model, body_name: str) -> int:
    """Find the qpos address of a body's freejoint (verbatim from original logic)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jadr = model.body_jntadr[body_id]
    for i in range(model.body_jntnum[body_id]):
        jid = jadr + i
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[jid]

    raise ValueError(f"Body '{body_name}' has no freejoint.")


class CSVLogger:
    """Handles CSV logging exactly as your original script does."""

    def __init__(self, csv_path):
        self.csv_path = csv_path

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["time", "object", "x", "y", "z", "force", "phase", "contact_idx"]
            )

    def log(self, t, obj_label, x, y, z, f_val, phase, contact_idx):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [t, obj_label, x, y, z, f_val, phase, contact_idx]
            )
