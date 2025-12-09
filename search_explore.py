import os
import time
import csv

import numpy as np
import mujoco
import mujoco.viewer

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_XML = os.path.join(SCRIPT_DIR, "Finger.xml")
SEARCH_CSV_DIR = os.path.join(SCRIPT_DIR, "search_csv")

RES = 128
DAMPING = 0.96
MAX_CONTACTS_PER_OBJECT = 20

# -------------------------------------------------------------
# OBJECT DEFINITIONS + MOTION PARAMS
# -------------------------------------------------------------
OBJECTS = [
    {
        "name": "sphere",
        "label": "sphere",
        "BASE_X": 0.1,
        "CONTACT_X": 0.056,
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
        "CONTACT_X": 0.105,
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
        "CONTACT_X": 0.052,
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
        "CONTACT_X": 0.058,
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
# HELPERS
# -------------------------------------------------------------
def get_freejoint_qposadr(model, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jadr = model.body_jntadr[body_id]
    for i in range(model.body_jntnum[body_id]):
        jid = jadr + i
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[jid]
    raise ValueError(f"Body '{body_name}' has no freejoint.")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    print("🚀 Starting multi-object tactile exploration…")

    mj_model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, data)

    # CSV output
    os.makedirs(SEARCH_CSV_DIR, exist_ok=True)
    csv_path = os.path.join(SEARCH_CSV_DIR, f"tactile_search_raw_{int(time.time())}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["time", "object", "x", "y", "z", "force", "phase", "contact_idx"]
        )

    # ---------------------------------------------------------
    # Initialize freejoint addresses + PARK ALL OBJECTS
    # ---------------------------------------------------------
    for obj in OBJECTS:
        q = get_freejoint_qposadr(mj_model, obj["name"])
        obj["x_idx"] = q
        obj["y_idx"] = q + 1
        obj["z_idx"] = q + 2
        obj["quat_slice"] = slice(q + 3, q + 7)

        data.qpos[q:q+3] = [0.3, 0.0, -0.2]
        data.qpos[q+3:q+7] = [1, 0, 0, 0]

    mujoco.mj_forward(mj_model, data)

    # ---------------------------------------------------------
    # Activate first object
    # ---------------------------------------------------------
    current_idx = 0
    active = OBJECTS[current_idx]
    cfg = active

    x_idx, y_idx, z_idx = active["x_idx"], active["y_idx"], active["z_idx"]
    quat_sl = active["quat_slice"]

    data.qpos[x_idx:x_idx+3] = [cfg["BASE_X"], 0.0, cfg["FIX_Z"]]

    if active["name"] == "sphere":
        data.qpos[quat_sl] = [1, 0, 0, 0]
    else:
        q0 = np.zeros(4)
        mujoco.mju_euler2Quat(q0, [0, np.pi/2, 0], "xyz")
        data.qpos[quat_sl] = q0

    mujoco.mj_forward(mj_model, data)

    # ---------------------------------------------------------
    # State variables
    # ---------------------------------------------------------
    rot_angle = 0.0
    rot_axis_cycle = 0.0

    phase = "forward"
    start_time = time.time()
    pause_start = 0
    rotate_start = 0
    contact_idx = 0
    contact_x = None

    # ---------------------------------------------------------
    # FORCE READOUT
    # ---------------------------------------------------------
    def tactile_force():
        fx, fy, fz = data.sensordata[0:3]
        touch = data.sensordata[3] if mj_model.nsensor > 1 else 0.0
        return np.sqrt(fx*fx + fy*fy + fz*fz) + touch

    def maintain_height():
        data.qpos[z_idx] = cfg["FIX_Z"]

    def park_inactive():
        for obj in OBJECTS:
            if obj is active:
                continue
            qi = obj["x_idx"]
            qs = obj["quat_slice"]
            data.qpos[qi:qi+3] = [0.3, 0.0, -0.2]
            data.qpos[qs] = [1, 0, 0, 0]

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------
    with mujoco.viewer.launch_passive(mj_model, data) as viewer:

        while viewer.is_running():
            t = time.time() - start_time
            f_val = tactile_force()

            maintain_height()
            park_inactive()

            x = float(data.qpos[x_idx])
            y = float(data.qpos[y_idx])
            z = float(data.qpos[z_idx])

            # -----------------------------------------------------
            # FORWARD
            # -----------------------------------------------------
            if phase == "forward":
                if active["name"] == "sphere":
                    data.qvel[x_idx] = -cfg["FORWARD_SPD"]
                else:
                    data.qpos[x_idx] -= cfg["FORWARD_SPD"] * mj_model.opt.timestep

                if f_val > cfg["FORCE_THR"]:
                    data.qvel[x_idx] = 0
                    contact_x = x
                    phase = "pause"
                    pause_start = time.time()

            # -----------------------------------------------------
            # PAUSE  (ONLY CONTACT IMPROVED — NOTHING ELSE CHANGED)
            # -----------------------------------------------------
            elif phase == "pause":
                data.qvel[x_idx] = 0

                if contact_x is not None:
                    data.qpos[x_idx] = contact_x

                # ⭐ ONLY MODIFICATION YOU ASKED FOR ⭐
                # Soft push deeper → cylinder + cone get stable force
                data.qpos[x_idx] -= cfg["NUDGE_X"] * 2.2
                data.qpos[z_idx] = cfg["FIX_Z"] - 0.001 * np.sin(time.time() * 40)

                if time.time() - pause_start > cfg["PAUSE_T"]:
                    phase = "backward"

            # -----------------------------------------------------
            # BACKWARD
            # -----------------------------------------------------
            elif phase == "backward":
                if active["name"] == "sphere":
                    data.qvel[x_idx] = +cfg["BACK_SPD"]
                else:
                    data.qpos[x_idx] += cfg["BACK_SPD"] * mj_model.opt.timestep

                if data.qpos[x_idx] >= cfg["BASE_X"]:
                    data.qvel[x_idx] = 0
                    phase = "rotate"
                    rotate_start = time.time()

            # -----------------------------------------------------
            # ROTATION  (UNCHANGED!!! ORIGINAL LOGIC)
            # -----------------------------------------------------
            elif phase == "rotate":
                f_val = 0.0

                rot_axis_cycle += mj_model.opt.timestep * (np.pi/4)
                rot_angle += mj_model.opt.timestep * (np.pi/3)

                ax = np.array([
                    0.6 + 0.4*np.sin(rot_axis_cycle),
                    0.4 + 0.6*np.cos(0.7*rot_axis_cycle),
                    0.2 + 0.8*np.sin(0.5*rot_axis_cycle),
                ])
                ax /= np.linalg.norm(ax)

                quat = np.zeros(4)
                mujoco.mju_axisAngle2Quat(quat, ax, rot_angle)
                data.qpos[quat_sl] = quat

                # wobble motion unchanged
                data.qpos[x_idx] = cfg["BASE_X"] + 0.005
                data.qpos[y_idx] = 0.02 * np.sin(rot_axis_cycle)
                data.qpos[z_idx] = cfg["FIX_Z"] + 0.005 * np.cos(rot_axis_cycle)

                if time.time() - rotate_start > cfg["ROT_T"]:
                    contact_idx += 1
                    phase = "forward"
                    pause_start = time.time()
                    contact_x = None

                    if contact_idx >= MAX_CONTACTS_PER_OBJECT:
                        data.qpos[active["x_idx"]:active["x_idx"]+3] = [0.3,0, -0.2]
                        data.qpos[active["quat_slice"]] = [1,0,0,0]

                        current_idx = (current_idx + 1) % len(OBJECTS)
                        active = OBJECTS[current_idx]
                        cfg = active

                        x_idx = active["x_idx"]
                        y_idx = active["y_idx"]
                        z_idx = active["z_idx"]
                        quat_sl = active["quat_slice"]

                        data.qpos[x_idx:x_idx+3] = [cfg["BASE_X"],0,cfg["FIX_Z"]]

                        if active["name"] == "sphere":
                            data.qpos[quat_sl] = [1,0,0,0]
                        else:
                            q0 = np.zeros(4)
                            mujoco.mju_euler2Quat(q0,[0,np.pi/2,0],"xyz")
                            data.qpos[quat_sl] = q0

                        mujoco.mj_forward(mj_model, data)

                        rot_angle = 0.0
                        rot_axis_cycle = 0.0
                        contact_idx = 0
                        contact_x = None
                        phase = "forward"
                        pause_start = time.time()

            # -----------------------------------------------------
            # PHYSICS STEP
            # -----------------------------------------------------
            mujoco.mj_step(mj_model, data)
            data.qvel[:] *= DAMPING
            viewer.sync()
            time.sleep(mj_model.opt.timestep)

            # -----------------------------------------------------
            # LOGGING
            # -----------------------------------------------------
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [t, active["label"], x, y, z, f_val, phase, contact_idx]
                )

    print(f"[INFO] CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
