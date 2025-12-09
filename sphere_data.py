import mujoco
import mujoco.viewer
import numpy as np
import os, time, csv

# === File and model setup ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_XML = os.path.join(SCRIPT_DIR, "scene_sphere.xml")   # must match file name
SPHERE_BODY = "sphere"

def get_freejoint_qposadr(model, body_name):
    """Return qpos address of the freejoint attached to a given body."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found.")
    jadr = model.body_jntadr[body_id]
    for i in range(model.body_jntnum[body_id]):
        jid = jadr + i
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[jid]
    raise ValueError(f"Body '{body_name}' has no freejoint.")

def main():
    print("🚀 Starting tactile simulation (sphere ↔ pad cycles)")
    print(f"📂 Loading model from: {MODEL_XML}")

    # --- Load model safely ---
    if not os.path.exists(MODEL_XML):
        raise FileNotFoundError(f"❌ XML file not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # --- Dataset setup ---
    save_dir = os.path.join(SCRIPT_DIR, "dataset", "sphere")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"tactile_data_{int(time.time())}.csv")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["time", "x_position", "y_position", "z_position", "force", "phase"]
        )

    # --- Locate sphere joint ---
    qposadr = get_freejoint_qposadr(model, SPHERE_BODY)
    x_idx, quat_slice = qposadr, slice(qposadr + 3, qposadr + 7)

    # --- Initial position ---
    data.qpos[x_idx:x_idx+3] = [0.095, 0.0, 0.07]
    mujoco.mj_forward(model, data)

    # --- Motion parameters ---
    base_x, contact_x = 0.095, 0.056
    forward_speed, backward_speed = 1.0, 0.9
    yaw_speed = 0.05
    pause_time, rotate_time = 0.08, 0.15
    damping = 0.96
    phase = "forward"
    yaw = 0.0
    start_time = time.time()

    # --- Tactile force helper ---
    def tactile_force():
        return float(np.sum(data.sensordata)) if model.nsensor > 0 else 0.0

    # --- Run simulation in viewer ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("👀 Running: forward ↔ backward + rotation cycles")

        pause_start = rotate_start = 0
        while viewer.is_running():
            t = time.time() - start_time
            f_val = tactile_force()
            x_pos, y_pos, z_pos = data.qpos[x_idx:x_idx+3]

            # --- Motion logic ---
            if phase == "forward":
                data.qvel[x_idx] = -forward_speed
                if f_val > 0.002 or data.qpos[x_idx] <= contact_x:
                    data.qvel[x_idx] = 0.0
                    phase = "pause"
                    pause_start = time.time()

            elif phase == "pause":
                data.qvel[x_idx] = 0.0
                if time.time() - pause_start > pause_time:
                    phase = "backward"

            elif phase == "backward":
                data.qvel[x_idx] = +backward_speed
                if data.qpos[x_idx] >= base_x:
                    data.qvel[x_idx] = 0.0
                    phase = "rotate"
                    rotate_start = time.time()

            elif phase == "rotate":
                data.qvel[x_idx] = 0.0
                yaw += yaw_speed
                quat = np.zeros(4)
                mujoco.mju_axisAngle2Quat(quat, np.array([0, 0, 1]), yaw)
                data.qpos[quat_slice] = quat
                if time.time() - rotate_start > rotate_time:
                    phase = "forward"

            # --- Physics step ---
            mujoco.mj_step(model, data)
            data.qvel[:] *= damping
            viewer.sync()

            # --- Real-time pacing ---
            time.sleep(model.opt.timestep)

            # --- Log tactile data ---
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([t, x_pos, y_pos, z_pos, f_val, phase])

    print(f"✅ Tactile data saved to: {csv_path}")

if __name__ == "__main__":
    main()
