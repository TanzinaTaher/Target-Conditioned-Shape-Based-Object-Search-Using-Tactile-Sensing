import mujoco
import mujoco.viewer
import numpy as np
import os, time, csv

# === CONFIGURATION ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_XML  = os.path.join(SCRIPT_DIR, "scene_cylinder.xml")
BODY_NAME  = "cylinder"

# --- Motion constants ---
BASE_X       = 0.095
CONTACT_X    = 0.052
FIX_Z        = 0.070
FORWARD_SPD  = 0.40
BACK_SPD     = 0.36
PAUSE_T      = 0.10
ROT_T        = 0.25
NUDGE_X      = 0.0005
FORCE_THR    = 0.0005
DAMPING      = 0.96

def get_freejoint_qposadr(model, body_name):
    """Return qpos address of the freejoint attached to a given body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jadr = model.body_jntadr[bid]
    for i in range(model.body_jntnum[bid]):
        jid = jadr + i
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[jid]
    raise ValueError(f"Body '{body_name}' has no freejoint.")

def main():
    print("🚀 Starting tactile cylinder simulation (stable contact + 3D logging)")
    print(f"📂 Loading model from: {MODEL_XML}")

    if not os.path.exists(MODEL_XML):
        raise FileNotFoundError(f"❌ XML file not found: {MODEL_XML}")

    # --- Load model ---
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # --- Dataset setup ---
    save_dir = os.path.join(SCRIPT_DIR, "dataset", "cylinder")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"tactile_data_{int(time.time())}.csv")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["time", "x_position", "y_position", "z_position", "force", "phase"])

    # --- Locate freejoint ---
    qposadr = get_freejoint_qposadr(model, BODY_NAME)
    x_idx   = qposadr
    quat_sl = slice(qposadr + 3, qposadr + 7)

    # --- Initialize pose ---
    data.qpos[x_idx:x_idx+3] = [BASE_X, 0.0, FIX_Z]
    mujoco.mju_euler2Quat(data.qpos[quat_sl], np.array([0.0, np.pi/2, 0.0]), "xyz")
    mujoco.mj_forward(model, data)

    # --- State ---
    phase = "forward"
    start = time.time()
    pause_start = rotate_start = 0.0
    rot_angle = rot_axis_cycle = 0.0

    # --- Force computation ---
    def tactile_force():
        if model.nsensor < 1:
            return 0.0
        fx, fy, fz = data.sensordata[0:3]
        touch_val = data.sensordata[3] if model.nsensor > 1 else 0.0
        total_force = np.sqrt(fx**2 + fy**2 + fz**2)
        return total_force + touch_val

    # --- Maintain fixed Z height ---
    def maintain_height():
        data.qpos[x_idx+2] = FIX_Z
        if data.qpos[x_idx+2] < FIX_Z:
            data.qvel[x_idx+2] = 0.0

    # --- Simulation loop ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - start
            f_val = tactile_force()

            maintain_height()

            # --- Motion logic ---
            if phase == "forward":
                data.qvel[x_idx] = -FORWARD_SPD
                if f_val > FORCE_THR or data.qpos[x_idx] <= CONTACT_X:
                    data.qvel[x_idx] = 0.0
                    phase = "pause"
                    pause_start = time.time()

            elif phase == "pause":
                # Maintain gentle pressure during contact
                if f_val < FORCE_THR:
                    data.qpos[x_idx] -= NUDGE_X * 3
                else:
                    data.qvel[x_idx] = 0.0
                if time.time() - pause_start > PAUSE_T:
                    phase = "backward"

            elif phase == "backward":
                data.qvel[x_idx] = +BACK_SPD
                if data.qpos[x_idx] >= BASE_X:
                    data.qvel[x_idx] = 0.0
                    phase = "rotate"
                    rotate_start = time.time()

            elif phase == "rotate":
                rot_axis_cycle += model.opt.timestep * (np.pi / 4)
                ax = np.array([
                    0.6 + 0.4*np.sin(rot_axis_cycle),
                    0.4 + 0.6*np.cos(0.7*rot_axis_cycle),
                    0.2 + 0.8*np.sin(0.5*rot_axis_cycle)
                ])
                ax /= np.linalg.norm(ax)
                rot_angle += model.opt.timestep * (np.pi / 3)
                quat = np.zeros(4)
                mujoco.mju_axisAngle2Quat(quat, ax, rot_angle)
                data.qpos[quat_sl] = quat

                # Small Y oscillation during rotation
                data.qpos[x_idx+1] = 0.01 * np.sin(rot_axis_cycle * 0.5)

                if time.time() - rotate_start > ROT_T:
                    phase = "forward"

            # --- Physics ---
            mujoco.mj_step(model, data)
            data.qvel[:] *= DAMPING
            viewer.sync()
            time.sleep(model.opt.timestep)

            # --- Log tactile data ---
            x, y, z = data.qpos[x_idx:x_idx+3]
            with open(csv_path, "a", newline="") as fcsv:
                csv.writer(fcsv).writerow([t, x, y, z, f_val, phase])

    print(f"✅ Tactile cylinder data saved to: {csv_path}")

if __name__ == "__main__":
    main()
