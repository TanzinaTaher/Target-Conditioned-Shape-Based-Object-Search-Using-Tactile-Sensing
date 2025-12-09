import os
import time
import csv

import numpy as np
import cv2
import mujoco
import mujoco.viewer

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_XML = os.path.join(SCRIPT_DIR, "Search_scene.xml")
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
CLASSIFIER_PATH = os.path.join(SCRIPT_DIR, "best_tactile_classifier.pth")

SEARCH_CSV_DIR = os.path.join(SCRIPT_DIR, "search_csv")

RES = 128
BLUR_KERNEL = (9, 9)
BLUR_SIGMA = 2.0
WINDOW_SIZE = 50
STRIDE = 25
IMG_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

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
        "COMPLEX_ROT": False,
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
        "NUDGE_X": 0.0012,       # ← increased nudge (was 0.0005)
        "FORCE_THR": 0.00040,    # ← slightly lower threshold
        "COMPLEX_ROT": True,
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
        "NUDGE_X": 0.0012,       # ← increased
        "FORCE_THR": 0.00040,    # ← lower threshold
        "COMPLEX_ROT": True,
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
        "NUDGE_X": 0.0014,       # ← highest nudge (cone is hardest)
        "FORCE_THR": 0.00035,    # ← slightly lower threshold
        "COMPLEX_ROT": True,
    },
]

DAMPING = 0.96
MAX_CONTACTS_PER_OBJECT = 20

# -------------------------------------------------------------
# USER INPUT: GOAL OBJECT
# -------------------------------------------------------------
GOAL = input("Enter goal object (sphere, cube, cylinder, cone): ").strip().lower()
if GOAL not in ["sphere", "cube", "cylinder", "cone"]:
    raise ValueError(f"Invalid GOAL '{GOAL}'")
print(f"🎯 GOAL OBJECT SET TO: {GOAL.upper()}")

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

def window_to_heatmap(window_rows):
    import pandas as pd
    df = pd.DataFrame(window_rows)

    xs = df["x_position"].values
    ys = df["y_position"].values
    fs = df["force"].values

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    pressure = np.zeros((RES, RES), dtype=float)

    if abs(xmax - xmin) > 1e-6 and abs(ymax - ymin) > 1e-6:
        u = ((xs - xmin) / (xmax - xmin) * (RES - 1)).astype(int)
        v = ((ys - ymin) / (ymax - ymin) * (RES - 1)).astype(int)
        for xx, yy, f in zip(u, v, fs):
            pressure[yy, xx] += f
    else:
        center = (RES // 2, RES // 2)
        mean_force = float(df["force"].mean())
        radius = int(np.clip(mean_force * 20, 4, 18))
        cv2.circle(pressure, center, radius, mean_force, -1)

    pressure = cv2.GaussianBlur(pressure, BLUR_KERNEL, sigmaX=BLUR_SIGMA)
    if pressure.max() < 1e-8:
        return None

    pressure_norm = (255 * pressure / pressure.max()).astype(np.uint8)
    return cv2.applyColorMap(pressure_norm, cv2.COLORMAP_JET)


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    print("🚀 Starting GOAL-CONDITIONED tactile search...")

    mj_model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, data)

    # Load classifier
    dummy = datasets.ImageFolder(root=DATASET_DIR)
    class_names = dummy.classes

    clf = models.efficientnet_b0(weights=None)
    clf.classifier[1] = nn.Linear(clf.classifier[1].in_features, len(class_names))
    clf.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    clf.to(DEVICE).eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # CSV output
    os.makedirs(SEARCH_CSV_DIR, exist_ok=True)
    csv_path = os.path.join(SEARCH_CSV_DIR, f"tactile_search_{int(time.time())}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "time","object","x","y","z","force","phase","contact_idx"
        ])

    # ---------------------------------------------------------
    # Initialize freejoint addresses + PARK ALL OBJECTS
    # ---------------------------------------------------------
    for obj in OBJECTS:
        q = get_freejoint_qposadr(mj_model, obj["name"])
        obj["x_idx"] = q
        obj["y_idx"] = q + 1
        obj["z_idx"] = q + 2
        obj["quat_slice"] = slice(q+3, q+7)

        data.qpos[q:q+3] = [0.3, 0.0, -0.2]
        data.qpos[q+3:q+7] = [1,0,0,0]

    mujoco.mj_forward(mj_model, data)

    # ---------------------------------------------------------
    # Activate first object
    # ---------------------------------------------------------
    current_idx = 0
    active = OBJECTS[current_idx]
    cfg = active

    x_idx,y_idx,z_idx = active["x_idx"],active["y_idx"],active["z_idx"]
    quat_sl = active["quat_slice"]

    data.qpos[x_idx:x_idx+3] = [cfg["BASE_X"],0.0,cfg["FIX_Z"]]

    if active["name"]=="sphere":
        data.qpos[quat_sl] = [1,0,0,0]
    else:
        q0=np.zeros(4)
        mujoco.mju_euler2Quat(q0, np.array([0,np.pi/2,0]), "xyz")
        data.qpos[quat_sl]=q0

    mujoco.mj_forward(mj_model,data)

    # ---------------------------------------------------------
    # Rotation state
    # ---------------------------------------------------------
    rot_angle=0.0
    rot_axis_cycle=0.0

    phase="forward"
    start_time=time.time()
    pause_start=0
    rotate_start=0
    contact_idx=0
    contact_x=None

    contact_rows=[]
    next_start=0

    # ---------------------------------------------------------
    # FORCE + HEIGHT
    # ---------------------------------------------------------
    def tactile_force():
        fx,fy,fz=data.sensordata[0:3]
        touch=data.sensordata[3] if mj_model.nsensor>1 else 0.0
        return np.sqrt(fx*fx+fy*fy+fz*fz)+touch

    def maintain_height():
        data.qpos[z_idx]=cfg["FIX_Z"]

    def park_inactive_objects():
        for obj in OBJECTS:
            if obj is active:
                continue
            qi=obj["x_idx"]
            qs=obj["quat_slice"]
            data.qpos[qi:qi+3] = [0.3,0.0,-0.2]
            data.qpos[qs] = [1,0,0,0]

    # ---------------------------------------------------------
    # MAIN SIM LOOP
    # ---------------------------------------------------------
    with mujoco.viewer.launch_passive(mj_model,data) as viewer:
        print("👀 Running tactile exploration...")

        while viewer.is_running():
            t=time.time()-start_time
            f_val=tactile_force()

            maintain_height()
            park_inactive_objects()

            x=float(data.qpos[x_idx])
            y=float(data.qpos[y_idx])
            z=float(data.qpos[z_idx])

            # -------------------------------------------------
            # STATE MACHINE
            # -------------------------------------------------
            if phase=="forward":
                if active["name"]=="sphere":
                    data.qvel[x_idx]=-cfg["FORWARD_SPD"]
                else:
                    data.qpos[x_idx]-=cfg["FORWARD_SPD"]*mj_model.opt.timestep

                if f_val>cfg["FORCE_THR"]:
                    data.qvel[x_idx]=0
                    contact_x=x
                    phase="pause"
                    pause_start=time.time()

            elif phase=="pause":
                data.qvel[x_idx]=0

                # Stronger controlled push-in
                if f_val < cfg["FORCE_THR"]:
                    data.qpos[x_idx] -= cfg["NUDGE_X"] * 1.8   # ← BOOST CONTACT

                if contact_x is not None:
                    data.qpos[x_idx] = contact_x - 0.0006      # ← EXTRA SMALL PUSH

                if time.time()-pause_start>cfg["PAUSE_T"]:
                    phase="backward"

            elif phase=="backward":
                if active["name"]=="sphere":
                    data.qvel[x_idx]=+cfg["BACK_SPD"]
                else:
                    data.qpos[x_idx]+=cfg["BACK_SPD"]*mj_model.opt.timestep

                if data.qpos[x_idx]>=cfg["BASE_X"]:
                    data.qvel[x_idx]=0
                    phase="rotate"
                    rotate_start=time.time()

            elif phase=="rotate":
                f_val=0.0

                rot_axis_cycle+=mj_model.opt.timestep*(np.pi/4)
                rot_angle+=mj_model.opt.timestep*(np.pi/3)

                ax=np.array([
                    0.6+0.4*np.sin(rot_axis_cycle),
                    0.4+0.6*np.cos(0.7*rot_axis_cycle),
                    0.2+0.8*np.sin(0.5*rot_axis_cycle),
                ])
                ax/=np.linalg.norm(ax)

                quat=np.zeros(4)
                mujoco.mju_axisAngle2Quat(quat,ax,rot_angle)
                data.qpos[quat_sl]=quat

                data.qpos[x_idx]=cfg["BASE_X"]+0.005
                data.qpos[y_idx]=0.02*np.sin(rot_axis_cycle)
                data.qpos[z_idx]=cfg["FIX_Z"]+0.005*np.cos(rot_axis_cycle)

                if time.time()-rotate_start>cfg["ROT_T"]:
                    contact_idx+=1
                    phase="forward"
                    pause_start=time.time()
                    contact_x=None

                    if contact_idx>=MAX_CONTACTS_PER_OBJECT:

                        data.qpos[active["x_idx"]:active["x_idx"]+3]=[0.3,0,-0.2]
                        data.qpos[active["quat_slice"]]=[1,0,0,0]

                        current_idx=(current_idx+1)%len(OBJECTS)
                        active=OBJECTS[current_idx]
                        cfg=active

                        x_idx,y_idx,z_idx=active["x_idx"],active["y_idx"],active["z_idx"]
                        quat_sl=active["quat_slice"]

                        data.qpos[x_idx:x_idx+3]=[cfg["BASE_X"],0,cfg["FIX_Z"]]

                        if active["name"]=="sphere":
                            data.qpos[quat_sl]=[1,0,0,0]
                        else:
                            q0=np.zeros(4)
                            mujoco.mju_euler2Quat(q0,[0,np.pi/2,0],"xyz")
                            data.qpos[quat_sl]=q0

                        mujoco.mj_forward(mj_model,data)

                        rot_angle=0.0
                        rot_axis_cycle=0.0
                        contact_idx=0
                        contact_rows.clear()
                        next_start=0
                        contact_x=None

                        phase="forward"
                        pause_start=time.time()

            mujoco.mj_step(mj_model,data)
            data.qvel[:]*=DAMPING
            viewer.sync()
            time.sleep(mj_model.opt.timestep)

            # -------------------------------------------------
            # CSV LOG
            # -------------------------------------------------
            with open(csv_path,"a",newline="") as f:
                csv.writer(f).writerow([t,active["label"],x,y,z,f_val,phase,contact_idx])

            # -------------------------------------------------
            # CLASSIFICATION (unchanged)
            # -------------------------------------------------
            if f_val>0 and phase=="pause":
                contact_rows.append({
                    "time":t,"x_position":x,"y_position":y,"force":f_val,
                    "object":active["label"]
                })

                while len(contact_rows)-WINDOW_SIZE>=next_start:
                    win=contact_rows[next_start:next_start+WINDOW_SIZE]
                    next_start+=STRIDE

                    heatmap=window_to_heatmap(win)
                    if heatmap is None:
                        continue

                    rgb=cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
                    pil=Image.fromarray(rgb)
                    inp=transform(pil).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        logits=clf(inp)
                        probs=torch.softmax(logits,dim=1)[0]
                        idx=int(torch.argmax(probs))
                        pred=class_names[idx]
                        conf=float(probs[idx])

                    print(f"[PRED] {pred} ({conf*100:.2f}%) | GOAL={GOAL}")

                    if pred==GOAL:
                        out_path=os.path.join(SCRIPT_DIR,f"FOUND_{GOAL}.png")
                        cv2.imwrite(out_path,heatmap)
                        print("\n🎯 GOAL FOUND!")
                        viewer.close()
                        return

    print(f"[INFO] CSV saved to {csv_path}")


if __name__=="__main__":
    main()
