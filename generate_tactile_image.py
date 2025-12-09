'''import os
import numpy as np
import pandas as pd
import cv2

# -----------------------------
# Configuration
# -----------------------------
csv_path = "tactile_data_sphere.csv"   # Input tactile CSV file
res = 128                              # Image resolution
blur_kernel = (9, 9)
blur_sigma = 2.0
WINDOW_SIZE = 50
STRIDE = 25

# -----------------------------
# Automatically infer shape name from file name
# -----------------------------
shape_name = os.path.splitext(os.path.basename(csv_path))[0].replace("tactile_data_", "")
save_dir = os.path.join("dataset", shape_name)
os.makedirs(save_dir, exist_ok=True)

print(f"Processing shape: {shape_name}")
print(f"Saving images to: {save_dir}")

# -----------------------------
# Load CSV and filter tactile contacts
# -----------------------------
df = pd.read_csv(csv_path)
df_contact = df[(df["force"] > 0) & (df["phase"] == "pause")].copy()
df_contact = df_contact.sort_values("time").reset_index(drop=True)
print(f"Using {len(df_contact)} contact rows (force>0, phase='pause').")

if df_contact.empty:
    raise ValueError("No valid tactile data found in this file.")

# -----------------------------
# Generate pseudo-GelSight images (local normalization)
# -----------------------------
count = 0
num_rows = len(df_contact)

for start in range(0, num_rows - WINDOW_SIZE + 1, STRIDE):
    window = df_contact.iloc[start:start + WINDOW_SIZE]

    # --- local normalization for this window ---
    xmin, xmax = window["x_position"].min(), window["x_position"].max()
    ymin, ymax = window["y_position"].min(), window["y_position"].max()

    # skip if no spatial variation (avoids degenerate frames)
    if abs(xmax - xmin) < 1e-6 or abs(ymax - ymin) < 1e-6:
        continue

    # map to image grid
    u = ((window["x_position"] - xmin) / (xmax - xmin) * (res - 1)).astype(int)
    v = ((window["y_position"] - ymin) / (ymax - ymin) * (res - 1)).astype(int)

    # build pressure map
    pressure_map = np.zeros((res, res), dtype=float)
    for x, y, f in zip(u, v, window["force"]):
        if 0 <= x < res and 0 <= y < res:
            pressure_map[v, x] += f

    # smooth and normalize
    pressure_map = cv2.GaussianBlur(pressure_map, blur_kernel, sigmaX=blur_sigma)
    max_val = pressure_map.max()
    if max_val < 1e-8:
        continue
    pressure_norm = (255 * pressure_map / (max_val + 1e-8)).astype(np.uint8)

    # apply colormap
    pseudo_img = cv2.applyColorMap(pressure_norm, cv2.COLORMAP_JET)

    # save with correct shape name
    img_name = os.path.join(save_dir, f"{shape_name}_{count:05d}.png")
    cv2.imwrite(img_name, pseudo_img)
    count += 1

    if count % 100 == 0:
        print(f"Saved {count} images...")

# -----------------------------
# Summary
# -----------------------------
if count > 0:
    print(f"\n✅ Done! Generated {count} pseudo-GelSight images for '{shape_name}' in '{save_dir}/'.")
else:
    print(f"\n⚠️ No images were generated for '{shape_name}'. Check the CSV content or window settings.")
'''

#for sphere
import os
import numpy as np
import pandas as pd
import cv2

# -----------------------------
# Configuration
# -----------------------------
csv_path = "tactile_data_cylinder.csv"   # Input tactile CSV file
res = 128                              # Image resolution
blur_kernel = (9, 9)
blur_sigma = 2.0
WINDOW_SIZE = 50                       # Number of rows per image
STRIDE = 25                            # Overlap stride

# -----------------------------
# Automatically infer shape name from file name
# -----------------------------
shape_name = os.path.splitext(os.path.basename(csv_path))[0].replace("tactile_data_", "")
save_dir = os.path.join("dataset", shape_name)
os.makedirs(save_dir, exist_ok=True)

print(f"Processing shape: {shape_name}")
print(f"Saving images to: {save_dir}")

# -----------------------------
# Load CSV and filter tactile contacts
# -----------------------------
df = pd.read_csv(csv_path)
df_contact = df[(df["force"] > 0) & (df["phase"] == "pause")].copy()
df_contact = df_contact.sort_values("time").reset_index(drop=True)
print(f"Using {len(df_contact)} contact rows (force>0, phase='pause').")

if df_contact.empty:
    raise ValueError("No valid tactile data found in this file.")

# -----------------------------
# Generate pseudo-GelSight images (with circular fallback)
# -----------------------------
count = 0
num_rows = len(df_contact)

for start in range(0, num_rows - WINDOW_SIZE + 1, STRIDE):
    window = df_contact.iloc[start:start + WINDOW_SIZE]

    # --- compute local coordinate range ---
    xmin, xmax = window["x_position"].min(), window["x_position"].max()
    ymin, ymax = window["y_position"].min(), window["y_position"].max()

    pressure_map = np.zeros((res, res), dtype=float)

    # --- CASE 1: Normal tactile motion (enough x/y variation) ---
    if (abs(xmax - xmin) >= 1e-6) and (abs(ymax - ymin) >= 1e-6):
        # map contact positions to pixel grid
        u = ((window["x_position"] - xmin) / (xmax - xmin) * (res - 1)).astype(int)
        v = ((window["y_position"] - ymin) / (ymax - ymin) * (res - 1)).astype(int)

        for x, y, f in zip(u, v, window["force"]):
            if 0 <= x < res and 0 <= y < res:
                pressure_map[v, x] += f

    # --- CASE 2: Stationary contact (sphere-like, little or no x/y motion) ---
    else:
        mean_force = float(window["force"].mean())
        # Simulate a circular pressure patch at the center
        center = (res // 2, res // 2)
        # Radius scaled by average force (bounded for stability)
        radius = int(np.clip(mean_force * 20, 4, 18))
        cv2.circle(pressure_map, center, radius, mean_force, -1)

    # --- Smooth and normalize the pressure map ---
    pressure_map = cv2.GaussianBlur(pressure_map, blur_kernel, sigmaX=blur_sigma)
    max_val = pressure_map.max()
    if max_val < 1e-8:
        continue

    pressure_norm = (255 * pressure_map / (max_val + 1e-8)).astype(np.uint8)
    pseudo_img = cv2.applyColorMap(pressure_norm, cv2.COLORMAP_JET)

    # --- Save image ---
    img_name = os.path.join(save_dir, f"{shape_name}_{count:05d}.png")
    cv2.imwrite(img_name, pseudo_img)
    count += 1

    if count % 100 == 0:
        print(f"Saved {count} images...")

# -----------------------------
# Summary
# -----------------------------
if count > 0:
    print(f"\n✅ Done! Generated {count} pseudo-GelSight images for '{shape_name}' in '{save_dir}/'.")
else:
    print(f"\n⚠️ No images were generated for '{shape_name}'. Check the CSV content or window settings.")
