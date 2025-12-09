import numpy as np
from stl import mesh

# --- Cone parameters ---
radius = 0.02       # base radius (m)
height = 0.04       # total height (m)
segments = 40       # smoothness (more = smoother)

# --- Generate base circle vertices ---
theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
circle = np.stack([
    radius * np.cos(theta),
    radius * np.sin(theta),
    np.zeros_like(theta)
], axis=1)

# --- Apex and base center ---
apex = np.array([[0, 0, height]])   # tip at +Z
base_center = np.array([[0, 0, 0]]) # base center
verts = np.vstack((circle, apex, base_center))

# --- Build faces ---
faces = []
apex_index = len(circle)
base_center_index = apex_index + 1

# Side triangles
for i in range(segments):
    j = (i + 1) % segments
    faces.append([i, j, apex_index])

# Base triangles
for i in range(segments):
    j = (i + 1) % segments
    faces.append([i, base_center_index, j])

faces = np.array(faces)

# --- Create mesh ---
cone_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for k in range(3):
        cone_mesh.vectors[i][k] = verts[f[k], :]

# --- Save binary STL ---
cone_mesh.save("pyramid_fixed.stl", mode=mesh.stl.Mode.BINARY)
print("✅ Saved 'pyramid_fixed.stl' (flat base at z=0, tip at z=0.04, binary STL)")
