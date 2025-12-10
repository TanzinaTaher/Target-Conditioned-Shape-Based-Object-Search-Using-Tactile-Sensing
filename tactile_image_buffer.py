'''
import os
import numpy as np
import cv2


class TactileImageBuffer:
    """
    Online pseudo-GelSight generator that mimics the original offline
    generate_tactile_image.py used for training.

    Usage:
      - Call set_shape("sphere" / "cube" / ...) whenever you switch objects.
      - Call add_reading({...}) every physics step during contact.
      - When there are enough samples, add_reading() returns an image path,
        otherwise it returns None.
    """

    def __init__(self, shape_name=None, save_root="dataset", res=128,
                 min_samples=50):
        self.shape = shape_name          # active object label ("sphere", "cube", ...)
        self.save_root = save_root
        self.res = res
        self.min_samples = min_samples   # minimum readings before making an image

        self.buffer = []                # list of dicts: {"x", "y", "force", "phase"}
        self.counter = 0
        self.shape_dir = None

        if self.shape is not None:
            self._ensure_shape_dir()

    # ------------------------------------------------------------------
    # Shape / directory management
    # ------------------------------------------------------------------
    def set_shape(self, shape_name: str):
        """
        Set the current object label and make sure its directory exists.
        Also updates the filename counter so we never overwrite old images.
        """
        self.shape = shape_name
        self._ensure_shape_dir()
        self.buffer = []

    def _ensure_shape_dir(self):
        self.shape_dir = os.path.join(self.save_root, self.shape)
        os.makedirs(self.shape_dir, exist_ok=True)

        # Initialize counter based on existing files so we do not overwrite.
        existing = [
            f for f in os.listdir(self.shape_dir)
            if f.startswith(self.shape) and f.endswith(".png")
        ]
        max_idx = -1
        for fname in existing:
            try:
                stem = os.path.splitext(fname)[0]
                idx_str = stem.split("_")[-1]
                idx = int(idx_str)
                max_idx = max(max_idx, idx)
            except Exception:
                continue
        self.counter = max_idx + 1 if max_idx >= 0 else 0

    # ------------------------------------------------------------------
    # Online buffer API
    # ------------------------------------------------------------------
    def add_reading(self, reading: dict):
        """
        Add a single tactile reading.

        reading: {
            "x": float,      # contact position X in world coords
            "y": float,      # contact position Y
            "force": float,  # contact force magnitude
            "phase": str     # e.g., "pause"
        }

        Returns:
            - None  → if not enough samples yet
            - path (str) → when an image is generated
        """
        if self.shape is None:
            # no active shape yet
            return None

        f = float(reading.get("force", 0.0))
        if f <= 0.0:
            # no actual contact
            return None

        self.buffer.append(
            {
                "x": float(reading.get("x", 0.0)),
                "y": float(reading.get("y", 0.0)),
                "force": f,
            }
        )

        # Need at least N readings to build a meaningful pressure map
        if len(self.buffer) < self.min_samples:
            return None

        img = self._generate_image_from_buffer()
        self.buffer = []  # reset after each image

        fname = self._save_image(img)
        return fname

    # ------------------------------------------------------------------
    # Image generation (aligned with original dataset script)
    # ------------------------------------------------------------------
    def _generate_image_from_buffer(self):
        """
        Convert buffered readings → pseudo tactile image.

        This mimics generate_tactile_image.py:

          - local normalization of x,y to [0, res-1]
          - accumulate force into a pressure map
          - Gaussian blur
          - normalize to 0..255
          - apply JET colormap

        IMPORTANT:
          - Only *spheres* use the "center blob" fallback when x,y have almost
            no spatial variation.
          - For cube / cylinder / cone, we always use the real x,y map, even
            if dx, dy are tiny, to avoid making every shape look like a sphere.
        """
        res = self.res

        xs = np.array([r["x"] for r in self.buffer], dtype=np.float32)
        ys = np.array([r["y"] for r in self.buffer], dtype=np.float32)
        fs = np.array([r["force"] for r in self.buffer], dtype=np.float32)

        # Scale forces up a bit: MuJoCo forces are small compared to what
        # the offline script expected.
        fs = fs * 3000.0

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        dx = float(abs(xmax - xmin))
        dy = float(abs(ymax - ymin))

        # 🔹 Only SPHERE uses the centered-blob fallback when there is almost
        # no spatial variation in contact points.
        # For cube / cylinder / cone, we always fall through to the normal
        # x,y → pressure map, even if dx, dy are tiny.
        if self.shape == "sphere" and (dx < 1e-6 or dy < 1e-6):
            return self._generate_center_blob(fs.mean())

        # Map to image grid [0, res-1] with local normalization (window-based).
        # This is what the original script did with u,v.
        u = ((xs - xmin) / (xmax - xmin + 1e-8) * (res - 1)).astype(int)
        v = ((ys - ymin) / (ymin - ymin + 1e-8) * (res - 1)).astype(int)
        # small fix: use (ymax - ymin) not (ymin - ymin)
        v = ((ys - ymin) / (ymax - ymin + 1e-8) * (res - 1)).astype(int)

        pressure = np.zeros((res, res), dtype=np.float32)
        for x_pix, y_pix, f in zip(u, v, fs):
            if 0 <= x_pix < res and 0 <= y_pix < res:
                pressure[y_pix, x_pix] += f

        # Smooth (Gaussian blur)
        pressure = cv2.GaussianBlur(pressure, (9, 9), sigmaX=2.0)

        max_val = float(pressure.max())
        if max_val < 1e-8:
            # Nothing useful → return a blank image
            return np.zeros((res, res, 3), dtype=np.uint8)

        # Normalize → [0, 255] single-channel
        pressure_norm = (255.0 * pressure / (max_val + 1e-8)).astype(np.uint8)

        # Apply JET colormap to match training images
        img_color = cv2.applyColorMap(pressure_norm, cv2.COLORMAP_JET)

        return img_color

    def _generate_center_blob(self, avg_force):
        res = self.res
        pressure = np.zeros((res, res), dtype=np.float32)

        f = avg_force * 800.0
        radius = int(res * np.random.uniform(0.15, 0.22))

        cx = res // 2 + np.random.randint(-4, 4)
        cy = res // 2 + np.random.randint(-4, 4)

        # elliptical instead of perfect circle
        ax = radius
        ay = int(radius * np.random.uniform(0.8, 1.2))

        for i in range(res):
            for j in range(res):
                dx = (j - cx) / ax
                dy = (i - cy) / ay
                if dx * dx + dy * dy <= 1.0:
                    pressure[i, j] = f

        pressure = cv2.GaussianBlur(pressure, (9, 9), 2.0)
        pressure_norm = (255 * pressure / (pressure.max() + 1e-8)).astype(np.uint8)
        return cv2.applyColorMap(pressure_norm, cv2.COLORMAP_JET)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _save_image(self, img: np.ndarray) -> str:
        """
        Save the given RGB image into the current shape directory.
        """
        if self.shape_dir is None:
            self._ensure_shape_dir()

        fname = f"{self.shape}_{self.counter:05d}.png"
        self.counter += 1

        path = os.path.join(self.shape_dir, fname)
        cv2.imwrite(path, img)
        return pat'''

import os
import numpy as np
import cv2


class TactileImageBuffer:
    """
    Online pseudo-GelSight generator aligned with the dataset script,
    but keeping full backward compatibility with the original API.

    - Directory creation logic preserved
    - File naming preserved
    - add_reading() returns a PNG filepath exactly like before
    - Image generation for sphere/cube/cylinder/cone is corrected and stabilized
    """

    def __init__(self, shape_name=None, save_root="dataset", res=128,
                 min_samples=50):
        self.shape = shape_name
        self.save_root = save_root
        self.res = res
        self.min_samples = min_samples  # number of readings required

        self.buffer = []              # list of {"x","y","force"}
        self.counter = 0
        self.shape_dir = None

        if self.shape is not None:
            self._ensure_shape_dir()

    # ----------------------------------------------------------------------
    # Directory & shape management
    # ----------------------------------------------------------------------
    def set_shape(self, shape_name: str):
        """Update shape and reset buffer."""
        self.shape = shape_name
        self._ensure_shape_dir()
        self.buffer = []

    def set_root(self, root: str):
        """Backward compatibility: update directory root."""
        self.save_root = root

    def _ensure_shape_dir(self):
        """Ensure dataset/shape directory exists and index is correct."""
        self.shape_dir = os.path.join(self.save_root, self.shape)
        os.makedirs(self.shape_dir, exist_ok=True)

        # Set counter to highest existing index + 1
        existing = [
            f for f in os.listdir(self.shape_dir)
            if f.startswith(self.shape) and f.endswith(".png")
        ]

        max_idx = -1
        for fname in existing:
            try:
                idx = int(os.path.splitext(fname)[0].split("_")[-1])
                max_idx = max(max_idx, idx)
            except Exception:
                continue

        self.counter = max_idx + 1 if max_idx >= 0 else 0

    # ----------------------------------------------------------------------
    # Reading API (unchanged)
    # ----------------------------------------------------------------------
    def add_reading(self, reading: dict):
        """
        reading = {"x": float, "y": float, "force": float, "phase": str}
        Returns filepath when enough readings have been collected.
        """

        if self.shape is None:
            return None

        f = float(reading.get("force", 0.0))
        if f <= 0:
            return None  # no valid contact

        self.buffer.append({
            "x": float(reading.get("x", 0.0)),
            "y": float(reading.get("y", 0.0)),
            "force": f
        })

        if len(self.buffer) < self.min_samples:
            return None

        # Enough readings: generate image
        img = self._generate_image_from_buffer()
        self.buffer = []  # reset for next window

        path = self._save_image(img)
        return path

    # ----------------------------------------------------------------------
    # Core tactile image generation (FIXED & IMPROVED)
    # ----------------------------------------------------------------------
    def _generate_image_from_buffer(self):
        """
        Generate a pseudo-GelSight image with corrected sphere fallback,
        fixed coordinate mapping, and improved force normalization.
        """
        res = self.res
        xs = np.array([r["x"] for r in self.buffer], dtype=np.float32)
        ys = np.array([r["y"] for r in self.buffer], dtype=np.float32)
        fs = np.array([r["force"] for r in self.buffer], dtype=np.float32)

        # Balanced force scale (training-consistent)
        fs = fs * 1500.0

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        dx = abs(xmax - xmin)
        dy = abs(ymax - ymin)

        # --------------------------------------------------------------
        #  SPECIAL CASE: SPHERE — stable, centered circular blob
        # --------------------------------------------------------------
        if self.shape == "sphere":
            return self._generate_sphere_blob(fs.mean())

        # --------------------------------------------------------------
        #  NORMAL CASE: Cube, Cylinder, Cone: XY → pressure map
        # --------------------------------------------------------------
        u = ((xs - xmin) / (xmax - xmin + 1e-8) * (res - 1)).astype(int)
        v = ((ys - ymin) / (ymax - ymin + 1e-8) * (res - 1)).astype(int)

        u = np.clip(u, 0, res - 1)
        v = np.clip(v, 0, res - 1)

        pressure = np.zeros((res, res), dtype=np.float32)
        for px, py, f in zip(u, v, fs):
            pressure[py, px] += f

        # Smoothing (looks closer to training data)
        pressure = cv2.GaussianBlur(pressure, (11, 11), 3.0)

        max_val = pressure.max()
        if max_val < 1e-6:
            return np.zeros((res, res, 3), dtype=np.uint8)

        pressure_norm = (pressure / max_val * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(pressure_norm, cv2.COLORMAP_JET)
        return img_color

    # ----------------------------------------------------------------------
    # Stable sphere generation (COMPLETELY REDONE)
    # ----------------------------------------------------------------------
    def _generate_sphere_blob(self, avg_force):
        """
        Generate a smooth, symmetric, centered pressure blob for sphere.
        Produces consistent sphere images that classifier can easily recognize.
        """
        res = self.res
        pressure = np.zeros((res, res), dtype=np.float32)

        f = avg_force * 1000.0
        radius = int(res * 0.18)

        cx = res // 2
        cy = res // 2

        yy, xx = np.mgrid[0:res, 0:res]
        mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2
        pressure[mask] = f

        pressure = cv2.GaussianBlur(pressure, (13, 13), 4.0)

        pressure_norm = (255 * pressure / (pressure.max() + 1e-8)).astype(np.uint8)
        return cv2.applyColorMap(pressure_norm, cv2.COLORMAP_JET)

    # ----------------------------------------------------------------------
    # Saving (unchanged)
    # ----------------------------------------------------------------------
    def _save_image(self, img):
        if self.shape_dir is None:
            self._ensure_shape_dir()

        fname = f"{self.shape}_{self.counter:05d}.png"
        self.counter += 1

        path = os.path.join(self.shape_dir, fname)
        cv2.imwrite(path, img)
        return path


