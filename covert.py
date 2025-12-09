from stl import mesh
import os

ASSET_DIR = "assets"
for f in os.listdir(ASSET_DIR):
    if f.endswith(".stl"):
        path = os.path.join(ASSET_DIR, f)
        print(f"Converting {path} ...")
        try:
            m = mesh.Mesh.from_file(path)
            m.save(path, mode=stl.Mode.ASCII)
            print("✅ Converted successfully")
        except Exception as e:
            print("❌ Failed:", e)
