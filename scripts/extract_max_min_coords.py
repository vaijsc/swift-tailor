import os
import sys

import numpy as np
import trimesh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def extract_max_min():
    from configs.envs import GLOBAL_CONFIG
    root = GLOBAL_CONFIG.gcd_data_dir / "GarmentCodeData_v2"
    subdirs = os.listdir(root)
    BODY_TYPE = ["default_body", "random_body"]

    max_coords = -np.inf
    min_coords = np.inf
    cnt = 0

    from tqdm import tqdm
    for subdir in subdirs:
        for body_type in BODY_TYPE:
            print(f"[DEBUG] Processing {subdir} - {body_type}")
            path = root / subdir / body_type / "data"
            assert path.exists(), f"Path {path} does not exist"

            for file in tqdm(os.listdir(path)):
                if not os.path.isdir(path / file):
                    continue
                file_path = path / file / f"{file}_sim.ply"
                assert file_path.exists(), f"File {file_path} does not exist"
                cnt += 1

                mesh = trimesh.load(file_path)

                max_coords = max(max_coords, mesh.vertices.max())
                min_coords = min(min_coords, mesh.vertices.min())

    print("[DEBUG] total files: ", cnt)

    return max_coords, min_coords


if __name__ == "__main__":
    np.save("max_min_coords", extract_max_min())
