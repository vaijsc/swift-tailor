import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Disable print
# import builtins
# builtins.print = lambda *args, **kwargs: None


def generate_uv(sample_path: str, config: dict) -> None:
    from configs import GLOBAL_CONFIG
    from src.preprocessing.unwarping import create_geometry_uv

    root = Path(sample_path)
    sample_id = root.name
    print("[DEBUG] Processing: ", root.relative_to(GLOBAL_CONFIG.gcd_data_dir))

    path_to_mesh = root / f"{sample_id}_sim.ply"
    path_to_texture = root / f"{sample_id}_texture.png"
    geometry_uv_path = root / f"{sample_id}_gim"

    npy_geometry_uv_path = root / f"{sample_id}_geometry.npy"
    npz_geometry_uv_path = root / f"{sample_id}_geometry.npz"

    if npy_geometry_uv_path.exists():
        npy_geometry_uv_path.unlink()
    if npz_geometry_uv_path.exists():
        npz_geometry_uv_path.unlink()

    # if geometry_uv_path.exists():
    #     print("[DEBUG] UV already exists")
    #     return

    assert path_to_mesh.exists(), \
        f"Path to mesh does not exist: {path_to_mesh}"
    assert path_to_texture.exists(), \
        f"Path to texture does not exist: {path_to_texture}"

    create_geometry_uv(
        obj_file_path=path_to_mesh,
        texture_file_path=path_to_texture,
        geometry_uv_path=geometry_uv_path,
        **config,
    )


def extract_subset(all_samples, num_subsets, subset_id):
    assert subset_id < num_subsets, \
        "Subset ID must be less than number of subsets"

    n_samples = len(all_samples)

    subset_size = n_samples // num_subsets + 1
    st_id = subset_size * subset_id
    end_id = min(st_id + subset_size, n_samples)

    return all_samples[st_id:end_id]


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run UV generation with multiprocessing")
    parser.add_argument(
        "--num-subsets",
        type=int,
        help="Number of sections to divide set of all samples",
        default=5,
    )

    parser.add_argument(
        "--subset-id",
        type=int,
        help="ID of subset to process",
        default=0,
    )
    args = parser.parse_args()

    from configs import GLOBAL_CONFIG
    from src.utils.mics import get_all_gcd_files
    all_samples = get_all_gcd_files()
    all_samples = sorted(all_samples)
    all_samples = extract_subset(all_samples, args.num_subsets, args.subset_id)

    with open(GLOBAL_CONFIG.config_dir / "preprocessing.yaml", "r") as file:
        config = yaml.safe_load(file)

    errors = pd.DataFrame(columns=["body_type", "sample_id", "error"])
    for idx, sample_path in enumerate(all_samples):
        try:
            print("[DEBUG] Progress: ", idx, "/", len(all_samples))
            generate_uv(sample_path, config)
        except Exception as e:
            print("[ERROR] ", e)
            path = Path(sample_path)
            sample_id = path.name
            body_type = path.parent.parent.name
            errors.loc[len(errors)] = [body_type, sample_id, e]
    if len(errors) > 0:
        errors.to_csv(GLOBAL_CONFIG.data_dir
                      / f"errors_{args.subset_id}.csv", index=False)
