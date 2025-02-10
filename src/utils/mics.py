import os
from typing import List

from configs import GLOBAL_CONFIG


def get_all_gcd_files() -> List[str]:
    """
    Extract all paths of samples in GarmentCodeData
    """
    root = GLOBAL_CONFIG.gcd_data_dir / "GarmentCodeData_v2"
    BODY_TYPE = ["default_body", "random_body"]

    subdirs = os.listdir(root)
    all_sample_paths = []
    for subdir in subdirs:
        for body_type in BODY_TYPE:
            path = root / subdir / body_type / "data"

            sample_paths = [
                path / sample
                for sample in os.listdir(path)
                if os.path.isdir(path / sample)
            ]

            all_sample_paths.extend(sample_paths)

    return all_sample_paths
