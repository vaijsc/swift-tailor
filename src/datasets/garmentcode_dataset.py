import json
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from configs.envs import GLOBAL_CONFIG
from src.datasets.utils import generate_split, scale_omg
from src.utils.logger import get_logger
from src.utils.mics import get_all_gcd_files

logger = get_logger(__name__)


class DATASET_SPLIT(Enum):
    TRAIN = "training"
    VAL = "validation"
    TEST = "test"
    TEST_MODE = "test_mode"


class GarmentCodeData(Dataset):
    def __init__(
        self,
        gim_size: int = 128,
        with_snapped_edges: bool = False,
        use_official_split: bool = True,
        split: Union[DATASET_SPLIT, str] = DATASET_SPLIT.TRAIN,
        split_ratio: Union[float, List[float]] = 0.9,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()

        self._gim_size = gim_size
        self._with_snapped_edges = with_snapped_edges
        self._split = DATASET_SPLIT(split)
        self._ratio: List[float] = (
            split_ratio
            if isinstance(split_ratio, List)
            else [split_ratio, 1 - split_ratio]
        )
        self._seed = seed

        # Load list of sample paths
        logger.info("Using official split: %s", use_official_split)
        self.samples = (
            self._extract_official_split()
            if use_official_split
            else self._extract_split()
        )

        self._label_mapping = self._get_label_idx_mapping()

    def _extract_official_split(self) -> np.ndarray:
        with open(
            GLOBAL_CONFIG.gcd_data_dir
            / "GarmentCodeData_v2_official_train_valid_test_data_split.json",
            "r",
        ) as f:
            split: Dict[str, List[str]] = json.load(f)

        if self._split == DATASET_SPLIT.TEST_MODE:
            samples = split["training"][:100]
        else:
            samples = split[self._split.value]

        # Add absolute path to samples
        samples = [GLOBAL_CONFIG.gcd_data_dir / sample for sample in samples]

        # Add prefix path "data" to sample_id due to extraction
        # ../sample_id -> ../data/sample_id
        samples = [sample.parent / "data" / sample.name for sample in samples]

        np_samples = np.array(samples)
        logger.info(f"Total orignal samples: {len(np_samples)}")
        np_samples = self._filter_existing_samples(np_samples)
        logger.info(f"Total valid samples after filter: {len(np_samples)}")

        return np_samples

    def _extract_split(self) -> np.ndarray:
        all_samples = get_all_gcd_files()

        indices = generate_split(
            len(all_samples),
            seed=self._seed,
            split_ratios=self._ratio,
            shuffle=True,
        )

        if (
            self._split == DATASET_SPLIT.TRAIN
            or self._split == DATASET_SPLIT.TEST_MODE
        ):
            split_indices = indices[0]
        elif self._split == DATASET_SPLIT.VAL:
            split_indices = indices[1]
        elif self._split == DATASET_SPLIT.TEST:
            assert len(indices) == 3, "Test split is not available"
            split_indices = indices[2]
        else:
            raise ValueError("Invalid split type")

        np_samples = np.array(all_samples)[split_indices]
        logger.info(f"Total orignal samples: {len(np_samples)}")
        np_samples = self._filter_existing_samples(np_samples)
        logger.info(f"Total valid samples after filter: {len(np_samples)}")

        if self._split == DATASET_SPLIT.TEST_MODE:
            np_samples = np_samples[:100]

        return np_samples

    def _filter_existing_samples(self, samples: np.ndarray) -> np.ndarray:
        is_valid_path = [
            (sample / f"{sample.stem}_gim.npz").exists() for sample in samples
        ]

        valid_indices = np.where(is_valid_path)[0]

        return samples[valid_indices]

    def _get_label_idx_mapping(self):
        design_path = (
            GLOBAL_CONFIG.root
            / 'assets' / 'design_params' / 'default.yaml'
        )
        design_params = yaml.safe_load(open(design_path, 'r'))

        component_names = ['upper', 'wb', 'bottom']
        components = [
            design_params['design']['meta'][name]['range']
            for name in component_names
        ]
        idx = {comb: i for i, comb in enumerate(product(*components))}

        assert idx[(None, None, None)] == len(idx) - 1, (
            "The last index should be reserved for "
            "the default combination (None, None, None)"
        )
        return idx

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        sample: Path = self.samples[index]  # type: ignore
        sample_id = sample.stem

        gim_path = sample / f"{sample_id}_gim.npz"
        gim = np.load(gim_path)["arr_0"]

        design_path = sample / f"{sample_id}_design_params.yaml"
        design_params = yaml.safe_load(open(design_path, "r"))['design']['meta']
        combination = (
            design_params["upper"]["v"],
            design_params["wb"]["v"],
            design_params["bottom"]["v"],
        )
        label = self._label_mapping[combination]

        option = "omg_down_star" if self._with_snapped_edges else "omg_down"
        gim = scale_omg(gim, self._gim_size)[option]  # H, W, C
        gim = np.transpose(gim, (2, 0, 1))  # C, H, W

        return {
            "gim": torch.Tensor(gim),
            "label": label,
        }

    @staticmethod
    def collate_fn(batch):
        gims = torch.stack([sample["gim"] for sample in batch])
        labels = torch.Tensor([sample["label"] for sample in batch])
        return {
            "gim": gims,
            "label": labels,
        }


if __name__ == "__main__":
    ...
