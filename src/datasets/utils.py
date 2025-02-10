from typing import Dict, List

import einops
import numpy as np
import scipy
import skimage

from src.utils.random import temp_seed


def generate_split(
    N: int,
    seed: int = 42,
    shuffle: bool = True,
    split_ratios: List[float] = [0.8, 0.1, 0.1],
) -> List[np.ndarray]:
    """
    Return a list of splits of np.arange(N) with deterministic mode
    using temporary random seed
    Args:
        N: number of samples
        seed: random seed. Defaults to 42.
        shuffle: shuffle the index. Defaults to True.
        split_ratios: list of ratios.

    Returns:
        List of index splits
    """

    splits = np.array(split_ratios)
    assert np.sum(splits) == 1

    index = np.arange(N)
    with temp_seed(seed):
        if shuffle:
            np.random.shuffle(index)
        accu_splits = np.round(np.cumsum(splits) * N).astype(int)
        assert accu_splits[-1] == N
        splits = np.split(index, accu_splits)[:-1]
    return splits


def scale_omg(
    omg: np.ndarray,
    target_size: int = 64,
    anti_aliasing: bool = False,
    visualize: bool = False,
) -> Dict[str, np.ndarray]:
    """Downsample omg with edge snapping

    Args:
        omg: (H, W, 3) omage tensor
        factor: downsample factor
        anti_aliasing: whether to use anti_aliasing
        visualize: whether to visualize the result
    Returns:
        dict, containing 'omg_down_star', 'omg_down', 'sov', 'edge_occ_down'
        'omg_down_star': np.ndarray, downsampled omg with edge snapping
        'omg_down': np.ndarray, downsampled omg without edge snapping
        'sov': np.ndarray, occupancy map with snapped boundaries highlighted
        'edge_occ_down': np.ndarray, edge occupancy map
    """
    assert omg.shape[0] == omg.shape[1], "Input omg must be square"

    # Padded omg to the nearest multiple of target_size
    padded_size = (omg.shape[0] + target_size - 1) // target_size * target_size
    pad_top = pad_left = (padded_size - omg.shape[0]) // 2
    pad_bottom = pad_right = padded_size - omg.shape[0] - pad_top

    padded_omg = np.pad(
        omg,
        (
            (pad_top, pad_bottom),
            (pad_left, pad_right),
            (0, 0),  # no padding for channel
        ),
        mode="constant",
        constant_values=-1,
    )
    omg = padded_omg
    factor = omg.shape[0] // target_size

    # TODO: resolove for < 64 x 64 omage

    # Occupancy map (all channels == 0)
    occ = np.any(omg != 0, axis=-1)

    # Extract edge in occupancy map
    # Edge = Occupancy map - Erosion of occupancy map
    eroded_mask = scipy.ndimage.binary_erosion(
        occ, structure=np.ones((3, 3))
    )  # square structure is needed to get the corners
    edge_occ = ~eroded_mask & occ
    edge_omg = omg.copy()
    edge_omg[edge_occ == 0] = -1.0

    # Seperate edge_occ and edge_omg into patches
    edge_occ_patches = einops.rearrange(
        edge_occ, "(h1 h2) (w1 w2) -> h1 w1 h2 w2", h2=factor, w2=factor
    )
    edge_occ_down = edge_occ_patches.max(axis=-1).max(axis=-1)
    eod_0_count = (edge_occ_patches == 0).sum(axis=-1).sum(axis=-1)
    eod_1_count = (edge_occ_patches == 1).sum(axis=-1).sum(axis=-1)
    edge_omg_patches = einops.rearrange(
        edge_omg, "(h1 h2) (w1 w2) c-> h1 w1 h2 w2 c", h2=factor, w2=factor
    )
    edge_omg_down = (
        edge_omg_patches.sum(axis=-2).sum(axis=-2) + eod_0_count[..., None]
    )
    edge_omg_down = np.divide(
        edge_omg_down,
        eod_1_count[..., None],
        out=np.zeros_like(edge_omg_down),
        where=eod_1_count[..., None] != 0,
    )

    omg_down = skimage.transform.resize(
        omg,
        (omg.shape[0] // factor,) * 2,
        order=0,
        preserve_range=False,
        anti_aliasing=anti_aliasing,
    )

    omg_down_star = edge_omg_down * (edge_occ_down[..., None]) + omg_down * (
        1 - edge_occ_down[..., None]
    )

    sov = np.any(omg_down != 0, axis=-1)  # for visualizaton
    sov = sov * 0.5 + edge_occ_down.astype(float)
    sov[sov >= 1.0] = 1.0
    if visualize:
        import matplotlib.pyplot as plt

        plt.imshow(sov, cmap="gray")
        plt.show()
        plt.imshow(edge_occ_down, cmap="gray")
        plt.show()
        plt.imshow(omg_down)
        plt.show()
        plt.imshow(omg_down_star)
        plt.show()

    return dict(
        omg_down_star=np.clip(omg_down_star, 0, 1),
        omg_down=np.clip(omg_down, 0, 1),
        sov=sov,
        edge_occ_down=edge_occ_down,
        edge_occ=edge_occ,
    )


def ratio2int(percentage, max_val):
    if 0 <= percentage <= 1 and type(percentage) is not int:
        out = percentage * max_val
    elif 1 <= percentage <= max_val:
        out = percentage
    elif max_val < percentage:
        out = max_val
    else:
        raise ValueError("percentage cannot be negative")
    return out


def parse_range(r, max_num):
    if type(r) is int:
        if r == -1:
            num = max_num
        else:
            raise ValueError("%s should be -1" % r)
    else:
        assert len(r) == 2
        r[0] = ratio2int(r[0], max_num)
        r[1] = ratio2int(r[1], max_num) + 1
        num = np.random.randint(*r)
    return num


def random_choice(total, choiceN, sort=False):
    locations = np.random.choice(total, size=choiceN, replace=False)
    locations = locations if not sort else np.sort(locations)
    return locations
