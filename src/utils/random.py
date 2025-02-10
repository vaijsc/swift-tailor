import contextlib
import random

import numpy as np
import torch


@contextlib.contextmanager
def temp_seed(seed):
    # Save current states
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    # Set new seed
    set_seed(seed)

    try:
        yield  # Run code within context
    finally:
        # Restore states
        set_state(np_state, py_state, torch_state, cuda_state)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_state(np_state, py_state, torch_state, cuda_state=None):
    np.random.set_state(np_state)
    random.setstate(py_state)
    torch.set_rng_state(torch_state)
    if cuda_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
