import random

import numpy as np


def pytest_runtest_setup() -> None:
    random.seed(0)
    np.random.seed(0)
    import torch
    torch.manual_seed(0)
