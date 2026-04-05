# =============================================================================
# graphrl/utils/seed.py —— 随机种子工具
# =============================================================================

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """固定所有随机源，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
