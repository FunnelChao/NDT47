import torch
import random
import numpy as np

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def r_squared(y_true, y_pred):
    """
    shape:[bsz,2]
    """
    mean_y_true = torch.mean(y_true,dim=0)
    ss_tot = torch.sum((y_true - mean_y_true) ** 2, dim=0)
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    r2 = 1 - (ss_res / ss_tot)
    return r2.mean()


import yaml
from easydict import EasyDict
def load_cfg(cfg_path):
    with open(cfg_path) as file:
        data = yaml.safe_load(file)
    return data