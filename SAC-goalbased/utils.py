import rlkit.torch.pytorch_util as ptu
import os
from shutil import copyfile, move
import time
import numpy as np

def move_to_cpu():
    """ Set device to cpu for torch.
    """
    ptu.set_gpu_mode(False)

def move_to_cuda(config):
    """ Set device to CUDA and which GPU, or CPU if not set.

    Args:
        config: Dict containing the configuration.
    """
    if config['use_gpu']:
        if 'cuda_device' in config:
            cuda_device = config['cuda_device']
        else:
            cuda_device = 0
        ptu.set_gpu_mode(True, cuda_device)



