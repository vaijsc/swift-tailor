import math
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from xgutils import *


class VisTest(plutil.VisCallback):
    def __init__(self, **kwargs):
        self.__dict__.update(locals())
        super().__init__(**kwargs)
        # self.vqvae = init_trained_model_from_ckpt(vqvae_opt)#.to("cuda")

    def compute_batch(self, batch):
        return ptutil.ths2nps(batch)

    def visualize_batch(self, computed):
        computed = ptutil.ths2nps(computed)
