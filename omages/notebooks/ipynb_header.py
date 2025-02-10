from IPython import get_ipython

# Get the current IPython instance
ipython = get_ipython()

if ipython is not None:
    # Run magic commands
    ipython.run_line_magic("matplotlib", "inline")
    ipython.run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    ipython.run_line_magic("reload_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import copy
import json
import logging
import os

# import torch
# sys.path.append('../')
# chdir to the parent folder of the absolute path of this file
import pathlib
import sys
import warnings

import einops
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy

file_path = pathlib.Path(__file__).parent.resolve()
print(file_path.parents[0])
os.chdir(file_path.parents[0])
import glob

import igl
import numpy as np
import pandas as pd

# torch.cuda.set_device("cuda:1")
import skimage
import torch
from xgutils import geoutil, nputil, plutil, ptutil, sysutil
from xgutils.vis import fresnelvis, plt3d, visutil

# from nnrecon.trainer2 import Trainer
from src.trainer import Trainer
