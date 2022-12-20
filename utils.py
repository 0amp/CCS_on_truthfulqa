import json
import os
import re
from collections import defaultdict
from typing import List, Optional, Tuple, Dict

from einops import rearrange
from tqdm import tqdm
import numpy
import torch

def untuple(x):
    return x[0] if isinstance(x, tuple) else x

def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    
    From util/nethook.py in ROME repo
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)