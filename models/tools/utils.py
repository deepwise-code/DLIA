from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch

def count_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params