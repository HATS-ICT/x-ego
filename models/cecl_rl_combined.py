"""
Model using CECL and RL combined, following architecture of AEComm, with message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CECLRLCombined(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        return x