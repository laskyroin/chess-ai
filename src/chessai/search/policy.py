import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import chess
from .base import Bot
from ..encoding import board_to_tensor

class PolicyBot(Bot):

    name = "Policy"

    def select_move(self, board):
        encoded = board