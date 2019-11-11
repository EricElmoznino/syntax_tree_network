import torch
from torch import nn
from models.GRU import GRUEncode


class TreeGRU(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.terminal_rule = nn.Linear(input_size, hidden_size)