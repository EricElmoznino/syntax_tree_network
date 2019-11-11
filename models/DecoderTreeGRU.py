import torch
from torch import nn
from torch.nn import functional as F
from models.GRU import GRUDecode


class DecoderTreeGRU(nn.Module):

    def __init__(self, output_size, hidden_size):
        super().__init__()

        self.terminal_rule = nn.Linear(hidden_size, output_size, bias=False)
        self.nonterminal_rule = GRUDecode(hidden_size)

    def forward(self, tree, encoding):
        predictions = self.recursive_forward(tree, encoding)
        predictions = torch.stack(predictions)
        if not self.training:
            predictions = F.softmax(predictions, dim=2)
        return predictions

    def recursive_forward(self, tree, hidden):
        if tree.is_preterminal:
            output = [self.terminal_rule(hidden)]
        else:
            output = []
            h_children = self.nonterminal_rule(hidden)
            for c, h_c in zip(tree.children, h_children):
                output += self.recursive_forward(c, h_c)
        return output
