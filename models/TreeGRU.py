import torch
from torch import nn
from models.GRU import GRUEncode


class TreeGRU(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.nonterminal_rule = GRUEncode(hidden_size)

    def forward(self, tree):
        return self.recursive_forward(tree)

    def recursive_forward(self, tree):
        if tree.is_preterminal:
            h = self.embedding(tree.children[0].node)
            h = torch.tanh(h)
        else:
            h_children = [self.recursive_forward(c) for c in tree.children]
            h = self.nonterminal_rule(*h_children)
        return h
