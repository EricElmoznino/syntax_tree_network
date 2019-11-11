import torch
from torch import nn


class TreeNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, activation='tanh'):
        super().__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.nonterminal_rule = nn.Linear(2 * hidden_size, hidden_size)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        else:
            raise ValueError('Unknown activation type: %s' % activation)

    def forward(self, tree):
        return self.recursive_forward(tree)

    def recursive_forward(self, tree):
        if tree.is_preterminal:
            h = self.embedding(tree.children[0].node)
        else:
            h_children = [self.recursive_forward(c) for c in tree.children]
            h_agr = torch.cat(h_children, dim=1)
            h = self.nonterminal_rule(h_agr)
        h = self.activation(h)
        return h
