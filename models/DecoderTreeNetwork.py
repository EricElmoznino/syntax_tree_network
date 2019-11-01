import torch
from torch import nn
from torch.nn import functional as F


class DecoderTreeNetwork(nn.Module):

    def __init__(self, output_size, hidden_size, activation='tanh'):
        super().__init__()

        self.hidden_size = hidden_size

        self.terminal_rule = nn.Linear(hidden_size, output_size, bias=False)
        self.nonterminal_rule = nn.Linear(hidden_size, 2 * hidden_size)

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
            h_children = self.activation(h_children)
            h_children = h_children.split(self.hidden_size, dim=1)
            for c, h_c in zip(tree.children, h_children):
                output += self.recursive_forward(c, h_c)
        return output
