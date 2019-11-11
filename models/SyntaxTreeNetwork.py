import math
import torch
from torch import nn
from torch.nn import functional as F


class SyntaxTreeNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_nonterminal_rules, num_nonterminals, activation='tanh'):
        super().__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)

        nonterminal_rule_weights = [nn.Parameter(torch.Tensor(hidden_size, 2 * hidden_size))
                                    for _ in range(num_nonterminal_rules)]
        for w in nonterminal_rule_weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.nonterminal_rule_weights = nn.ParameterList(nonterminal_rule_weights)

        nonterminal_biases = [nn.Parameter(torch.zeros(hidden_size))
                              for _ in range(num_nonterminals)]
        self.nonterminal_biases = nn.ParameterList(nonterminal_biases)

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
        bias = self.nonterminal_biases[tree.node]
        if tree.is_preterminal:
            h = self.embedding(tree.children[0].node) + bias
        else:
            weight = self.nonterminal_rule_weights[tree.rule]
            h_children = [self.recursive_forward(c) for c in tree.children]
            h_agr = torch.cat(h_children, dim=1)
            h = F.linear(h_agr, weight, bias)
        h = self.activation(h)
        return h
