import math
import torch
from torch import nn
from torch.nn import functional as F


class DecoderSyntaxTreeNetwork(nn.Module):

    def __init__(self, output_size, hidden_size,
                 num_nonterminal_rules, num_terminal_rules, num_nonterminals, activation='tanh'):
        super().__init__()

        self.hidden_size = hidden_size

        nonterminal_rule_weights = [nn.Parameter(torch.Tensor(hidden_size * 2,hidden_size))
                                    for _ in range(num_nonterminal_rules)]
        for w in nonterminal_rule_weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        terminal_rule_weights = [nn.Parameter(torch.Tensor(output_size, hidden_size))
                                 for _ in range(num_terminal_rules)]
        for w in terminal_rule_weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        nonterminal_biases = [nn.Parameter(torch.zeros(hidden_size))
                              for _ in range(num_nonterminals)]

        self.nonterminal_rule_weights = nn.ParameterList(nonterminal_rule_weights)
        self.terminal_rule_weights = nn.ParameterList(terminal_rule_weights)
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

    def forward(self, tree, encoding):
        predictions = self.recursive_forward(tree, encoding)
        predictions = torch.stack(predictions)
        if not self.training:
            predictions = F.softmax(predictions, dim=2)
        return predictions

    def recursive_forward(self, tree, hidden):
        if tree.is_preterminal:
            weight = self.terminal_rule_weights[tree.rule]
            output = [F.linear(hidden, weight)]
        else:
            weight = self.nonterminal_rule_weights[tree.rule]
            bias = [self.nonterminal_biases[c.node] for c in tree.children]
            bias = torch.cat(bias)
            output = []
            h_children = F.linear(hidden, weight, bias)
            h_children = self.activation(h_children)
            h_children = h_children.split(self.hidden_size, dim=1)
            for c, h_c in zip(tree.children, h_children):
                output += self.recursive_forward(c, h_c)
        return output
