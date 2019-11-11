import math
import torch
from torch import nn
from torch.nn import functional as F
from models.GRU import GRUDecode


class DecoderSyntaxTreeGRU(nn.Module):

    def __init__(self, output_size, hidden_size,
                 num_nonterminal_rules, num_terminal_rules, num_nonterminals):
        super().__init__()

        nonterminal_rules = [GRUDecode(hidden_size) for _ in range(num_nonterminal_rules)]
        self.nonterminal_rules = nn.ModuleList(nonterminal_rules)

        terminal_rule_weights = [nn.Parameter(torch.Tensor(output_size, hidden_size))
                                 for _ in range(num_terminal_rules)]
        for w in terminal_rule_weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.terminal_rule_weights = nn.ParameterList(terminal_rule_weights)

        nonterminal_biases = [nn.Parameter(torch.zeros(hidden_size))
                              for _ in range(num_nonterminals)]
        self.nonterminal_biases = nn.ParameterList(nonterminal_biases)

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
            nonterminal_rule = self.nonterminal_rules[tree.rule]
            bias = [self.nonterminal_biases[c.node] for c in tree.children]
            output = []
            h_children = nonterminal_rule(hidden)
            h_children = [h_c + b for h_c, b in zip(h_children, bias)]
            for c, h_c in zip(tree.children, h_children):
                output += self.recursive_forward(c, h_c)
        return output
