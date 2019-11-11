import torch
from torch import nn
from models.GRU import GRUEncode


class SyntaxTreeGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_nonterminal_rules, num_nonterminals):
        super().__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)

        nonterminal_rules = [GRUEncode(hidden_size) for _ in range(num_nonterminal_rules)]
        self.nonterminal_rules = nn.ModuleList(nonterminal_rules)

        nonterminal_biases = [nn.Parameter(torch.Tensor(hidden_size))
                              for _ in range(num_nonterminals)]
        self.nonterminal_biases = nn.ParameterList(nonterminal_biases)

    def forward(self, tree):
        return self.recursive_forward(tree)

    def recursive_forward(self, tree):
        bias = self.nonterminal_biases[tree.node]
        if tree.is_preterminal:
            h = self.embedding(tree.children[0].node) + bias
            h = torch.tanh(h)
        else:
            nonterminal_rule = self.nonterminal_rules[tree.rule]
            h_children = [self.recursive_forward(c) for c in tree.children]
            h = nonterminal_rule(*h_children) + bias
        return h
