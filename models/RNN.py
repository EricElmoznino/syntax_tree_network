import torch
from torch import nn


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, activation='tanh'):
        super().__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, nonlinearity=activation)

    def forward(self, tree):
        def recursive_words(t):
            if t.is_terminal:
                return [self.embedding(t.node)]
            w = []
            for c in t.children:
                w += recursive_words(c)
            return w
        words = torch.stack(recursive_words(tree))

        _, h = self.rnn(words)
        h = h[-1]
        return h
