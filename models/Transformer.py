from torch import nn


class Transformer(nn.Module):

    def __init__(self, hidden_size, n_layers=3, final_activation='tanh'):
        super().__init__()

        self.layers = []
        for i in range(n_layers - 1):
            self.layers += [nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True)]
        self.layers += [nn.Linear(hidden_size, hidden_size),
                        nn.BatchNorm1d(hidden_size)]

        if final_activation == 'tanh':
            self.layers += [nn.Tanh()]
        elif final_activation == 'sigmoid':
            self.layers += [nn.Sigmoid()]
        elif final_activation == 'relu':
            self.layers += [nn.ReLU(inplace=True)]
        elif final_activation == 'prelu':
            self.layers += [nn.PReLU()]
        else:
            raise ValueError('Unknown activation type: %s' % final_activation)

        self.layers = nn.Sequential(*self.layers)

    def forward(self, input):
        return self.layers(input)
