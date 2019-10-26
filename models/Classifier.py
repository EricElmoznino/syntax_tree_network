from torch import nn
from torch.nn import functional as F


class Classifier(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()

        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, input):
        x = self.linear(input)
        if not self.training:
            x = F.softmax(x, dim=1)
        return x
