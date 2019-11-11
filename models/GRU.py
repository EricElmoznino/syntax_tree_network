from torch import nn


class GRUEncode(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.l_rl = nn.Linear(hidden_size, hidden_size)
        self.r_rl = nn.Linear(hidden_size, hidden_size)

        self.l_rr = nn.Linear(hidden_size, hidden_size)
        self.r_rr = nn.Linear(hidden_size, hidden_size)

        self.l_zl = nn.Linear(hidden_size, hidden_size)
        self.r_zl = nn.Linear(hidden_size, hidden_size)

        self.l_zr = nn.Linear(hidden_size, hidden_size)
        self.r_zr = nn.Linear(hidden_size, hidden_size)

        self.l_z = nn.Linear(hidden_size, hidden_size)
        self.r_z = nn.Linear(hidden_size, hidden_size)

        self.l = nn.Linear(hidden_size, hidden_size)
        self.r = nn.Linear(hidden_size, hidden_size)

    def forward(self, left, right):
        r_l = nn.Sigmoid()(self.l_rl(left) + self.r_rl(right))
        r_r = nn.Sigmoid()(self.l_rr(left) + self.r_rr(right))
        z_l = nn.Sigmoid()(self.l_zl(left) + self.r_zl(right))
        z_r = nn.Sigmoid()(self.l_zr(left) + self.r_zr(right))
        z = nn.Sigmoid()(self.l_z(left) + self.r_z(right))
        h_tilde = nn.Tanh()(self.l(r_l * left) + self.r(r_r * right))
        h = z_l * left + z_r * right + z * h_tilde
        return h


class GRUDecode(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.l_r = nn.Linear(hidden_size, hidden_size)
        self.r_r = nn.Linear(hidden_size, hidden_size)

        self.l_z = nn.Linear(hidden_size, hidden_size)
        self.r_z = nn.Linear(hidden_size, hidden_size)

        self.l_n = nn.Linear(hidden_size, hidden_size)
        self.r_n = nn.Linear(hidden_size, hidden_size)

    def forward(self, h):
        r_l = nn.Sigmoid()(self.l_r(h))
        z_l = nn.Sigmoid()(self.l_z(h))
        n_l = nn.Tanh()(r_l * self.l_n(h))
        h_l = (1 - z_l) * n_l + z_l * h

        r_r = nn.Sigmoid()(self.r_r(h))
        z_r = nn.Sigmoid()(self.r_z(h))
        n_r = nn.Tanh()(r_l * self.r_n(h))
        h_r = (1 - z_r) * n_r + z_r * h

        return h_l, h_r
