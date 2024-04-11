import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x: torch.Tensor):
        res = self.fc1(x)
        res = torch.relu(res)
        res = self.fc2(res)
        #res = res.view(-1)
        return res
