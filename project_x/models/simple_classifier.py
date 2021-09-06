import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_shape: int, num_classes: int):
        super(Model, self).__init__()
        self._features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        n_sizes = self._get_conv_output(input_shape)
        self._linear = nn.Sequential(
            nn.Linear(n_sizes, 512),
            nn.Linear(512, 128),
            nn.Linear(128, num_classes)
        )

    def _get_conv_output(self, shape: int) -> int:
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x: torch.Tensor()) -> torch.Tensor():
        if self.training:
            x = self._features(x)
            x = x.view(x.size(0), -1)
            x = F.log_softmax(self._linear(x))
            return x
        else:
            x = self._features(x)
            x = x.view(x.size(0), -1)
            x = F.log_softmax(self._linear(x))
            return x