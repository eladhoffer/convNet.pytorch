import torch
import torch.nn as nn

__all__ = ['mnist']

class mnist_model(nn.Module):

    def __init__(self):
        super(mnist_model, self).__init__()
        self.feats = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3,  1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3,  1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )

        self.classifier = nn.Conv2d(128, 10, 1)
        self.avgpool = nn.AvgPool2d(6, 6)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        out = self.feats(inputs)
        out = self.dropout(out)
        out = self.classifier(out)
        out = self.avgpool(out)
        out = out.view(-1, 10)
        return out


def mnist(**kwargs):
    return mnist_model()
