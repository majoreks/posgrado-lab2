import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
        )

        self.mlp = nn.Sequential(
            nn.Linear(16 * 16 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 15),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c*h*w)
        x = self.mlp(x)
        return x
