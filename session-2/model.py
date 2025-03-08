import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, mlp_width):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        mlp_in = 16 * 16 * 64
        mlp_out = 15
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_width),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_width, mlp_out),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c*h*w)
        x = self.mlp(x)
        return x
