import torch
import torch.nn as nn

class Four_conv(nn.Module):
    def __init__(self):
        super(Four_conv, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(2048,64),
            nn.ELU(inplace=True),
            nn.Dropout(.5),
            nn.Linear(64,5))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x1 = x.view(x.size(0), -1)
        x2 = self.fc_layers(x1)
        x2 = self.sigmoid(x2)
        return x1,x2