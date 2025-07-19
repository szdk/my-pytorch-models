
import torch.nn as nn

# window_size = 1125
class Cnn1_1125(nn.Module):
    def __init__(self, outputLayers):
        super(Cnn1_1125, self).__init__()

        #nn.Conv2d(1, 8, kernel_size=(3, 6), stride=1, padding=(1, 0)),
        # nn.ReLU(),
        # nn.MaxPool2d(),
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),   # [B, 16, 1125, 6]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1)),                        # [B, 16, 375, 6]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),             # [B, 32, 375, 6]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 1)),                        # [B, 32, 75, 6]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),             # [B, 64, 75, 6]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 2))                         # [B, 64, 15, 3]
        )

        self.fc = nn.Sequential(
            nn.Flatten(),              # [B, 64*15*3 = 2880]
            nn.Linear(2880, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, outputLayers)          # Final output: [B, outputLayers]
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x
