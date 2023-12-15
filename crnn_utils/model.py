import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, img_height, img_channels, num_classes, hidden_size=256):
        super(CRNN, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more convolutional layers as needed
        )

        # RNN layers
        self.rnn = nn.LSTM(64, hidden_size, bidirectional=True, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # CNN forward pass
        x = self.cnn(x)

        # Prepare data for RNN
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)

        # RNN forward pass
        x, _ = self.rnn(x)

        # Fully connected layer
        x = self.fc(x)

        return x
