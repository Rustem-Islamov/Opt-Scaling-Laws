import torch
import torch.nn as nn
import torch.nn.functional as F

class CifarNet(nn.Module):
    """
    A simple CNN model adapted for the CIFar-10 dataset (3x32x32 images).
    """
    def __init__(self):
        super(CifarNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
                # x = self.dropout1(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
                # x = self.dropout2(x)
        
        output = self.fc2(x)

        return output