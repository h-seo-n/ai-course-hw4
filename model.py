import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """
    LeNet-5 for CIFAR-10
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=400)
        self.fc2 = nn.Linear(in_features=400, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=84)
        self.fc4 = nn.Linear(in_features=84, out_features=num_classes)
    
        self.activation = nn.Tanh()

    def forward(self, x):
        # Conv layer 1
        x = self.pool1(self.activation(self.conv1(x)))

        # Conv layer 2
        x = self.pool2(self.activation(self.conv2(x)))

        # Flattening (16*5*5 -> 400)
        x = torch.flatten(x, 1)

        # Linear Layer
        x = self.activation(self.fc1(x))

        # Linear Layer 2
        x = self.activation(self.fc2(x))

        # Linear Layer 3
        x = self.activation(self.fc3(x))

        # Final Linear Layer
        x = self.fc4(x)

        return x


class ResidualBlock(nn.Module):
    """
    Residual Block for ResNet
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.stride = stride

        self.block = nn.Sequential(
            # First
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Second
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # skip connection
        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.downsample(x)
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out
        

class SimplifiedResNet(nn.Module):
    """
    Simplified ResNet for CIFAR-10
    """
    def __init__(self, num_classes=10, num_blocks=3):
        super(SimplifiedResNet, self).__init__()
        
        # Initial Conv layer
        self.in_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        # Residual layer
        self.layer1 = self._make_layer(16, num_blocks=1, stride=1)
        self.layer2 = self._make_layer(32, num_blocks=1, stride=2)
        self.layer3 = self._make_layer(64, num_blocks=1, stride=2)

        # Final layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
