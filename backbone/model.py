import torch
import torch.nn as nn
from typing import Callable, List, Optional
import random
import torchvision
from torchvision.models import resnet18, regnet_y_1_6gf


class simpleMLP(nn.Module):
    def __init__(self,
                in_channels: int,
                hidden_channels: List[int],
                norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                inplace: Optional[bool] = None,
                bias: bool = True,
                dropout: float = 0.0,
                use_sigmoid = False,
                ):
        super(simpleMLP, self).__init__()
        params = {} if inplace is None else {"inplace": inplace}
        self.use_sigmoid = use_sigmoid
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))
        #sigmoid

        # layers.append(torch.nn.Linear(hidden_channels[-1], hidden_channels[-1], bias=bias))
        # layers.append(torch.nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)


    def forward(self, x):
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        # print(x.shape)
        return x



def MyResnet18(pretrained=True,num_classes = 1000,in_channels = 3):
    model = resnet18(pretrained=pretrained)

    if num_classes!=1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    if in_channels!=3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model

def Myregnet16(pretrained=True,num_classes = 1000,in_channels = 3):
    model = regnet_y_1_6gf(pretrained=pretrained)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    if in_channels!=3:
        model.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
    #
    return model


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)  # 输出层有10个类别

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = Myregnet16(pretrained=True,num_classes = 18,in_channels = 1)
    sample = torch.randn([512, 1, 28, 28])
    # weights = ResNet18_Weights.DEFAULT
    # preprocess = weights.transforms()
    # print(preprocess)
    out = model(sample)
    print(out.shape)
    # print(model)
