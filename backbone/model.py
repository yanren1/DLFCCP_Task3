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


class SparseAutoencoder(nn.Module):
    def __init__(self,
                in_channels: int,
                encoder_channels: List[int],
                decoder_channels: List[int],
                norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                inplace: Optional[bool] = None,
                bias: bool = True,
                dropout: float = 0.0,
                sparsity_target=0.1,

                ):
        super(SparseAutoencoder, self).__init__()
        params = {} if inplace is None else {"inplace": inplace}

        self.sparsity_target = sparsity_target

        encoder_layers = []
        in_dim = in_channels
        for hidden_dim in encoder_channels:
            encoder_layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                encoder_layers.append(norm_layer(hidden_dim))
            encoder_layers.append(activation_layer(**params))
            encoder_layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = encoder_channels[-1]
        for hidden_dim in decoder_channels[:-1]:
            decoder_layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                decoder_layers.append(norm_layer(hidden_dim))
            decoder_layers.append(activation_layer(**params))
            decoder_layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        decoder_layers.append(torch.nn.Linear(in_dim, decoder_channels[-1], bias=bias))
        decoder_layers.append(torch.nn.Dropout(dropout, **params))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # print(x.shape)
        x = x.view(x.size(0), -1)
        encoder = self.encoder(x)
        out = self.decoder(encoder)

        # print(x.shape)
        return out,encoder

    def compute_sparsity_loss(self, activation):
        # Compute the average activation of each hidden unit
        average_activation = torch.mean(activation, dim=0)

        # Sparsity loss using Kullback-Leibler (KL) divergence
        sparsity_loss = torch.sum(self.sparsity_target * torch.log(self.sparsity_target / average_activation) +
                                  (1 - self.sparsity_target) * torch.log(
            (1 - self.sparsity_target) / (1 - average_activation)))

        return sparsity_loss


class CNNAutoencoder(nn.Module):
    def __init__(self,
                in_channels: int,
                encoder_channels: List[int],
                decoder_channels: List[int],
                num_features: int,
                num_classes: int,
                norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                inplace: Optional[bool] = None,
                dropout: float = 0.0,

                ):
        super(CNNAutoencoder, self).__init__()
        params = {} if inplace is None else {"inplace": inplace}

        encoder_layers = []
        in_dim = in_channels
        for hidden_dim in encoder_channels:
            encoder_layers.append(nn.Conv2d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
            if norm_layer is not None:
                encoder_layers.append(norm_layer(hidden_dim))
            encoder_layers.append(activation_layer(**params))
            encoder_layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = encoder_channels[-1]
        for hidden_dim in decoder_channels:
            decoder_layers.append(nn.ConvTranspose2d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
            if norm_layer is not None:
                decoder_layers.append(norm_layer(hidden_dim))
            decoder_layers.append(activation_layer(**params))
            decoder_layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim
        decoder_layers.append(nn.Conv2d(in_dim, in_channels, kernel_size=3, stride=1, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(decoder_channels[-1] * 28 * 28, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        # print(x.shape)
        # x = self.classifier(x)

        return x



if __name__ == '__main__':
    model = CNNAutoencoder(in_channels=3,
                encoder_channels=[16,32],
                decoder_channels=[32,16],
                num_features=1024,
                num_classes=10,
                norm_layer= nn.BatchNorm2d,
                dropout=0.1)
    sample = torch.randn([32, 3, 28, 28])
    out = model(sample)
    print(out.shape)
