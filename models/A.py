import torch
from torch import nn

import timm

from .CBAM import *

from utils.FCRDCT import *

class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input):
        return self.main(input)

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        return self.main(input)

class Attributor(nn.Module):
    def __init__(self, image_size, n_classes):
        super(Attributor, self).__init__()

        self.mask = CBAM(inplanes=3, planes=32)

        nf = 64
        nc = 3
        self.main = nn.Sequential(
            dcgan_conv(nc, nf),
            vgg_layer(nf, nf),

            dcgan_conv(nf, nf * 2),
            vgg_layer(nf * 2, nf * 2),

            dcgan_conv(nf * 2, nf * 4),
            vgg_layer(nf * 4, nf * 4),

            dcgan_conv(nf * 4, nf * 8),
            vgg_layer(nf * 8, nf * 8),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(nf * 8, n_classes, bias=True)
        )

    def forward(self, x):
        x = self.mask(x) * x
        
        embedding = self.main(x)
        feature = self.pool(embedding)
        feature = feature.view(feature.shape[0], -1)
        cls_output = self.classification_head(feature)

        return cls_output

    def forward_features(self, x):
        x = self.mask(x) * x
        
        embedding = self.main(x)
        feature = embedding.view(embedding.shape[0], -1)
        feature = self.pool(embedding)
        feature = feature.view(feature.shape[0], -1)

        return feature
        
    def get_mask(self, x):
        return (self.mask(x) + 1.0) / 2.0

    def get_masked(self, x):
        masked = self.mask(x) * x
        return masked