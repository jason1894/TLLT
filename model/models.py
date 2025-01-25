"""This is scripts include the NN models"""
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models

from utils.random_seed import set_seed

# the Represent model, denconected NN
class Bottleneck(nn.Module):
    def __init__(self, nchannels, growthrate):
        super(Bottleneck, self).__init__()
        interchannels = 4*growthrate
        self.bn1 = nn.BatchNorm2d(nchannels)
        self.conv1 = nn.Conv2d(nchannels, interchannels, kernel_size=1, bias = False)
        
        self.bn2 = nn.BatchNorm2d(interchannels)
        self.conv2 = nn.Conv2d(interchannels, growthrate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out
    

class Singlelayer(nn.Module):
    def __init__(self, nchannels, growthrate):
        super(Singlelayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nchannels)
        self.conv1 = nn.Conv2d(nchannels, growthrate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x))) 
        out = torch.cat((x, out), 1)
        return out   


class Transition(nn.Module):
    def __init__(self, nchannles, noutchannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nchannles)
        self.conv1 = nn.Conv2d(nchannles, noutchannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class Densenet(nn.Module):
    def __init__(self, growthrate, depth, reduction, ndim, bottleneck):
        super(Densenet, self).__init__()

        ndenseblocks = (depth -4) // 3
        if bottleneck:
            ndenseblocks  //= 2

        nchannels = 2*growthrate
        self.conv1 = nn.Conv2d(1, nchannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nchannels, growthrate, ndenseblocks, bottleneck)
        nchannels += ndenseblocks*growthrate
        noutchannels = int(math.floor(nchannels*reduction))    
        self.trans1 = Transition(nchannles=nchannels, noutchannels=noutchannels)

        nchannels = noutchannels
        self.dense2 = self._make_dense(nchannels, growthrate, ndenseblocks, bottleneck)
        nchannels += ndenseblocks*growthrate
        noutchannels = int(math.floor(nchannels*reduction))
        self.trans2 = Transition(nchannles=nchannels, noutchannels=noutchannels)

        nchannels = noutchannels
        self.dens3 = self._make_dense(nchannels, growthrate, ndenseblocks, bottleneck)
        nchannels += ndenseblocks*growthrate

        self.bn1 = nn.BatchNorm2d(nchannels)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(nchannels,ndim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_() 

    def _make_dense(self, nchannesl, gropwthrate, ndenseblocks, bottleneck):
        layers = []
        for i in range(int(ndenseblocks)):
            if bottleneck:
                layers.append(Bottleneck(nchannels=nchannesl, growthrate= gropwthrate))
            else:
                layers.append(Singlelayer(nchannels=nchannesl, growthrate=gropwthrate))
            nchannesl += gropwthrate
        return nn.Sequential(*layers)    


    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dens3(out)
        out = self.avgpool(out).view(out.size(0), -1)
        latent = self.fc1(out)
        return latent
    
    def latdim(self):
        return self.fc1.out_features

# resnet18 backbone for real data office-Caltech 
class ResNetBackbone(nn.Module):
    def __init__(self, ndim, pretrained=False):
        super(ResNetBackbone, self).__init__()
        if pretrained:
            #default not load the pretrained weight
            print("Use the pretrained weight for resnet")
            resnet = models.resnet18(weights="IMAGENET1K_V1")
        else:    
            resnet = models.resnet18(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        self.ndim = ndim
        self.fc = nn.Linear(self._feature_dim, ndim)
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def output_num(self):
        return self.ndim  

# ------------------------------------------------------------------------------    
# The discriminator model in Represention process
class Discriminator(nn.Module):
    def __init__(self, ndim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(ndim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64,1)
        )

    def forward(self, x):
        validity = self.model(x)
        return validity



# ------------------------------------------------------------------------------
# The predict model
class Predict_h(nn.Module):
    def __init__(self, input, hidensize,  outclass):
        super(Predict_h, self).__init__()

        self.fc1 = nn.Linear(input, hidensize)
        self.fc2 = nn.Linear(hidensize, hidensize)
        self.fc3 = nn.Linear(hidensize, outclass)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    