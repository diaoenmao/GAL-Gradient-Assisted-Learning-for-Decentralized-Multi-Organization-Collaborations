import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, normalize, feature_split
from config import cfg


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(Block, self).__init__()
        self.n1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride):
        super(Bottleneck, self).__init__()
        self.n1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.n2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(out)))
        out = self.conv3(F.relu(self.n3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2)
        self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion)
        self.linear = nn.Linear(hidden_size[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        output = {}
        x = input[cfg['data_tag']]
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.n4(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        output['score'] = out
        if 'assist' in input:
            if self.training:
                if input['assist'] is None:
                    target = F.one_hot(input['label'], cfg['classes_size']).float()
                    target[target == 0] = 1e-4
                    target = torch.log(target)
                    output['loss_local'] = F.mse_loss(output['score'], target)
                    output['loss'] = F.cross_entropy(output['score'], input['label'])
                else:
                    input['assist'].requires_grad = True
                    loss = F.cross_entropy(input['assist'], input['label'], reduction='sum')
                    loss.backward()
                    target = copy.deepcopy(input['assist'].grad)
                    output['loss_local'] = F.mse_loss(output['score'], target)
                    input['assist'] = input['assist'].detach()
                    output['score'] = input['assist'] - cfg['assist_rate'] * output['score']
                    output['loss'] = F.cross_entropy(output['score'], input['label'])
            else:
                output['score'] = input['assist']
                output['loss'] = F.cross_entropy(output['score'], input['label'])
        else:
            output['loss'] = F.cross_entropy(output['score'], input['label'])
        return output


def resnet18():
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = cfg['resnet']['hidden_size']
    model = ResNet(data_shape, hidden_size, Block, [1, 1, 1, 2], classes_size)
    model.apply(init_param)
    return model


def resnet34():
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = cfg['resnet']['hidden_size']
    model = ResNet(data_shape, hidden_size, Block, [3, 4, 6, 3], classes_size)
    model.apply(init_param)
    return model


def resnet50():
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = cfg['resnet']['hidden_size']
    model = ResNet(data_shape, hidden_size, Bottleneck, [3, 4, 6, 3], classes_size)
    model.apply(init_param)
    return model


def resnet101():
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = cfg['resnet']['hidden_size']
    model = ResNet(data_shape, hidden_size, Bottleneck, [3, 4, 23, 3], classes_size)
    model.apply(init_param)
    return model


def resnet152():
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = cfg['resnet']['hidden_size']
    model = ResNet(data_shape, hidden_size, Bottleneck, [3, 8, 36, 3], classes_size)
    model.apply(init_param)
    return model