import torch
from torch import nn

# model definition
# use residential block
# 2D resnet input 200*161
# 5 residual block each block has 3 conv layers
class Bottleneck(nn.Module):
    # output channels = expansion * input channels
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, conv_nums=3, down_sample=None):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
    
    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu3(out)

        return out


# whole model for acoustic signal feature
class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, blocks_num=[3, 4, 6, 3, 3], num_classes=250, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 32

        self.conv1 = nn.Conv2d(1,self.in_channel, kernel_size=1,padding=0)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # stride=2 padding=1 size/2
        self.layer1 = self._make_layer(block, 32, block_num=blocks_num[0], stride=2)
        self.layer2 = self._make_layer(block, 64, block_num=blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 128, block_num=blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 256, block_num=blocks_num[3], stride=2)
        self.layer5 = self._make_layer(block, 512, block_num=blocks_num[4], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(in_features=512*block.expansion,out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d): # normal distribution initialize the weights
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        

    def _make_layer(self, block, channel, block_num=3, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(
            block(
                self.in_channel,
                channel,
                downsample=downsample,
                stride=stride,
            )
        )
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(
                    self.in_channel,
                    channel
                )
            )
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x