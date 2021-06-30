import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets
import torch.distributed as dist

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

num_classes = 4000

class Block(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        if self.downsample != None:
            residual = self.downsample(x)
        else:
            residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu2(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_planes):
        super(ResNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(Block, 32, 3)
        self.layer2 = self.make_layer(Block, 64, 4, stride=2)
        self.layer3 = self.make_layer(Block, 128, 12, stride=2)
        self.layer4 = self.make_layer(Block, 256, 3, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * Block.expansion, num_classes)
    
    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(epoch):
    print("Training... Epoch = %d" % epoch)
    for video, s, target in data_loader:
        # print(video[0].shape, s.shape, target.shape)
        # print(video[0])
        idata = video[0].permute(0, 3, 1, 2)[:,:,0:224,0:256]
        # print(idata.shape)
        pred = model(idata)
        # print(pred.shape)
        loss = nllloss(pred, target)

        optimizer4nn.zero_grad()
        loss.backward()
        optimizer4nn.step()

kinetics_data = datasets.Kinetics400('.data',frames_per_clip=1, step_between_clips=5,
                 extensions=('mp4',))
data_loader = DataLoader(kinetics_data, batch_size=1, shuffle=True)

# Model
model = ResNet(3)

# NLLLoss
nllloss = nn.NLLLoss() #CrossEntropyLoss = log_softmax + NLLLoss


# optimzer4nn
optimizer4nn = optim.SGD(model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)

for epoch in range(100):
    # print optimizer4nn.param_groups[0]['lr']
    train(epoch+1)
    sheduler.step()