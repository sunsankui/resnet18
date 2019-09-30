import torchvision
import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        '''

        :param ch_in: 输入通道数
        :param ch_out: 输出通道数
        :param stride: 步长
        '''
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1   = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2   = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """

        :param self:
        :param x:[b,c,h,w]
        :return: x
        """
        out = F.relu(self.bn1(self.conv1(x))) #out->[4,32,25,25]
        # print(out.shape)
        out = self.bn2(self.conv2(out))
        # print(out.shape)
        out = self.extra(x) + out
        out = F.relu(out)
        return out
class ResNet18(nn.Module):
    def __init__(self,num_class):

        super(ResNet18,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=3,padding=0),
            nn.BatchNorm2d(16)
        )
        self.blk1 = ResBlock(16,32,stride=3)
        self.blk2 = ResBlock(32,64,stride=3)
        self.blk3 = ResBlock(64,128,stride=2)
        self.blk4 = ResBlock(128,256,stride=2)

        self.outlayer = nn.Linear(256*3*3,num_class)
    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = x.view(x.size(0),-1)
        x = self.outlayer(x)
        return x

def main():
    x = torch.randn(4,3,224,224)
    # rsl = ResBlock(3,16,2)
    # print(rsl(x).shape)
    net = ResNet18(5)
    print(net(x).shape)
    # for num in net.parameters():
    #     print(num,num.numel())
    # p = sum(map(lambda p:p.numel(),net.parameters()))
    # print("parameters size:",p)
if __name__ == '__main__':
    main()