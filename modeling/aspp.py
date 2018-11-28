import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.diconv = nn.Conv2d(inplanes, planes, kernel_size, padding = padding, dilation = dilation, bias = False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x):
        x = self.diconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            
class _GlobalContext(nn.Module):
    def __init__(self, inplanes, planes):
        super(_GlobalContext, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.net = nn.Sequential(
        nn.Conv2d(inplanes, planes, 1, bias = False),
        nn.BatchNorm2d(planes),
        nn.ReLU(inplace = True),
        )
    def forward(self, x):
        size = x.size()[2:]
        x = self.pooling(x)
        x = self.net(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x
        
class ASPP(nn.Module):
    def __init__(self, backbone, output_stride):
        super(ASPP, self).__init__()
        self.output_stride = output_stride
        if self.output_stride == 16:
            rates = [1, 6, 12, 18]
        elif self.output_stride == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        if backbone == 'drn':
            self.inplanes = 512
        elif backbone == 'mobilenet':
            self.inplanes = 320
        elif backbone == 'resnet':
            self.inplanes = 2048
        else:
            raise NotImplementedError
        self.planes = 256
        self.aspp0 = _ASPPModule(self.inplanes, self.planes, 1, padding = 0, dilation = rates[0])
        self.aspp1 = _ASPPModule(self.inplanes, self.planes, 3, padding = rates[1], dilation = rates[1])
        self.aspp2 = _ASPPModule(self.inplanes, self.planes, 3, padding = rates[2], dilation = rates[2])
        self.aspp3 = _ASPPModule(self.inplanes, self.planes, 3, padding = rates[3], dilation = rates[3])
        self.global_context = _GlobalContext(self.inplanes, self.planes)
        self.bottleneck = nn.Sequential(
        nn.Conv2d(5 * self.planes, self.planes, 1, bias = False),
        nn.BatchNorm2d(self.planes),
        nn.ReLU(inplace = True)
        )
        self._init_weight()
        
    def forward(self, x):
        x1 = self.aspp0(x)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.global_context(x)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.bottleneck(x)
        return x
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

