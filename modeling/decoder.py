import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        
        squeeze = 48
        self.bottleneck = nn.Sequential(
        nn.Conv2d(low_level_inplanes, squeeze, 1, bias = False),
        nn.BatchNorm2d(squeeze),
        nn.ReLU(inplace = True)
        )
        planes = 256
        self.out = nn.Sequential(nn.Conv2d(planes + squeeze, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(),
                                nn.Conv2d(planes, num_classes, kernel_size=1, stride=1))
        self._init_weight()                        
    
    def forward(self, x, low_level_feat):
        size = low_level_feat.size()[2:]
        low_level_feat = self.bottleneck(low_level_feat)
        x = F.interpolate(x, size = size, mode = 'bilinear', align_corners = True)
        x = torch.cat((x, low_level_feat), dim = 1)
        x = self.out(x)
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        