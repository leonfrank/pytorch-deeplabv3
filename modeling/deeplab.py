import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling import ASPP, Decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone, output_stride, num_classes, freeze_bn = False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        self.backbone = build_backbone(backbone, output_stride)
        self.aspp = ASPP(backbone, output_stride)
        self.decoder = Decoder(backbone, num_classes)
        
        if freeze_bn:
            self.freeze_bn()
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x
        
if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())