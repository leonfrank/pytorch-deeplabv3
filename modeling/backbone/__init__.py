from modeling.backbone import resnet
# from modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride)
    # elif backbone == 'xception':
        # return xception.AlignedXception(output_stride)
    # elif backbone == 'drn':
        # return drn.drn_d_54()
    # elif backbone == 'mobilenet':
        # return mobilenet.MobileNetV2(output_stride)
    else:
        raise NotImplementedError
