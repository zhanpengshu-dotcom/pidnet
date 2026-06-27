# ------------------------------------------------------------------------------
# PIDNet v1: 基线模型（原始PIDNet，无SPM，无ODConv）
# 用于消融实验A：PIDNet-S baseline
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
from .hdp_modules_v3a import BMHR
import logging

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False



class PIDNet(nn.Module):

    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96,
                 head_planes=128, augment=True, use_bmhr=False,
                 bmhr_reduction=4, bmhr_use_boundary_gate=True,
                 bmhr_zero_init=True):
        super(PIDNet, self).__init__()
        self.augment = augment

        # I Branch
        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # P Branch
        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 2, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 2, momentum=bn_mom),
                                          )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes * 2, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        if use_bmhr:
            self.bmhr = BMHR(
                channels=planes * 4,
                reduction=bmhr_reduction,
                use_boundary_gate=bmhr_use_boundary_gate,
                zero_init=bmhr_zero_init
            )
        else:
            self.bmhr = nn.Identity()

        # Prediction Head
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if use_bmhr and bmhr_zero_init:
            nn.init.constant_(self.bmhr.expand.block[1].weight, 0)
            nn.init.constant_(self.bmhr.expand.block[1].bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)

        return layer

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(
                        self.diff3(x),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)
        if self.augment:
            temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(
                        self.diff4(x),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)
        if self.augment:
            temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        if isinstance(self.bmhr, nn.Identity):
            x_ = self.bmhr(x_)
        else:
            x_ = self.bmhr(x_, x_d)
        x = F.interpolate(
                        self.spp(self.layer5(x)),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)

        x_ = self.final_layer(self.dfm(x_, x, x_d))

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_

def _load_pretrained(model, pretrained_dict):
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']

    model_dict = model.state_dict()
    normalized = {}
    for key, value in pretrained_dict.items():
        clean_key = key[6:] if key.startswith('model.') else key
        normalized[clean_key] = value

    loaded = {}
    checkpoint_not_used = []
    shape_mismatch = []
    for key, value in normalized.items():
        if key not in model_dict:
            checkpoint_not_used.append(key)
        elif value.shape != model_dict[key].shape:
            shape_mismatch.append(
                '{}: checkpoint {} vs model {}'.format(
                    key, tuple(value.shape), tuple(model_dict[key].shape)
                )
            )
        else:
            loaded[key] = value

    model_not_loaded = [key for key in model_dict.keys() if key not in loaded]
    expected_random_init = [
        key for key in model_not_loaded
        if key.startswith('bmhr.')
        or key.startswith('seghead_p.conv2.')
        or key.startswith('final_layer.conv2.')
    ]

    logging.info('Attention!!!')
    logging.info('Loaded parameter tensors: {}'.format(len(loaded)))
    logging.info('Checkpoint keys not used: {}'.format(len(checkpoint_not_used)))
    logging.info(checkpoint_not_used)
    logging.info('Shape mismatched tensors: {}'.format(len(shape_mismatch)))
    logging.info(shape_mismatch)
    logging.info('Model tensors not loaded: {}'.format(len(model_not_loaded)))
    logging.info(model_not_loaded)
    logging.info('Expected random initialized tensors: {}'.format(
        len(expected_random_init)
    ))
    logging.info(expected_random_init)
    logging.info('Over!!!')

    model_dict.update(loaded)
    model.load_state_dict(model_dict, strict=False)


def get_seg_model(cfg, imgnet_pretrained):
    use_bmhr = cfg.MODEL.USE_BMHR
    bmhr_reduction = cfg.MODEL.BMHR_REDUCTION
    bmhr_use_boundary_gate = cfg.MODEL.BMHR_USE_BOUNDARY_GATE
    bmhr_zero_init = cfg.MODEL.BMHR_ZERO_INIT

    if 's' in cfg.MODEL.NAME:
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES,
                       planes=32, ppm_planes=96, head_planes=128,
                       augment=True, use_bmhr=use_bmhr,
                       bmhr_reduction=bmhr_reduction,
                       bmhr_use_boundary_gate=bmhr_use_boundary_gate,
                       bmhr_zero_init=bmhr_zero_init)
    elif 'm' in cfg.MODEL.NAME:
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES,
                       planes=64, ppm_planes=96, head_planes=128,
                       augment=True, use_bmhr=use_bmhr,
                       bmhr_reduction=bmhr_reduction,
                       bmhr_use_boundary_gate=bmhr_use_boundary_gate,
                       bmhr_zero_init=bmhr_zero_init)
    else:
        model = PIDNet(m=3, n=4, num_classes=cfg.DATASET.NUM_CLASSES,
                       planes=64, ppm_planes=112, head_planes=256,
                       augment=True, use_bmhr=use_bmhr,
                       bmhr_reduction=bmhr_reduction,
                       bmhr_use_boundary_gate=bmhr_use_boundary_gate,
                       bmhr_zero_init=bmhr_zero_init)

    if imgnet_pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        _load_pretrained(model, pretrained_state)
    else:
        pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        _load_pretrained(model, pretrained_dict)

    return model
