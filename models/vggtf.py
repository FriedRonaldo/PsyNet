import torch.nn as nn
import torchvision.models as vmodels

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vggcam16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vggcam19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'N', 512, 512, 512, 512, 'N'],
}


class VGGTF(nn.Module):
    def __init__(self, features, tftypes=['rotation', 'translation', 'shear', 'hflip'], tfnums=[4, 3, 3, 2], with_cam=False):
        super(VGGTF, self).__init__()
        # parameters setting
        self.num_cls = tfnums
        self.blocks = nn.ModuleList()
        self.with_cam = with_cam
        print('USE TF IN NET:\t', tftypes)
        print("USE TF NUMS:\t", self.num_cls)

        chinter = 512

        if self.with_cam:
            if 'odd' in tftypes:
                print('APPEND ODD')
                self.odd = nn.Conv2d(chinter, self.num_cls[7], 1, 1)
                self.blocks.append(self.odd)
            if 'rotation' in tftypes:
                print('APPEND ROTATION')
                self.rotation = nn.Conv2d(chinter, self.num_cls[0], 1, 1)
                self.blocks.append(self.rotation)
            if 'translation' in tftypes:
                print('APPEND TRANSLATION')
                self.translation = nn.Conv2d(chinter, self.num_cls[1], 1, 1)
                self.blocks.append(self.translation)
            if 'shear' in tftypes:
                print('APPEND SHEAR')
                self.shear = nn.Conv2d(chinter, self.num_cls[2], 1, 1)
                self.blocks.append(self.shear)
            if 'hflip' in tftypes:
                print('APPEND HFLIP')
                self.hflip = nn.Conv2d(chinter, self.num_cls[3], 1, 1)
                self.blocks.append(self.hflip)
            if 'scale' in tftypes:
                print('APPEND SCALE')
                self.scale = nn.Conv2d(chinter, self.num_cls[4], 1, 1)
                self.blocks.append(self.scale)
            if 'vflip' in tftypes:
                print('APPEND VFLIP')
                self.vflip = nn.Conv2d(chinter, self.num_cls[5], 1, 1)
                self.blocks.append(self.vflip)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            if 'vtranslation' in tftypes:
                print('APPEND VTRANSLATION')
                self.vtranslation = nn.Conv2d(chinter, self.num_cls[6], 1, 1)
                self.blocks.append(self.vtranslation)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if 'rotation' in tftypes:
                print('APPEND ROTATION')
                self.rotation = nn.Linear(chinter, self.num_cls[0])
                self.blocks.append(self.rotation)
            if 'translation' in tftypes:
                print('APPEND TRANSLATION')
                self.translation = nn.Linear(chinter, self.num_cls[1])
                self.blocks.append(self.translation)
            if 'shear' in tftypes:
                print('APPEND SHEAR')
                self.shear = nn.Linear(chinter, self.num_cls[2])
                self.blocks.append(self.shear)
            if 'hflip' in tftypes:
                print('APPEND HFLIP')
                self.hflip = nn.Linear(chinter, self.num_cls[3])
                self.blocks.append(self.hflip)
            if 'scale' in tftypes:
                print('APPEND SCALE')
                self.scale = nn.Linear(chinter, self.num_cls[4])
                self.blocks.append(self.scale)

        self._initialize_weights()

        self.features = features
        self.features_1 = features[0:40]
        self.features_2 = features[40:]

    def forward(self, x):
        logits = []
        cams = []
        # x = self.features(x)
        # x = self.inter(x)
        # print(self.features)
        # exit()
        relu1 = self.features_1(x)
        x = self.features_2(relu1)
        if self.with_cam:
            chatt = x.mean(1)
            for block in self.blocks:
                cam = block(x)
                cams.append(cam)
                logit = self.pool(cam).squeeze()
                logits.append(logit)
            cams.append(relu1.mean(1))
            cams.append(chatt)
            return logits, cams

        else:
            flat = x.view(x.size(0), -1)

            for block in self.blocks:
                logit = block(flat)
                logits.append(logit)
            return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_ch=3, batch_norm=False):
    layers = []
    in_channels = in_ch
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, momentum=0.001), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vggcamtf(vggconfig, tftypes, tfnums, pretrained, **kwargs):
    use_bn = ('bn' in vggconfig)

    if pretrained:
        print('USE PRETRAINED BACKBONE')
        model = VGGTF(make_layers(cfg[vggconfig[:8]], batch_norm=use_bn),
                      tftypes=tftypes, tfnums=tfnums, with_cam=True)

        model_dict = model.state_dict()
        if vggconfig == 'vggcam16':
            vgg = vmodels.vgg16(pretrained=True)
        elif vggconfig == 'vggcam19':
            vgg = vmodels.vgg19(pretrained=True)
        elif vggconfig == 'vggcam16bn':
            vgg = vmodels.vgg16_bn(pretrained=True)
        elif vggconfig == 'vggcam19bn':
            vgg = vmodels.vgg19_bn(pretrained=True)
        else:
            print("NOT IMPLEMENTED FOR ", vggconfig)
            exit(-3)

        pretrained_dict = vgg.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    else:
        model = VGGTF(features=make_layers(cfg[vggconfig[:8]], batch_norm=use_bn),
                      tftypes=tftypes, tfnums=tfnums, with_cam=True)
    return model