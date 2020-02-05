import torch
import torch.nn as nn
import torchvision.models as vmodels
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_att_map


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vggimg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'N'],
    'vggcam16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'vggcam19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'N', 512, 512, 512, 512, 'N'],
}


class VGG(nn.Module):
    def __init__(self, features, init_weights=False, dataset='imagenet'):
        super(VGG, self).__init__()
        # parameters setting
        classes = {'imagenet': 1000}
        self.dataset = dataset
        num_classes = classes[self.dataset.lower()]

        # network layers setting
        self.features = features

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.maxpool = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(nn.Linear(7 * 7 * 512, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout2d(0.5),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout2d(0.5),
                                        nn.Linear(4096, num_classes))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        chatt = out
        out = self.maxpool(out)
        out = self.pool(out)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        return out, [chatt]

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


class VGGCAM(nn.Module):
    def __init__(self, features, init_weights=False, dataset='CUB', method='CAM', th=0.6):
        super(VGGCAM, self).__init__()
        # parameters setting
        classes = {'cub': 200}
        self.dataset = dataset.lower()
        num_classes = classes[self.dataset]
        self.method = method.lower()
        self.th = th

        if method == 'cam':
            self.inter = nn.Conv2d(512, 1024, 1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(nn.Dropout2d(), nn.Linear(1024, num_classes))
        elif method == 'acol1':
            self.cls = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(1024, 1024, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(1024, num_classes, 1))
            self.cls_erase = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1),
                                           nn.ReLU(),
                                           nn.Conv2d(1024, 1024, 3, 1, 1),
                                           nn.ReLU(),
                                           nn.Conv2d(1024, num_classes, 1))
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif method == 'acolcam':
            self.classifier = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(1024, 1024, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(1024, num_classes, 1))
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif method == 'adl':
            print('NOT IMPLEMENTED ADL')
            exit(-3)
        elif method == 'none':
            self.pool = nn.AdaptiveAvgPool2d((7, 7))

            self.classifier = nn.Sequential(nn.Linear(7 * 7 * 512, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout2d(0.5),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout2d(0.5),
                                            nn.Linear(4096, num_classes))

        if init_weights:
            self._initialize_weights()

        self.features = features

    def forward(self, x, y_in=None):
        x = self.features(x)

        if self.method == 'cam':
            chatt = x.mean(1)
            out = self.inter(x)
            out = self.pool(out).squeeze()
            logit = self.classifier(out)
            return logit, [chatt]

        elif self.method == 'acolcam':
            chatt = x
            out = self.classifier(x)
            logit = self.pool(out).squeeze()
            return logit, [chatt]

        elif self.method == 'acol1':
            # attention map and backbone
            attmap = x.mean(1)

            feat = F.avg_pool2d(x, 3, 1, 1)

            # branch A
            out = self.cls(feat)
            logit_org = self.pool(out).squeeze()

            # erase
            localization_map_normed = norm_att_map(attmap)
            feat_erase, mask = self.erase_feature_maps(localization_map_normed, feat, self.th)

            # branch B
            out_erase = self.cls_erase(feat_erase)
            logit_erase = self.pool(out_erase).squeeze()

            return [logit_org, logit_erase], [mask, localization_map_normed]
        elif self.method == 'adl':
            print('NOT IMPLEMENTED ADL')
            exit(-3)
        elif self.method == 'none':
            attmap = x.mean(1)
            out = self.pool(x).view(x.size(0), -1)
            logit = self.classifier(out)
            return logit, [attmap]

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

    def erase_feature_maps(self, atten_map_normed, feature_maps, threshold):
        # atten_map_normed = torch.unsqueeze(atten_map_normed, dim=1)
        # atten_map_normed = self.up_resize(atten_map_normed)
        if len(atten_map_normed.size())>3:
            atten_map_normed = torch.squeeze(atten_map_normed)
        atten_shape = atten_map_normed.size()

        pos = torch.ge(atten_map_normed, threshold)
        mask = torch.ones(atten_shape).cuda()
        mask[pos.data] = 0.0
        mask = torch.unsqueeze(mask, dim=1)
        #erase
        erased_feature_maps = feature_maps * Variable(mask)

        return erased_feature_maps, mask


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
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


def make_layers_imagenet(cfg, batch_norm=False):
    layers = []
    in_channels = 3
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


def vgg(depth, **kwargs):
    model = VGG(make_layers(cfg['vgg'+str(depth)], batch_norm=False), **kwargs)
    return model


def vggimg(vggconfig='vggimg16', dataset='imagenet', **kwargs):
    use_bn = ('bn' in vggconfig)
    model = VGG(make_layers_imagenet(cfg[vggconfig], use_bn), dataset=dataset, **kwargs)
    model_dict = model.state_dict()
    if vggconfig == 'vggimg16':
        vgg = vmodels.vgg16(pretrained=True)
    else:
        print('NOT IMPLEMENTED', vggconfig)
        exit(-3)
    pretrained_dict = vgg.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def vggcam(pretrained, vggconfig='vggcam16', method='cam', **kwargs):
    use_bn = ('bn' in vggconfig)

    if pretrained:
        model = VGGCAM(make_layers(cfg[vggconfig[:8]], batch_norm=use_bn), method=method, **kwargs)
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
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'features' in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        model = VGGCAM(make_layers(cfg[vggconfig[:8]], use_bn), **kwargs)
    return model
