from models.vgg import *
from models.resnet import *


def networks(network, **kwargs):
    if network == 'vgg':
        return vgg(19, **kwargs)
    elif network == 'vggcam':
        return vggcam(19, **kwargs)
