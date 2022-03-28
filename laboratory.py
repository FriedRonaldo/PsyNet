import torch
import torch.nn as nn
import torchvision.models as vmodels
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_att_map
import numpy as np
import os
from bs4 import BeautifulSoup
import re

from dogs import Dogs
from datasetgetter import get_dataset
import argparse
import torchvision.transforms as transforms
from glob import glob
import cv2
from PIL import Image


parser = argparse.ArgumentParser(description='PyTorch Simultaneous Training')
parser.add_argument('--data_dir', default='../data/', help='path to dataset')
parser.add_argument('--dataset', default='DOGS', help='type of dataset', choices=['CUB', 'IMAGENET', 'CARS', 'DOGS'])
parser.add_argument('--model_name', type=str, default='LOC', help='model name')
parser.add_argument('--network', type=str, default='vggcam16',
                    choices=['vggcam16', 'vggcam16bn', 'vggcam19', 'vggcam19bn', 'vggimg16'])
parser.add_argument('--method', type=str, default='cam', choices=['cam', 'acol1', 'acol2', 'adl', 'acolcam', 'none'], help='')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=80, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Total batch size - e.g) num_gpus = 2 , batch_size = 128 then, effectively, 64')
parser.add_argument('--val_batch', default=32, type=int)
parser.add_argument('--image_size', default=224, type=int, help='Input image size')


args = parser.parse_args()


data_dir = '../data/AIRCRAFT/data'

img_list_path = os.path.join(data_dir, 'images_test.txt')
img_list = []

size_file = open('images_train_val.txt', 'w')

with open(img_list_path) as f:
    img_list = f.read().split()

for img_name in img_list:
    img_path = os.path.join(data_dir, 'images', img_name + '.jpg')
    img = Image.open(img_path)
    size_file.write("{} {} {}\n".format(img_name, img.size[0], img.size[1]))

# files = sorted(glob(img_dir + "/*.jpg"))

#
# for imgname in files:
#     img = cv2.imread(imgname)
#     size_file.write("{} {}\n".format(img.shape[1], img.shape[0]))

size_file.close()
print('DONE')

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# transforms_train = transforms.Compose([transforms.Resize((224, 224)),
#                                        transforms.ToTensor(),
#                                        normalize])
#
# _, dataset = get_dataset('DOGS', args)
#
# val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.val_batch,
#                                          num_workers=args.workers, shuffle=False)
#
#
# val_iter = iter(val_loader)
#
# bboxes = dataset.load_bboxes()
#
# x_in, y, name = next(val_iter)
#
# means = [0.485, .456, .406]
# stds = [.229, .224, .225]
# means = torch.reshape(torch.tensor(means), (1, 3, 1, 1))
# stds = torch.reshape(torch.tensor(stds), (1, 3, 1, 1))
#
# x_in_ = (x_in * stds + means) * 255
#
# x_in_ = x_in_.cpu().detach().numpy()
#
#
# x_in_ = np.transpose(x_in_, (0, 2, 3, 1))
#
# import cv2
#
# for imgidx in range(args.val_batch):
#
#     x_tmp = x_in_[imgidx]
#
#     print(name[imgidx])
#     bbox = bboxes[name[imgidx].item()]
#     print(bbox)
#
#     gxa = int(bbox[0])
#     gya = int(bbox[1])
#     gxb = int(bbox[2])
#     gyb = int(bbox[3])
#     x_tmp = cv2.cvtColor(x_tmp, cv2.COLOR_RGB2BGR)
#     x_tmp = cv2.rectangle(x_tmp, (max(1, gxa), max(1, gya)),
#                                            (min(args.image_size + 1, gxb), min(args.image_size + 1, gyb)), (0, 255, 0),
#                                            2)
#
#     cv2.imwrite('tmp{}.png'.format(imgidx), x_tmp)
#
