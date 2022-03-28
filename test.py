from tqdm import trange
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import pickle

from cub200 import Cub2011
import os
import torchvision.transforms as transforms

from utils import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transforms_val = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     normalize])
val_dataset = Cub2011(os.path.join('../data', 'CUB'),
                      transform=transforms_val, train=False, with_id=True, mode='full')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         num_workers=8, pin_memory=True, shuffle=True)
gpu = 0
val_iter = iter(val_loader)
pi = torch.tensor(np.pi)

for ii in range(20):

    x_org, _, img_id = next(val_iter)
    x_org = x_org.cuda(gpu, non_blocking=True)

    # odd = torch.tensor([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).cuda(gpu, non_blocking=True)
    # odd = torch.tensor([3.0, 2.5, 2.2, 1.8, 1.5, 1.2, 0.0]).cuda(gpu, non_blocking=True)
    odd = torch.tensor([0.4]).cuda(gpu, non_blocking=True)
    # vodd = torch.tensor([-3.0, 3.0]).cuda(gpu, non_blocking=True)
    # rot = torch.tensor([0, 90]).cuda(gpu, non_blocking=True)

    oddmat = torch.zeros(x_org.size(0), 3, 3).cuda(gpu, non_blocking=True)
    oddmat[:, 0, 0] = 1.0
    oddmat[:, 1, 1] = 1.0
    oddmat[:, 2, 2] = 1.0

    voddmat = torch.zeros(x_org.size(0), 3, 3).cuda(gpu, non_blocking=True)
    voddmat[:, 0, 0] = 1.0
    voddmat[:, 1, 1] = 1.0
    voddmat[:, 2, 2] = 1.0


    rotmat = torch.zeros(x_org.size(0), 3, 3).cuda(gpu, non_blocking=True)
    rotmat[:, 2, 2] = 1.0

    odds = []

    for k in odd:
        oddmat[:, 0, 2] = k
        oddmat[:, 1, 2] = k
        # for l in rot:
        #     cosR = torch.cos(l * pi / 180.0)
        #     sinR = torch.sin(l * pi / 180.0)
        #     rotmat[:, 0, 0] = cosR
        #     rotmat[:, 0, 1] = -sinR
        #     rotmat[:, 1, 0] = sinR
        #     rotmat[:, 1, 1] = cosR
        # for l in vodd:
        #     voddmat[:, 0, 2] = k
        #     voddmat[:, 1, 2] = l
        #     for j in rot:
        #         cosR = torch.cos(j * pi / 180.0)
        #         sinR = torch.sin(j * pi / 180.0)
        #         rotmat[:, 0, 0] = cosR
        #         rotmat[:, 0, 1] = -sinR
        #         rotmat[:, 1, 0] = sinR
        #         rotmat[:, 1, 1] = cosR
        theta = oddmat
            # theta = torch.matmul(oddmat, rotmat)
        # theta = torch.matmul(voddmat, rotmat)
        # theta = torch.matmul(theta, rotmat)
        theta = theta[:, :2, :]

        affgrid = F.affine_grid(theta, x_org.size()).cuda(gpu, non_blocking=True)
        x_aff = F.grid_sample(x_org, affgrid, padding_mode='reflection')

        odds.append(x_aff)

    x_res = torch.cat(odds, dim=3)
    vutils.save_image(x_res, os.path.join('ODD_cmp_{}.png'.format(ii)),
                      normalize=True)