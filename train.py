from tqdm import trange
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils import *

import torchvision.utils as vutils


def trainTF(train_loader, networks, opts, epoch, args, additional):
    # avg meter
    losses = AverageMeter()
    top1s = dict()
    tot_types = ['rotation', 'translation', 'shear', 'hflip', 'scale', 'vflip', 'vtranslation', 'odd']
    for tftype in tot_types:
        top1s[tftype] = AverageMeter()

    # set nets
    C = networks[0]
    # set opts
    c_opt = opts[0]
    # switch to train mode
    C.train()
    # summary writer
    # writer = additional[0]
    train_it = iter(train_loader)
    t_train = trange(0, len(train_loader), initial=0, total=len(train_loader))
    pi = torch.tensor(np.pi)

    for i in t_train:
        try:
            x_org, _ = next(train_it)
        except StopIteration:
            continue

        x_org = x_org.cuda(args.gpu, non_blocking=True)

        rot_label = torch.tensor(np.random.choice(args.tfnums[0], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)
        trs_label = torch.tensor(np.random.choice(args.tfnums[1], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)
        sh_label = torch.tensor(np.random.choice(args.tfnums[2], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)
        hf_label = torch.tensor(np.random.choice(args.tfnums[3], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)
        sc_label = torch.tensor(np.random.choice(args.tfnums[4], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)
        vf_label = torch.tensor(np.random.choice(args.tfnums[5], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)
        vtrs_label = torch.tensor(np.random.choice(args.tfnums[6], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)
        odd_label = torch.tensor(np.random.choice(args.tfnums[7], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)

        odd_val = float(args.o_value)

        # odd = ((odd_label - (args.tfnums[7] // 2)).float() * 3.0).float()
        odd1 = ((odd_label - (args.tfnums[7] // 2)).float() * odd_val).float()
        odd2 = ((odd_label - (args.tfnums[7] // 2)).float() * odd_val).float()
        odd1[odd1 == 2 * odd_val] = odd_val
        odd2[odd2 == 2 * odd_val] = -odd_val
        odd1[odd1 == -2 * odd_val] = -odd_val
        odd2[odd2 == -2 * odd_val] = odd_val
        # rot = (rot_label * (360.0 / args.tfnums[0])).float()
        rot = (rot_label * 90.0).float()
        trs = ((trs_label - (args.tfnums[1]//2)).float() * 3.0).float()
        sh = ((sh_label - 1) * 30.0).float()
        hf = (2 * (hf_label - 0.5)).float()
        sc = 1.0 - ((sc_label - 1.0).float() * 0.3)
        vf = (2 * (vf_label - 0.5)).float()
        vtrs = ((vtrs_label - (args.tfnums[6]//2)).float() * 3.0).float()

        cosR = torch.cos(rot * pi / 180.0)
        sinR = torch.sin(rot * pi / 180.0)
        tanS = torch.tan(sh * pi / 180.0)

        rotmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)
        trsmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)
        shmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)
        hfmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)
        scmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)
        vfmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)
        vtrsmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)
        oddmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)

        rotmat[:, 0, 0] = cosR
        rotmat[:, 0, 1] = -sinR
        rotmat[:, 1, 0] = sinR
        rotmat[:, 1, 1] = cosR
        rotmat[:, 2, 2] = 1.0

        trsmat[:, 0, 0] = 1.0
        trsmat[:, 0, 2] = 0.0
        trsmat[:, 1, 1] = 1.0
        trsmat[:, 1, 2] = trs
        trsmat[:, 2, 2] = 1.0

        shmat[:, 0, 0] = 1.0
        shmat[:, 0, 1] = tanS
        shmat[:, 1, 1] = 1.0
        shmat[:, 2, 2] = 1.0

        hfmat[:, 0, 0] = hf
        hfmat[:, 1, 1] = 1.0
        hfmat[:, 2, 2] = 1.0
        # print(sc)
        scmat[:, 0, 0] = sc
        scmat[:, 1, 1] = sc
        scmat[:, 2, 2] = 1.0

        vfmat[:, 0, 0] = 1.0
        vfmat[:, 1, 1] = vf
        vfmat[:, 2, 2] = 1.0

        vtrsmat[:, 0, 0] = 1.0
        vtrsmat[:, 0, 2] = vtrs
        vtrsmat[:, 1, 1] = 1.0
        vtrsmat[:, 1, 2] = 0.0
        vtrsmat[:, 2, 2] = 1.0

        oddmat[:, 0, 0] = 1.0
        oddmat[:, 0, 2] = odd1
        oddmat[:, 1, 1] = 1.0
        oddmat[:, 1, 2] = odd2
        oddmat[:, 2, 2] = 1.0

        mats = []
        labels = []

        if 'odd' in args.tftypes:
            mats.append(oddmat)
            labels.append(odd_label)
        if 'rotation' in args.tftypes:
            mats.append(rotmat)
            labels.append(rot_label)
        if 'translation' in args.tftypes:
            mats.append(trsmat)
            labels.append(trs_label)
        if 'shear' in args.tftypes:
            mats.append(shmat)
            labels.append(sh_label)
        if 'hflip' in args.tftypes:
            mats.append(hfmat)
            labels.append(hf_label)
        if 'scale' in args.tftypes:
            mats.append(scmat)
            labels.append(sc_label)
        if 'vflip' in args.tftypes:
            mats.append(vfmat)
            labels.append(vf_label)
        if 'vtranslation' in args.tftypes:
            mats.append(torch.matmul(vtrsmat, trsmat))
            labels.append(vtrs_label * 3 + trs_label)

        theta = mats[0]

        for matidx in range(1, len(mats)):
            theta = torch.matmul(theta, mats[matidx])
        theta = theta[:, :2, :]
        affgrid = F.affine_grid(theta, x_org.size()).cuda(args.gpu, non_blocking=True)

        x_aff = F.grid_sample(x_org, affgrid, padding_mode='reflection')

        # vutils.save_image(x_org, os.path.join(args.res_dir, 'HEAT_TEST_{}_{}_0.png'.format(epoch, i)),
        #                   nrow=int(np.sqrt(x_org.size(0))),
        #                   normalize=True)
        #
        # vutils.save_image(x_aff, os.path.join(args.res_dir, 'HEAT_TEST_{}_{}_1.png'.format(epoch, i)),
        #                   nrow=int(np.sqrt(x_aff.size(0))),
        #                   normalize=True)
        #
        # exit()

        c_logit, _ = C(x_aff)

        c_loss = torch.tensor([0.0]).cuda(args.gpu, non_blocking=True)

        for logitidx in range(len(c_logit)):
            tmp_loss = F.cross_entropy(c_logit[logitidx], labels[logitidx])
            c_loss = c_loss + tmp_loss

        c_opt.zero_grad()
        c_loss.backward()
        c_opt.step()

        for tmpidx in range(len(args.tftypes)):
            tmpacc = accuracy(c_logit[tmpidx], labels[tmpidx])
            top1s[args.tftypes[tmpidx]].update(tmpacc[0].item(), x_org.size(0))

        losses.update(c_loss.item(), x_org.size(0))

        # if i % args.log_step == 0:
        t_train.set_description('Epoch: [{}/{}], Loss per batch: C[{:.3f}] / '
                                'Avg Loss: C[{losses.avg:.3f}] R[{rotacc1.avg:.3f}] '
                                'T[{trsacc1.avg:.3f}] S[{shacc1.avg:.3f}] F[{hfacc1.avg:.3f}] C[{scacc1.avg:.3f}] '
                                'V[{vfacc1.avg:.3f}] X[{vtrsacc1.avg:.3f}] O[{oddacc1.avg:.3f}]'
                                .format(epoch, args.epochs, c_loss.item(),
                                        losses=losses, rotacc1=top1s['rotation'], trsacc1=top1s['translation'],
                                        shacc1=top1s['shear'], hfacc1=top1s['hflip'], scacc1=top1s['scale'],
                                        vfacc1=top1s['vflip'], vtrsacc1=top1s['vtranslation'], oddacc1=top1s['odd']))
    return {'R': top1s['rotation'].avg, 'T': top1s['translation'].avg, 'S': top1s['shear'].avg, 'H': top1s['hflip'].avg,
            'C': top1s['scale'].avg, 'V': top1s['vflip'].avg, 'X': top1s['vtranslation'].avg, 'O': top1s['odd'].avg}


def trainFull(train_loader, networks, opts, epoch, args, additional):
    # avg meter
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set nets
    C = networks[0]
    # set opts
    c_opt = opts[0]
    # switch to train mode
    C.train()
    # summary writer
    # writer = additional[0]
    train_it = iter(train_loader)
    t_train = trange(0, len(train_loader), initial=0, total=len(train_loader))

    for i in t_train:
        try:
            x_in, y_in = next(train_it)
        except StopIteration:
            continue

        x_in = x_in.cuda(args.gpu, non_blocking=True)
        y_in = y_in.cuda(args.gpu, non_blocking=True)

        c_logit, _ = C(x_in)

        if args.method == 'acol1':
            c_loss = torch.tensor([0.0]).cuda(args.gpu, non_blocking=True)

            for logitidx in range(len(c_logit)):
                tmp_loss = F.cross_entropy(c_logit[logitidx], y_in)
                c_loss = c_loss + tmp_loss
        else:
            c_loss = F.cross_entropy(c_logit, y_in)

        c_opt.zero_grad()
        c_loss.backward()
        c_opt.step()
        if args.method == 'acol1':
            acc1, acc5 = accuracy(c_logit[0], y_in, (1, 5))
        else:
            acc1, acc5 = accuracy(c_logit, y_in, (1, 5))

        top1.update(acc1[0].item(), x_in.size(0))
        top5.update(acc5[0].item(), x_in.size(0))

        losses.update(c_loss.item(), x_in.size(0))

        if i % args.log_step == 0:
            t_train.set_description('Epoch: [{}/{}], Loss per batch: C[{:.3f}] / '
                                    'Avg Loss: C[{losses.avg:.3f}] 1[{top1.avg:.3f}] '
                                    '5[{top5.avg:.3f}]'
                                    .format(epoch, args.epochs, c_loss.item(),
                                            losses=losses, top1=top1, top5=top5))
    return {'top1acc': top1.avg, 'top5acc': top5.avg}