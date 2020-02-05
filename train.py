from tqdm import trange
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils import *


def trainTF(train_loader, networks, opts, epoch, args, additional):
    # avg meter
    losses = AverageMeter()
    top1s = dict()
    tot_types = ['rotation', 'translation', 'shear', 'hflip', 'scale', 'odd']
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
        trs_lable = torch.tensor(np.random.choice(args.tfnums[1], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)
        sh_label = torch.tensor(np.random.choice(args.tfnums[2], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)
        hf_label = torch.tensor(np.random.choice(args.tfnums[3], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)
        sc_label = torch.tensor(np.random.choice(args.tfnums[4], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)
        od_lable = torch.tensor(np.random.choice(args.tfnums[5], size=(x_org.size(0),))).cuda(args.gpu, non_blocking=True)

        rot = (rot_label * (360.0/args.tfnums[0])).float()
        trs = ((trs_lable - (args.tfnums[1]//2)).float() * args.tfval['T']).float()
        sh = ((sh_label - 1) * args.tfval['S']).float()
        hf = (2 * (hf_label - 0.5)).float()
        sc = 1.0 - ((sc_label - 1.0).float() * args.tfval['C'])
        od = ((od_lable - (args.tfnums[5] // 2)).float() * args.tfval['O']).float()

        cosR = torch.cos(rot * pi / 180.0)
        sinR = torch.sin(rot * pi / 180.0)
        tanS = torch.tan(sh * pi / 180.0)

        rotmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)
        trsmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)
        shmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)
        hfmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)
        scmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)
        odmat = torch.zeros(x_org.size(0), 3, 3).cuda(args.gpu, non_blocking=True)

        rotmat[:, 0, 0] = cosR
        rotmat[:, 0, 1] = -sinR
        rotmat[:, 1, 0] = sinR
        rotmat[:, 1, 1] = cosR
        rotmat[:, 2, 2] = 1.0

        trsmat[:, 0, 0] = 1.0
        trsmat[:, 0, 2] = trs
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

        scmat[:, 0, 0] = sc
        scmat[:, 1, 1] = sc
        scmat[:, 2, 2] = 1.0

        odmat[:, 0, 0] = 1.0
        odmat[:, 0, 2] = od
        odmat[:, 1, 1] = 1.0
        odmat[:, 1, 2] = od
        odmat[:, 2, 2] = 1.0

        mats = []
        labels = []

        if 'odd' in args.tftypes:
            mats.append(odmat)
            labels.append(od_lable)
        if 'rotation' in args.tftypes:
            mats.append(rotmat)
            labels.append(rot_label)
        if 'translation' in args.tftypes:
            mats.append(trsmat)
            labels.append(trs_lable)
        if 'shear' in args.tftypes:
            mats.append(shmat)
            labels.append(sh_label)
        if 'hflip' in args.tftypes:
            mats.append(hfmat)
            labels.append(hf_label)
        if 'scale' in args.tftypes:
            mats.append(scmat)
            labels.append(sc_label)

        theta = mats[0]

        for matidx in range(1, len(mats)):
            theta = torch.matmul(theta, mats[matidx])
        theta = theta[:, :2, :]

        affgrid = F.affine_grid(theta, x_org.size()).cuda(args.gpu, non_blocking=True)
        x_aff = F.grid_sample(x_org, affgrid, padding_mode='reflection')

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

        if i % args.log_step == 0:
            t_train.set_description('Epoch: [{}/{}] '
                                    'Avg Loss: C[{losses.avg:.3f}] R[{rotacc1.avg:.3f}] '
                                    'T[{trsacc1.avg:.3f}] S[{shacc1.avg:.3f}] F[{hfacc1.avg:.3f}] '
                                    'C[{scacc1.avg:.3f}] O[{odacc1.avg:.3f}]'
                                    .format(epoch, args.epochs,
                                            losses=losses, rotacc1=top1s['rotation'], trsacc1=top1s['translation'],
                                            shacc1=top1s['shear'], hfacc1=top1s['hflip'], scacc1=top1s['scale'],
                                            odacc1=top1s['odd']))
    return {'R': top1s['rotation'].avg, 'T': top1s['translation'].avg,
            'S': top1s['shear'].avg, 'H': top1s['hflip'].avg,
            'C': top1s['scale'].avg, 'O': top1s['odd'].avg}


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
