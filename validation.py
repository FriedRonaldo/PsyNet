from tqdm import trange
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import pickle

from utils import *


def validateTF(data_loader, networks, epoch, args, saveimgs=False, additional=None):
    losses = AverageMeter()
    top1s = dict()
    tot_types = ['rotation', 'translation', 'shear', 'hflip', 'scale', 'odd']
    for tftype in tot_types:
        top1s[tftype] = AverageMeter()
    # set nets
    C = networks[0]
    C.eval()
    # init data_loader
    val_iter = iter(data_loader)

    if args.dataset.lower() == 'cub':
        bbox_total = load_bbox_size(img_size=args.image_size)
    elif args.dataset.lower() in ['dogs', 'cars', 'aircraft']:
        bbox_total = additional["dataset"].load_bboxes()
    else:
        print("NOT IMPLEMENTED BBOXES FOR :", args.dataset)
        exit(-3)

    means = [0.485, .456, .406]
    stds = [.229, .224, .225]
    means = torch.reshape(torch.tensor(means), (1, 3, 1, 1)).cuda(args.gpu)
    stds = torch.reshape(torch.tensor(stds), (1, 3, 1, 1)).cuda(args.gpu)
    pi = torch.tensor(np.pi)

    use_total = False
    num_part = 128

    with torch.no_grad():
        t_val = trange(0, len(data_loader), initial=0, total=len(data_loader))

        seen = 0
        hit_gtknown = 0
        gtknown = 0.0

        for i in t_val:
            try:
                x_org, _, img_id = next(val_iter)
            except StopIteration:
                continue

            x_org = x_org.cuda(args.gpu, non_blocking=True)

            rot_label = torch.tensor(np.random.choice(args.tfnums[0], size=(x_org.size(0),))).cuda(args.gpu,
                                                                                                   non_blocking=True)
            trs_lable = torch.tensor(np.random.choice(args.tfnums[1], size=(x_org.size(0),))).cuda(args.gpu,
                                                                                                   non_blocking=True)
            sh_label = torch.tensor(np.random.choice(args.tfnums[2], size=(x_org.size(0),))).cuda(args.gpu,
                                                                                                  non_blocking=True)
            hf_label = torch.tensor(np.random.choice(args.tfnums[3], size=(x_org.size(0),))).cuda(args.gpu,
                                                                                                  non_blocking=True)
            sc_label = torch.tensor(np.random.choice(args.tfnums[4], size=(x_org.size(0),))).cuda(args.gpu,
                                                                                                  non_blocking=True)
            od_lable = torch.tensor(np.random.choice(args.tfnums[5], size=(x_org.size(0),))).cuda(args.gpu,
                                                                                                  non_blocking=True)

            rot = (rot_label * (360.0 / args.tfnums[0])).float()
            trs = ((trs_lable - (args.tfnums[1] // 2)).float() * args.tfval['T']).float()
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

            if args.method == 'tfcat':
                x_in = torch.cat((x_aff, x_org), 1)
                c_logit, _ = C(x_in)
            else:
                c_logit, _ = C(x_aff)

            c_loss = torch.tensor([0.0]).cuda(args.gpu, non_blocking=True)

            for logitidx in range(len(c_logit)):
                tmp_loss = F.cross_entropy(c_logit[logitidx], labels[logitidx])
                c_loss = c_loss + tmp_loss

            for tmpidx in range(len(args.tftypes)):
                tmpacc = accuracy(c_logit[tmpidx], labels[tmpidx])
                top1s[args.tftypes[tmpidx]].update(tmpacc[0].item(), x_org.size(0))

            losses.update(c_loss.item(), x_org.size(0))

            # if args.test:
            #     x_org = x_aff

            _, attmap = C(x_org)

            attmap = attmap[-1]

            x_org_ = x_org * stds + means
            x_org_ = x_org_.cpu().detach().numpy()

            attmap = norm_att_map(attmap)
            attmap = F.upsample(attmap.unsqueeze(dim=1), (x_org.size(2), x_org.size(3)), mode='bilinear')
            attmap = attmap.cpu().detach().numpy()

            x_org_ = np.transpose(x_org_, (0, 2, 3, 1))
            attmap = np.transpose(attmap, (0, 2, 3, 1))

            res = None

            for bidx in range(x_org.size(0)):
                _, cammed = cammed_image(x_org_[bidx], attmap[bidx])
                heatmap = intensity_to_rgb(np.squeeze(attmap[bidx]), normalize=True).astype('uint8')
                img_bbox = x_org_[bidx].copy()

                gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)

                th_val = 0.2 * np.max(gray_heatmap)

                _, th_gray_heatmap = cv2.threshold(gray_heatmap, int(th_val), 255, cv2.THRESH_BINARY)

                try:
                    _, contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                except:
                    contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                try:
                    bbox = bbox_total[img_id[bidx].item()]
                except:
                    bbox = bbox_total[img_id[bidx]]
                _img_bbox = (img_bbox.copy() * 255).astype('uint8')
                gxa = int(bbox[0])
                gya = int(bbox[1])
                gxb = int(bbox[2])
                gyb = int(bbox[3])
                cammed = cv2.rectangle(cammed, (max(1, gxa), max(1, gya)),
                                       (min(args.image_size + 1, gxb), min(args.image_size + 1, gyb)), (0, 255, 0),
                                       2)

                if len(contours) != 0:
                    c = max(contours, key=cv2.contourArea)

                    x, y, w, h = cv2.boundingRect(c)

                    estimated_box = [x, y, x + w, y + h]

                    IOU = calculate_IOU(bbox, estimated_box)
                    seen += 1
                    if IOU >= 0.5:
                        hit_gtknown += 1

                    cammed = cv2.rectangle(cammed, (max(1, estimated_box[0]), max(1, estimated_box[1])),
                                           (min(args.image_size + 1, estimated_box[2]),
                                            min(args.image_size + 1, estimated_box[3])), (255, 0, 0), 2)
                    if saveimgs:
                        if res is None:
                            res = np.copy(np.expand_dims(cammed, 0))
                        else:
                            cammed = np.expand_dims(cammed, 0)
                            res = np.concatenate((res, cammed), 0)

                gtknown = (hit_gtknown / seen) * 100

            if saveimgs:
                res = np.transpose(res, (0, 3, 1, 2))
                res = torch.Tensor(res)

                vutils.save_image(res.detach().cpu(), os.path.join(args.res_dir, 'HEAT_TEST_{}_{}.png'.format(epoch, i)),
                                  nrow=int(np.sqrt(res.size(0))),
                                  normalize=True)

            t_val.set_description('VAL: [{}/{}] '
                                  'Avg Loss: C[{losses.avg:.3f}] R[{rotacc1.avg:.3f}] '
                                  'T[{trsacc1.avg:.3f}] S[{shacc1.avg:.3f}] F[{hfacc1.avg:.3f}] '
                                  'C[{scacc1.avg:.3f}] O[{odacc1.avg:.3f}] LOC[{gtknown:.3f}]'
                                  .format(epoch, args.epochs, c_loss.item(),
                                          losses=losses, rotacc1=top1s['rotation'], trsacc1=top1s['translation'],
                                          shacc1=top1s['shear'], hfacc1=top1s['hflip'], scacc1=top1s['scale'],
                                          odacc1=top1s['odd'], gtknown=gtknown))
        print(gtknown)
    return {'R': top1s['rotation'].avg, 'T': top1s['translation'].avg, 'S': top1s['shear'].avg, 'H': top1s['hflip'].avg,
            'C': top1s['scale'].avg, 'O': top1s['odd'].avg, 'GT': gtknown}


def validateFull(data_loader, networks, epoch, args, saveimgs=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # set nets
    C = networks[0]
    C.eval()
    # init data_loader
    val_iter = iter(data_loader)
    bbox_total = load_bbox_size(img_size=args.image_size)

    means = [0.485, .456, .406]
    stds = [.229, .224, .225]
    means = torch.reshape(torch.tensor(means), (1, 3, 1, 1)).cuda(args.gpu)
    stds = torch.reshape(torch.tensor(stds), (1, 3, 1, 1)).cuda(args.gpu)
    with torch.no_grad():
        t_val = trange(0, len(data_loader), initial=0, total=len(data_loader))

        seen = 0
        hit_gtknown = 0
        hit_top1 = 0
        fail = 0
        gtknown = 0.0
        top1_loc = 0.0

        use_total = False
        num_part = 128

        for i in t_val:
            try:
                x_in, y_in, img_id = next(val_iter)
            except StopIteration:
                continue

            x_in = x_in.cuda(args.gpu, non_blocking=True)
            y_in = y_in.cuda(args.gpu, non_blocking=True)

            c_logit, featmap = C(x_in)

            if args.method == 'acol1':
                c_loss = torch.tensor([0.0]).cuda(args.gpu, non_blocking=True)

                for logitidx in range(len(c_logit)):
                    tmp_loss = F.cross_entropy(c_logit[logitidx], y_in)
                    c_loss = c_loss + tmp_loss
            else:
                c_loss = F.cross_entropy(c_logit, y_in)

            if args.method == 'acol1':
                acc1, acc5 = accuracy(c_logit[0], y_in, (1, 5))
                _, pred = c_logit[0].topk(1, 1, True, True)
                pred = pred.t()
            else:
                acc1, acc5 = accuracy(c_logit, y_in, (1, 5))
                _, pred = c_logit.topk(1, 1, True, True)
                pred = pred.t()

            top1.update(acc1[0].item(), x_in.size(0))
            top5.update(acc5[0].item(), x_in.size(0))

            losses.update(c_loss.item(), x_in.size(0))

            correct = pred.eq(y_in.view(1, -1).expand_as(pred))
            wrongs = [c == 0 for c in correct.cpu().numpy()][0]

            if use_total:
                attmap = featmap[-1].mean(1)
            else:
                attmap = featmap[-1]
                attmap_absmean = torch.mean(torch.abs(attmap), (2, 3))
                attmap_sorted, attmap_sidx = torch.sort(attmap_absmean, 1, descending=True)

                attmap_extracted = torch.stack([attmap[idx, attmap_sidx[idx, :num_part]] for idx in range(attmap.size(0))])
                attmap = attmap_extracted.mean(1)

            attmap = norm_att_map(attmap)
            attmap = F.upsample(attmap.unsqueeze(dim=1), (x_in.size(2), x_in.size(3)), mode='bilinear')
            x_in_ = x_in * stds + means

            x_in_ = x_in_.cpu().detach().numpy()
            attmap = attmap.cpu().detach().numpy()

            x_in_ = np.transpose(x_in_, (0, 2, 3, 1))
            attmap = np.transpose(attmap, (0, 2, 3, 1))
            res = None

            for bidx in range(x_in.size(0)):
                _, cammed = cammed_image(x_in_[bidx], attmap[bidx])
                heatmap = intensity_to_rgb(np.squeeze(attmap[bidx]), normalize=True).astype('uint8')
                img_bbox = x_in_[bidx].copy()

                gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)

                th_val = 0.2 * np.max(gray_heatmap)

                _, th_gray_heatmap = cv2.threshold(gray_heatmap, int(th_val), 255, cv2.THRESH_BINARY)

                try:
                    _, contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                except:
                    contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                bbox = bbox_total[img_id[bidx].item()]
                _img_bbox = (img_bbox.copy() * 255).astype('uint8')
                gxa = int(bbox[0])
                gya = int(bbox[1])
                gxb = int(bbox[2])
                gyb = int(bbox[3])
                cammed = cv2.rectangle(cammed, (max(1, gxa), max(1, gya)),
                                       (min(args.image_size + 1, gxb), min(args.image_size + 1, gyb)), (0, 255, 0),
                                       2)

                if len(contours) != 0:
                    c = max(contours, key=cv2.contourArea)

                    x, y, w, h = cv2.boundingRect(c)

                    estimated_box = [x, y, x + w, y + h]

                    IOU = calculate_IOU(bbox, estimated_box)
                    seen += 1
                    if IOU >= 0.5:
                        hit_gtknown += 1
                        if not wrongs[bidx]:
                            hit_top1 += 1
                    if wrongs[bidx]:
                        fail += 1

                    xA = max(bbox[0], estimated_box[0])
                    yA = max(bbox[1], estimated_box[1])
                    xB = min(bbox[2], estimated_box[2])
                    yB = min(bbox[3], estimated_box[3])

                    cammed = cv2.rectangle(cammed, (max(1, estimated_box[0]), max(1, estimated_box[1])),
                                           (min(args.image_size + 1, estimated_box[2]),
                                            min(args.image_size + 1, estimated_box[3])), (255, 0, 0), 2)
                    cammed = cv2.rectangle(cammed, (max(1, xA), max(1, yA)),
                                           (min(args.image_size + 1, xB),
                                            min(args.image_size + 1, yB)), (255, 255, 255), 2)
                    if saveimgs:
                        if res is None:
                            res = np.copy(np.expand_dims(cammed, 0))
                        else:
                            cammed = np.expand_dims(cammed, 0)
                            res = np.concatenate((res, cammed), 0)

                gtknown = (hit_gtknown / seen) * 100
                top1_loc = (hit_top1 / seen) * 100

            if saveimgs:
                res = np.transpose(res, (0, 3, 1, 2))
                res = torch.Tensor(res)

                vutils.save_image(res.detach().cpu(), os.path.join(args.res_dir, 'HEAT_TEST_{}_{}.png'.format(epoch, i)),
                                  nrow=int(np.sqrt(res.size(0))),
                                  normalize=True)

            t_val.set_description('VAL: [{}/{}], Loss per batch: C[{:.3f}] / '
                                  'Avg Loss: C[{losses.avg:.3f}] 1[{top1.avg:.3f}] '
                                  '5[{top5.avg:.3f}] L[{top1_loc:.3f}] G[{gtknown:.3f}]'
                                  .format(epoch, args.epochs, c_loss.item(),
                                          losses=losses, top1=top1, top5=top5,
                                          top1_loc=top1_loc, gtknown=gtknown))
        print(gtknown, top1_loc)
    return {'top1acc': top1.avg, 'top5acc': top5.avg, 'top1loc': top1_loc, 'gtknown': gtknown}


def validateImage(data_loader, networks, epoch, args, saveimgs=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # set nets
    C = networks[0]
    C.eval()
    # init data_loader
    val_iter = iter(data_loader)

    with open(os.path.join(args.data_dir, 'META', 'gt_imagenet'), 'rb') as f:
        gt_meta = pickle.load(f)

    means = [0.485, .456, .406]
    stds = [.229, .224, .225]
    means = torch.reshape(torch.tensor(means), (1, 3, 1, 1)).cuda(args.gpu)
    stds = torch.reshape(torch.tensor(stds), (1, 3, 1, 1)).cuda(args.gpu)
    with torch.no_grad():
        t_val = trange(0, len(data_loader), initial=0, total=len(data_loader))

        gt_idx = 0

        for i in t_val:
            try:
                x_in, y_in = next(val_iter)
            except StopIteration:
                continue

            x_in = x_in.cuda(args.gpu, non_blocking=True)
            y_in = y_in.cuda(args.gpu, non_blocking=True)

            c_logit, featmap = C(x_in)

            c_loss = F.cross_entropy(c_logit, y_in)

            acc1, acc5 = accuracy(c_logit, y_in, (1, 5))
            _, pred = c_logit.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(y_in.view(1, -1).expand_as(pred)).view(-1, 1)
            top1.update(acc1[0].item(), x_in.size(0))
            top5.update(acc5[0].item(), x_in.size(0))

            losses.update(c_loss.item(), x_in.size(0))

            attmap = featmap[-1].mean(1)

            # attmap = featmap[-1]
            # attmap_absmean = torch.mean(torch.abs(attmap), (2, 3))
            # attmap_sorted, attmap_sidx = torch.sort(attmap_absmean, 1, descending=True)

            # attmap_extracted = torch.stack([attmap[idx, attmap_sidx[idx, :64]] for idx in range(attmap.size(0))])
            # attmap = attmap_extracted.mean(1)

            attmap = norm_att_map(attmap)
            attmap = F.upsample(attmap.unsqueeze(dim=1), (x_in.size(2), x_in.size(3)), mode='bilinear')
            x_in_ = x_in * stds + means

            x_in_ = x_in_.cpu().detach().numpy()
            attmap = attmap.cpu().detach().numpy()

            x_in_ = np.transpose(x_in_, (0, 2, 3, 1))
            attmap = np.transpose(attmap, (0, 2, 3, 1))
            res = None

            for bidx in range(x_in.size(0)):
                _, cammed = cammed_image(x_in_[bidx], attmap[bidx])
                heatmap = intensity_to_rgb(np.squeeze(attmap[bidx]), normalize=True).astype('uint8')
                img_bbox = x_in_[bidx].copy()

                gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)

                th_val = 0.35 * np.max(gray_heatmap)

                _, th_gray_heatmap = cv2.threshold(gray_heatmap, int(th_val), 255, cv2.THRESH_BINARY)

                try:
                    _, contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                except:
                    contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                bbox = []

                bbox = gt_meta['gt_boxes'][gt_idx]
                _img_bbox = (img_bbox.copy() * 255).astype('uint8')

                if len(contours) != 0:
                    c = max(contours, key=cv2.contourArea)

                    x, y, w, h = cv2.boundingRect(c)

                    estimated_box = [x, y, x + w, y + h]
                    if correct[bidx]:
                        cammed = cv2.rectangle(cammed, (max(1, estimated_box[0]), max(1, estimated_box[1])),
                                               (min(args.image_size + 1, estimated_box[2]),
                                                min(args.image_size + 1, estimated_box[3])), (0, 255, 0), 2)
                    else:
                        cammed = cv2.rectangle(cammed, (max(1, estimated_box[0]), max(1, estimated_box[1])),
                                               (min(args.image_size + 1, estimated_box[2]),
                                                min(args.image_size + 1, estimated_box[3])), (255, 0, 0), 2)
                    if saveimgs:
                        if res is None:
                            res = np.copy(np.expand_dims(cammed, 0))
                        else:
                            cammed = np.expand_dims(cammed, 0)
                            res = np.concatenate((res, cammed), 0)

            if saveimgs:
                res = np.transpose(res, (0, 3, 1, 2))
                res = torch.Tensor(res)

                vutils.save_image(res.detach().cpu(), os.path.join(args.res_dir, 'HEAT_TEST_{}_{}.png'.format(epoch, i)),
                                  nrow=int(np.sqrt(res.size(0))),
                                  normalize=True)

            t_val.set_description('VAL: [{}/{}], Loss per batch: C[{:.3f}] / '
                                  'Avg Loss: C[{losses.avg:.3f}] 1[{top1.avg:.3f}] '
                                  '5[{top5.avg:.3f}] L[{top1_loc:.3f}] G[{gtknown:.3f}]'
                                  .format(epoch, args.epochs, c_loss.item(),
                                          losses=losses, top1=top1, top5=top5,
                                          top1_loc=0.0, gtknown=0.0))
    return {'top1acc': top1.avg, 'top5acc': top5.avg, 'top1loc': 0.0, 'gtknown': 0.0}
