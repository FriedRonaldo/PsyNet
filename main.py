import argparse
import time
import warnings
from tqdm import trange
from datetime import datetime
import numpy as np
from glob import glob
from shutil import copyfile
import cv2
from datasets.datasetgetter import get_dataset

import torch
from torch import autograd
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from models.resnet import rescamtf
from models.vgg import *
from models.vggtf import vggcamtf
from models.senet import serescamtf
from train import *
from validation import *

from utils import *

parser = argparse.ArgumentParser(description='PyTorch Simultaneous Training')
parser.add_argument('--data_dir', default='../data/', help='path to dataset')
parser.add_argument('--dataset', default='CUB', help='type of dataset',
                    choices=['CUB', 'IMAGENET', 'CARS', 'DOGS', 'AIRCRAFT'])
parser.add_argument('--network', type=str, default='serescam50',
                    choices=['vggcam16', 'vggcam16bn', 'vggcam19', 'vggcam19bn', 'vggimg16',
                             'rescam18', 'rescam34', 'rescam50', 'rescam101', 'rescam152',
                             'serescam50', 'serescam101', 'serescam152'])
parser.add_argument('--method', type=str, default='tf', choices=['cam', 'acol1', 'acol2', 'adl', 'acolcam', 'none', 'tf'], help='')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=80, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Total batch size - e.g) num_gpus = 2 , batch_size = 128 then, effectively, 64')
parser.add_argument('--val_batch', default=32, type=int)
parser.add_argument('--image_size', default=224, type=int, help='Input image size')
parser.add_argument('--log_step', default=50, type=int, help='print frequency (default: 50)')
parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('--validation', dest='validation', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--port', default='8888', type=str)
parser.add_argument('--tftypes', type=str, default='RTSHCO',
                    help='R: rotation / T: translation / S: shear / H : hflip / C : scale / O : odd')


def main():
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if "vgg" in args.network:
        tmppre = '_V'
    elif 'seres' in args.network:
        tmppre = '_S'

    args.prefix = args.dataset + '_' + args.tftypes + tmppre

    if args.load_model is None:
        args.model_name = '{}_{}'.format(args.prefix, datetime.now().strftime("%m-%d_%H-%M-%S"))
    else:
        args.model_name = args.load_model

    makedirs('./logs')
    makedirs('./results')

    args.log_dir = os.path.join('./logs', args.model_name)
    args.res_dir = os.path.join('./results', args.model_name)

    makedirs(args.log_dir)
    makedirs(os.path.join(args.log_dir, 'codes'))
    makedirs(os.path.join(args.log_dir, 'codes', 'models'))
    makedirs(os.path.join(args.log_dir, 'codes', 'datasets'))
    makedirs(args.res_dir)

    if args.load_model is None:
        pyfiles = glob("./*.py")
        modelfiles = glob('./models/*.py')
        datafiles = glob('./datasets/*.py')
        for py in pyfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes') + "/" + py)
        for py in modelfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))
        for py in datafiles:
            copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))

    formatted_print('Total Number of GPUs:', ngpus_per_node)
    formatted_print('Total Number of Workers:', args.workers)
    formatted_print('Epochs:', args.epochs)
    formatted_print('Batch Size:', args.batch_size)
    formatted_print('Image Size:', args.image_size)
    formatted_print('Log step:', args.log_step)
    formatted_print('Log DIR:', args.log_dir)
    formatted_print('Result DIR:', args.res_dir)
    formatted_print('Network:', args.network)
    formatted_print('Method:', args.method)

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    if len(args.gpu) == 1:
        args.gpu = 0
    else:
        args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:'+args.port,
                                world_size=args.world_size, rank=args.rank)

    ################
    # Define model #
    ################
    # ['rotation', 'translation', 'shear', 'hflip', 'scale']
    tmptftypes = []
    if 'R' in args.tftypes:
        tmptftypes.append('rotation')
    if 'T' in args.tftypes:
        tmptftypes.append('translation')
    if 'S' in args.tftypes:
        tmptftypes.append('shear')
    if 'H' in args.tftypes:
        tmptftypes.append('hflip')
    if 'C' in args.tftypes:
        tmptftypes.append('scale')
    if 'O' in args.tftypes:
        tmptftypes.append('odd')

    args.tftypes_org = args.tftypes
    args.tftypes = tmptftypes
    args.tfnums = [4, 3, 3, 2, 3, 5]
    args.tfval = {'T': 0.1, 'C': 0.3, 'S': 30, 'O': 3.0}

    print("=> Creating Classifier")
    if 'tf' in args.method:
        print('=> Use transform:\t', args.tftypes)
        print('=> Use nums:\t', args.tfnums)
        print('=> Use vals:\t', args.tfval)
        if 'vgg' in args.network:
            print("USE VGGCAMTF")
            classifier = vggcamtf(args.network, args.tftypes, args.tfnums, pretrained=True)
        elif 'seres' in args.network:
            print('USE SERESNET')
            classifier = serescamtf(args.network, args.tftypes, args.tfnums, pretrained=True)
        elif 'res' in args.network:
            print('USE RESCAMTF')
            classifier = rescamtf(args.network, args.tftypes, args.tfnums, pretrained=True)
        else:
            print("NOT IMPLEMENTED")
            return

    else:
        if args.dataset.lower() == 'imagenet':
            if 'vgg' in args.network:
                print('USE VGG')
                assert args.network in ['vggimg16']
                classifier = vggimg(args.network)
        else:
            if 'vgg' in args.network:
                print('USE VGG')
                classifier = vggcam(True, args.network, args.method, dataset=args.dataset)
            elif args.network == 'res':
                print('USE RES')
                print("NOT IMPLEMENTED")
                return
            else:
                print("NOT IMPLEMENTED")
                return

    networks = [classifier]

    if args.distributed:
        if args.gpu is not None:
            print('Distributed', args.gpu)
            torch.cuda.set_device(args.gpu)
            networks = [x.cuda(args.gpu) for x in networks]
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            networks = [torch.nn.parallel.DistributedDataParallel(x, device_ids=[args.gpu], output_device=args.gpu) for x in networks]
        else:
            networks = [x.cuda() for x in networks]
            networks = [torch.nn.parallel.DistributedDataParallel(x) for x in networks]

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        networks = [x.cuda(args.gpu) for x in networks]
    else:
        networks = [torch.nn.DataParallel(x).cuda() for x in networks]

    classifier, = networks
    ######################
    # Loss and Optimizer #
    ######################
    if args.distributed:
        c_opt = torch.optim.SGD(classifier.module.parameters(), 0.001, momentum=0.9, nesterov=True)
    else:
        c_opt = torch.optim.SGD(classifier.parameters(), 0.001, momentum=0.9, nesterov=True)

    ##############
    # Load model #
    ##############
    if args.load_model is not None:
        check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
        to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['C_state_dict'])
            c_opt.load_state_dict(checkpoint['c_optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.log_dir))

    cudnn.benchmark = True

    ###########
    # Dataset #
    ###########
    train_dataset, val_dataset = get_dataset(args.dataset, args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch,
                                             num_workers=args.workers, pin_memory=True)

    ######################
    # Validate and Train #
    ######################
    if args.validation:
        if 'tf' in args.method:
            validateTF(val_loader, networks, 678, args, True, additional={"dataset": val_dataset})
        else:
            if args.dataset.lower() == 'imagenet':
                validateImage(val_loader, networks, 123, args, True)
            else:
                validateFull(val_loader, networks, 123, args, True)
        return

    elif args.test:
        if 'tf' in args.method:
            validateTF(val_loader, networks, 456, args, True, additional={"dataset": val_dataset})
        else:
            if args.dataset.lower() == 'imagenet':
                validateImage(val_loader, networks, 456, args, True)
            else:
                validateFull(val_loader, networks, 456, args, True)
        return

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
        record_txt = open(os.path.join(args.log_dir, "record.txt"), "a+")
        record_txt.write('Network\t:\t{}\n'.format(args.network))
        record_txt.write('Method\t:\t{}\n'.format(args.method))
        record_txt.write('DATASET\t:\t{}\n'.format(args.dataset))
        record_txt.write('TFTYPES\t:\t{}\n'.format(args.tftypes))
        record_txt.write('TFNUMS\t:\t{}\n'.format(args.tfnums))
        record_txt.write('TFVALS\t:\t{}\n'.format(args.tfval))
        record_txt.close()

    best = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch in [60, ]:
            for param_group in c_opt.param_groups:
                param_group['lr'] *= 0.2
                print(param_group['lr'])

        if 'tf' in args.method:
            acc_train = trainTF(train_loader, networks, [c_opt], epoch, args, [None])
            acc_val = validateTF(val_loader, networks, epoch, args, saveimgs=False, additional={"dataset": val_dataset})
            best_criterion = acc_val['GT']
        else:
            acc_train = trainFull(train_loader, networks, [c_opt], epoch, args, [None])
            acc_val = validateFull(val_loader, networks, epoch, args, saveimgs=False)
            best_criterion = acc_val['top1loc']

        ##############
        # Save model #
        ##############
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if epoch == 0:
                check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
            if best < best_criterion:
                best = best_criterion
                best_ckpt = [f for f in glob(os.path.join(args.log_dir, "*.ckpt")) if "best" in f]
                if len(best_ckpt) != 0:
                    os.remove(best_ckpt[0])
                save_checkpoint({
                    'epoch': epoch + 1,
                    'C_state_dict': classifier.state_dict(),
                    'c_optimizer': c_opt.state_dict(),
                }, check_list, args.log_dir, 'best' + str(best).replace(".", "_"))
            if (epoch + 1) % (args.epochs//10) == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'C_state_dict': classifier.state_dict(),
                    'c_optimizer': c_opt.state_dict(),
                }, check_list, args.log_dir, epoch + 1)
            if epoch == (args.epochs - 1):
                check_list.close()
            record_txt = open(os.path.join(args.log_dir, "record.txt"), "a+")
            if 'tf' in args.method:
                record_txt.write(
                    'Epoch : {:3d}, Train R : {:.4f} T : {:.4f} S : {:.4f} H : {:.4f} C : {:.4f} O : {:.4f},'
                    ' VAL R : {:.4f} T : {:.4f} S : {:.4f} H : {:.4f} C : {:.4f} O : {:.4f}, GT : {:.4f}\n'.
                        format(epoch, acc_train['R'], acc_train['T'],
                               acc_train['S'], acc_train['H'], acc_train['C'], acc_train['O'],
                               acc_val['R'], acc_val['T'], acc_val['S'], acc_val['H'], acc_val['C'], acc_val['O'],
                               acc_val['GT']))
            else:
                record_txt.write(
                    'Epoch : {:3d}, Train top1 : {:.4f} top5 : {:.4f} ,'
                    ' VAL top1 : {:.4f} top5 : {:.4f} LOC : {:.4f} GT : {:.4f}\n'.
                    format(epoch, acc_train['top1acc'], acc_train['top5acc'],
                           acc_val['top1acc'], acc_val['top5acc'], acc_val['top1loc'], acc_val['gtknown']))

            record_txt.close()
            copyfile(os.path.join(args.log_dir, "record.txt"), os.path.join(args.log_dir, "recordNOW.txt"))

            print('BEST LOC EVER : {:.3f}'.format(best))


if __name__ == '__main__':
    main()
