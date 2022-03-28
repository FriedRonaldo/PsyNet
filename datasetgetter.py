# from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from testfolder import ImageFolderWithPath
from cub200 import Cub2011
import os
import torchvision.transforms as transforms
from dogs import Dogs
from cars import Cars
from aircraft import AIRCRAFT


def get_dataset(dataset, args):
    if dataset.lower() == 'cub':
        print('USE CUB DATASET')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transforms_train = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                                               # transforms.RandomHorizontalFlip(),
                                               # transforms.RandomVerticalFlip(),
                                               transforms.ToTensor(),
                                               normalize])
        transforms_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

        train_dataset = Cub2011(os.path.join(args.data_dir, args.dataset),
                                transform=transforms_train, download=False, mode='full')
        val_dataset = Cub2011(os.path.join(args.data_dir, args.dataset),
                              transform=transforms_val, train=False, with_id=True, mode='full')
    elif dataset.lower() == 'imagenet':
        print("USE IMAGENET")
        datapath = args.data_dir
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = ImageFolderWithPath(traindir, transforms.Compose([transforms.RandomResizedCrop(224),
                                                                 transforms.RandomHorizontalFlip(),
                                                                 transforms.ToTensor(), normalize, ]), with_name=False)
        val_dataset = ImageFolderWithPath(valdir, transforms.Compose([transforms.Resize((224, 224)),
                                                              transforms.ToTensor(),
                                                              normalize, ]), with_name=True)
    elif dataset.lower() == 'cifar10':
        print('NOT IMPLEMENTED DATASET :', dataset)
        exit(-3)
    elif dataset.lower() == 'cifar100':
        print('NOT IMPLEMENTED DATASET :', dataset)
        exit(-3)
    elif dataset.lower() == 'cars':
        print('USE CARS DATASET')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transforms_train = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                                               # transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize])
        transforms_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

        train_dataset = Cars(args.data_dir, True, transforms_train, False)
        val_dataset = Cars(args.data_dir, False, transforms_val, True)

    elif dataset.lower() == 'dogs':
        print('USE DOGS DATASET')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transforms_train = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                                               # transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize])
        transforms_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

        train_dataset = Dogs(args.data_dir, transform=transforms_train, with_id=False)
        val_dataset = Dogs(args.data_dir, transform=transforms_val, with_id=True, train=False)

    elif dataset.lower() == 'aircraft':
        print('USE AIRCRAFT DATASET')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transforms_train = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                                               # transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize])
        transforms_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

        train_dataset = AIRCRAFT(args.data_dir, transform=transforms_train, with_id=False, train='train')
        val_dataset = AIRCRAFT(args.data_dir, transform=transforms_val, with_id=True, train='test')
        # val_dataset = AIRCRAFT(args.data_dir, transform=transforms_val, with_id=True, train='val')

    else:
        print('NOT IMPLEMENTED DATASET :', dataset)
        exit(-3)

    return train_dataset, val_dataset