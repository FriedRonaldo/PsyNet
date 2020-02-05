import os
import re
import math
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import cv2
from utils import *

import matplotlib.pyplot as plt


from torch.autograd import Variable


def load_ground_truth_imagenet(data_root):
    ground_truth = dict(
        image_list=[],
        black_list=[],
        image_sizes={},
        gt_labels=[],
        gt_bboxes={},
        class_names=[],
        class_words=[],
    )
    # load image list
    ground_truth['image_list'] = [line.split('.')[0] for line in
                                  list(open(os.path.join(data_root, 'imagenet_val.txt'), 'r'))]
    # load black list
    ground_truth['black_list'] = [int(x.strip()) for x in list(
        open(os.path.join(data_root, 'ILSVRC2014_clsloc_validation_blacklist.txt'), 'r'))]

    for item in list(open(os.path.join(data_root, 'synset_words.txt'), 'r')):
        a = re.match(r'^(\w\d+) (.+)', item)
        ground_truth['class_words'].append(a.group(1))
        ground_truth['class_names'].append(a.group(2))
    category_ori = -1
    # load annotations
    for idx, img_name in enumerate(ground_truth['image_list']):
        # if img_name != 'ILSVRC2012_val_00002005':
        #     continue
        with open(os.path.join(data_root, 'val', img_name + '.xml')) as f:
            anno = BeautifulSoup(''.join(f.readlines()), "lxml")
        ground_truth['image_sizes'][img_name] = (int(anno.find('size').height.contents[0]), int(anno.find('size').width.contents[0]))
        bboxes = anno.findAll('object')
        cur_bboxes = []
        for bbox_idx, bbox in enumerate(bboxes):
            category = ground_truth['class_words'].index(str(bbox.find('name').contents[0]))
            # if bbox_idx == 0:
            #     category_ori = category
            cur_bboxes.append((int(bbox.xmin.contents[0]), int(bbox.ymin.contents[0]),
                                              int(bbox.xmax.contents[0]), int(bbox.ymax.contents[0])))
            # ground_truth['gt_bboxes'].append((int(bbox.xmin.contents[0]), int(bbox.ymin.contents[0]),
            #                                   int(bbox.xmax.contents[0]), int(bbox.ymax.contents[0])))
            # if category_ori != category:
            #     print('differ!!')
        # print('')
        ground_truth['gt_labels'].append(category)
        ground_truth['gt_bboxes'][img_name] = cur_bboxes
    # ground_truth['gt_bboxes'] = np.array(ground_truth['gt_bboxes'])
    return ground_truth


def load_image_imagenet(image_name, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image_raw = Image.open(image_name).convert("RGB")
    image_normalized = torch.from_numpy(np.array(image_raw)).permute(2, 0, 1).cuda().float() / 255.0
    image_normalized = (image_normalized - torch.Tensor(mean).cuda().view(3, 1, 1)) \
        / torch.Tensor(std).cuda().view(3, 1, 1)
    input_var = Variable(image_normalized.unsqueeze(0), volatile=True)
    return image_raw, input_var


def draw_bboxes(img, bboxes, class_names, width=3, font_size=20, color=(255, 255, 0)):
    img = img.copy()
    # fnt = ImageFont.truetype('arial.ttf', font_size)
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(img)
        draw.line((xmin, ymin, xmax, ymin), fill=color, width=width)
        draw.line((xmax, ymin, xmax, ymax), fill=color, width=width)
        draw.line((xmin, ymax, xmax, ymax), fill=color, width=width)
        draw.line((xmin, ymin, xmin, ymax), fill=color, width=width)
        # draw.text((xmin, ymin), '{}({:.2f})'.format(class_names[int(class_idx)], score), font=fnt, fill=color)
    return img

import itertools

# my Rectangle = (x1, y1, x2, y2), a bit different from OP's x, y, w, h
def intersection(rectA, rectB): # check if rect A & B intersect
    a, b = rectA, rectB
    startX = max(min(a[0], a[2]), min(b[0], b[2]))
    startY = max(min(a[1], a[3]), min(b[1], b[3]))
    endX = min(max(a[0], a[2]), max(b[0], b[2]))
    endY = min(max(a[1], a[3]), max(b[1], b[3]))
    if startX < endX and startY < endY:
        return True
    else:
        return False


def combineRect(rectA, rectB): # create bounding box for rect A & B
    a, b = rectA, rectB
    startX = min(a[0], b[0])
    startY = min(a[1], b[1])
    endX = max(a[2], b[2])
    endY = max(a[3], b[3])
    return (startX, startY, endX, endY)


def checkIntersectAndCombine(rects):
    if rects is None:
        return None
    mainRects = rects
    noIntersect = False
    while noIntersect == False and len(mainRects) > 1:
        # mainRects = list(set(mainRects))
        # get the unique list of rect, or the noIntersect will be
        # always true if there are same rect in mainRects
        newRectsArray = []
        for rectA, rectB in itertools.combinations(mainRects, 2):
            newRect = []
            if intersection(rectA, rectB):
                newRect = combineRect(rectA, rectB)
                newRectsArray.append(newRect)
                noIntersect = False
                # delete the used rect from mainRects
                if rectA in mainRects:
                    mainRects.remove(rectA)
                if rectB in mainRects:
                    mainRects.remove(rectB)
        if len(newRectsArray) == 0:
            # if no newRect is created = no rect in mainRect intersect
            noIntersect = True
        else:
            # loop again the combined rect and those remaining rect in mainRects
            mainRects = mainRects + newRectsArray
    return mainRects

import pickle

if __name__ == '__main__':
    # ground_truth_imagenet = load_ground_truth_imagenet('../data/IMAGENET')

    # with open('../data/IMAGENET/gt_imanget', 'wb') as f:
    #     pickle.dump(ground_truth_imagenet, f)


    with open('./imagenetmeta/gt_imagenet', 'rb') as f:
        ground_truth_imagenet = pickle.load(f)

    print(ground_truth_imagenet['image_list'][:100])
    print(len(ground_truth_imagenet['image_list']))
    print(ground_truth_imagenet['gt_bboxes']['ILSVRC2012_val_00000001'])

    # image_name = 'ILSVRC2012_val_00002005'
    # categories = ['n03804744']
    # img = Image.open(os.path.join('../data/IMAGENET/n03804744', image_name + '.JPEG'))
    # group_rects = checkIntersectAndCombine(ground_truth_imagenet['gt_bboxes'][image_name])
    # img_draw = draw_bboxes(img, group_rects, ground_truth_imagenet['class_names'])
    #
    # plt.rcParams["figure.figsize"] = (8, 8)
    # plt.imshow(img_draw)
    # plt.show()
