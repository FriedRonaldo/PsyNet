import torch.utils.data as data
import scipy.io
import numpy as np
import os

from PIL import Image
from bs4 import BeautifulSoup

# reference code : https://deepbaksuvision.github.io/Modu_ObjectDetection/posts/03_01_dataloader.html
class VOC2007(data.Dataset):

    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = '.jpg'
    CLASS_FOLDER = "ImageSets/Main"

    # def __init__(self, data_dir, is_train, transform=None, with_id=False):
    def __init__(self, root, train=True, transform=None, target_transform=None, resize=224,
                 class_path='../data/VOC2007/voc.names', category='aeroplane', with_id=False):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize_factor = resize
        self.class_path = class_path
        self.with_id = with_id

        self.category = category
        self.ground_truth = dict(
            class_words=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                         'bus', 'car', 'cat', 'chair', 'cow',
                         'diningtable', 'dog', 'horse', 'motorbike', 'person',
                         'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
            image_list=[],
            image_sizes={},
            gt_bboxes={},
        )

        if train:
            file_path = os.path.join(self.root, self.CLASS_FOLDER, self.category + '_train.txt')
        else:
            file_path = os.path.join(self.root, self.CLASS_FOLDER, self.category + '_val.txt')

        with open(file_path) as f:
            for line in f:
                img_name, val = line.split()
                if val != '-1':
                    # self.file_path.append(os.path.join(self.root, self.IMAGE_FOLDER, name + '.jpg'))
                    # self.xml_path.append(os.path.join(self.root, self.LABEL_FOLDER, name + '.xml'))
                    with open(os.path.join(self.root, self.LABEL_FOLDER, img_name + '.xml')) as f:
                        anno = BeautifulSoup(''.join(f.readlines()), "lxml")
                    bboxes = anno.findAll('object')
                    cur_bboxes = []
                    for bbox_idx, bbox in enumerate(bboxes):
                        category = self.ground_truth['class_words'].index(str(bbox.find('name').contents[0]))
                        if str(bbox.find('name').contents[0]) != self.category:
                            continue
                        cur_bboxes.append((int(bbox.xmin.contents[0]), int(bbox.ymin.contents[0]),
                                           int(bbox.xmax.contents[0]), int(bbox.ymax.contents[0])))

                    self.ground_truth['image_list'].append(img_name)
                    self.ground_truth['image_sizes'][img_name] = (int(anno.find('size').height.contents[0]), int(anno.find('size').width.contents[0]))
                    # self.ground_truth['gt_labels'].append(category)
                    self.ground_truth['gt_bboxes'][img_name] = cur_bboxes

        # print(self.ground_truth['gt_bboxes'])
        # print(self.ground_truth['image_list'])
        # exit()
        # with open(class_path) as f:
        #     self.classes = f.read().splitlines()
        #
        # if not self._check_exists():
        #     raise RuntimeError("Dataset not found.")
        #
        # self.data = self.cvtData()

    def __len__(self):
        return len(self.ground_truth['image_list'])

    def __getitem__(self, index):

        img_id = self.ground_truth['image_list'][index]
        img_path = os.path.join(self.root, self.IMAGE_FOLDER, img_id + self.IMG_EXTENSIONS)

        img = Image.open(img_path).convert('RGB')
        # img = img.resize((self.resize_factor, self.resize_factor))

        target = self.category

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            # Future works
            pass

        if self.with_id:
            return img, target, img_id

        return img, target

    def load_bboxes(self, img_size=224):
        bboxes = {}
        for img_name in self.ground_truth['image_list']:
            new_bboxes = []
            for bbox in self.ground_truth['gt_bboxes'][img_name]:
                x1, y1, x2, y2 = bbox
                h, w = self.ground_truth['image_sizes'][img_name]
                x_scale = img_size / w
                y_scale = img_size / h

                x_new = int(np.round(x1 * x_scale))
                y_new = int(np.round(y1 * y_scale))
                x_max = int(np.round(x2 * x_scale))
                y_max = int(np.round(y2 * y_scale))

                new_bboxes.append([x_new, y_new, x_max, y_max])
            bboxes[img_name] = new_bboxes
        return bboxes        # bboxes = {annotation: [box, size] for annotation, box, _ in self.data}
        # for key, val in bboxes.items():
        #     x1, y1, x2, y2 = bboxes[key][0]
        #     w, h = bboxes[key][1]
        #     x_scale = img_size / w
        #     y_scale = img_size / h
        #
        #     x_new = int(np.round(x1 * x_scale))
        #     y_new = int(np.round(y1 * y_scale))
        #     x_max = int(np.round(x2 * x_scale))
        #     y_max = int(np.round(y2 * y_scale))
        #
        #     bboxes[key] = [x_new, y_new, x_max, y_max]
        # return bboxes


if __name__ == '__main__':
    test = VOC2007(root='../data/VOC2007', with_id=True)
    img, target, id = test.__getitem__(0)
    print(test.ground_truth['gt_bboxes'][id])