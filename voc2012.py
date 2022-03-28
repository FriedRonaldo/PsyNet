import torch.utils.data as data
import scipy.io
import numpy as np
import os

from PIL import Image
from convert2Yolo.Format import YOLO as cvtYOLO
from convert2Yolo.Format import VOC as cvtVOC


# reference code : https://deepbaksuvision.github.io/Modu_ObjectDetection/posts/03_01_dataloader.html
class VOC2012(data.Dataset):

    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = '.jpg'

    # def __init__(self, data_dir, is_train, transform=None, with_id=False):
    def __init__(self, root, train=True, transform=None, target_transform=None, resize=448,
                 class_path='../data/VOC2012/voc.names'):
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

        with open(class_path) as f:
            self.classes = f.read().splitlines()

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        self.data = self.cvtData()

        print(self.classes)
        # self.transform = transform
        # self.with_id = with_id
        # self.data_dir = os.path.join(data_dir, 'stanfordCar')
        # self.annoroot_dir = os.path.join(self.data_dir, 'devkit')

        # if is_train:
        #     self.image_dir = os.path.join(self.data_dir, 'cars_train')
        #     self.anno_dir = os.path.join(self.annoroot_dir, 'cars_train_annos.mat')
        #     self.size_dir = os.path.join(self.annoroot_dir, 'cars_train_size.txt')
        # else:
        #     self.image_dir = os.path.join(self.data_dir, 'cars_test')
        #     self.anno_dir = os.path.join(self.annoroot_dir, 'cars_test_annos_withlabels.mat')
        #     self.size_dir = os.path.join(self.annoroot_dir, 'cars_test_size.txt')
        #
        # # x1 y1 x2 y2 label filename
        # self.annotations = scipy.io.loadmat(self.anno_dir)
        # self.annotations = self.annotations['annotations'][0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        key = list(self.data[index].keys())[0]

        img = Image.open(key).convert('RGB')
        current_shape = img.size
        img = img.resize((self.resize_factor, self.resize_factor))

        target = self.data[index][key]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            # Future works
            pass

        return img, target, current_shape

    def _check_exists(self):
        print("Image Folder : {}".format(os.path.join(self.root, self.IMAGE_FOLDER)))
        print("Label Folder : {}".format(os.path.join(self.root, self.LABEL_FOLDER)))

        return os.path.exists(os.path.join(self.root, self.IMAGE_FOLDER)) and \
               os.path.exists(os.path.join(self.root, self.LABEL_FOLDER))

    def cvtData(self):

        result = []
        voc = cvtVOC()

        yolo = cvtYOLO(os.path.abspath(self.class_path))
        flag, self.dict_data = voc.parse(os.path.join(self.root, self.LABEL_FOLDER))

        try:

            if flag:
                flag, data = yolo.generate(self.dict_data)

                keys = list(data.keys())
                keys = sorted(keys, key=lambda key: int(key.split("_")[-1]))

                for key in keys:
                    contents = list(filter(None, data[key].split("\n")))
                    target = []
                    for i in range(len(contents)):
                        tmp = contents[i]
                        tmp = tmp.split(" ")
                        for j in range(len(tmp)):
                            tmp[j] = float(tmp[j])
                        target.append(tmp)

                    result.append(
                        {os.path.join(self.root, self.IMAGE_FOLDER, "".join([key, self.IMG_EXTENSIONS])): target})

                return result

        except Exception as e:
            raise RuntimeError("Error : {}".format(e))

    def load_bboxes(self, img_size=224):
        size_file = open(self.size_dir, 'r')
        sizes = size_file.readlines()
        size_idx = 0
        bboxes = dict()
        for idx in range(len(self.annotations)):
            w, h = int(sizes[size_idx].strip().split(' ')[0]), int(sizes[size_idx].strip().split(' ')[1])
            size_idx += 1
            x1, y1, x2, y2 = [self.annotations[idx][0].item(), self.annotations[idx][1].item(),
                              self.annotations[idx][2].item(), self.annotations[idx][3].item()]

            x_scale = img_size / w
            y_scale = img_size / h

            x_new = int(np.round(x1 * x_scale))
            y_new = int(np.round(y1 * y_scale))
            x_max = int(np.round(x2 * x_scale))
            y_max = int(np.round(y2 * y_scale))

            bboxes[self.annotations[idx][-1].item()] = [x_new, y_new, x_max, y_max]
        return bboxes


if __name__ == '__main__':
    test = VOC2012(root='../data/VOC2012')
    print(test.__getitem__(0))