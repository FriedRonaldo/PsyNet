import torch.utils.data as data
import numpy as np
import os

from PIL import Image


class AIRCRAFT(data.Dataset):

    IMAGE_FOLDER = "AIRCRAFT/data/images"
    DATA_FOLDER = "AIRCRAFT/data"
    IMG_EXTENSIONS = '.jpg'
    # CLASS_FOLDER = "ImageSets/Main"

    # def __init__(self, data_dir, is_train, transform=None, with_id=False):
    def __init__(self, root, train='train', transform=None, resize=224, with_id=False):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.train = train
        self.resize_factor = resize
        self.with_id = with_id

        self.ground_truth = dict(
            image_list=[],
            image_sizes={},
            gt_bboxes={},
        )

        if train == 'train':
            file_path = os.path.join(self.root, self.DATA_FOLDER, 'images_train_size.txt')
        elif train == 'val':
            file_path = os.path.join(self.root, self.DATA_FOLDER, 'images_val_size.txt')
        elif train == 'test':
            file_path = os.path.join(self.root, self.DATA_FOLDER, 'images_test_size.txt')

        bbox_file_path = os.path.join(self.root, self.DATA_FOLDER, 'images_box.txt')

        # get metadata about image_list / image_size
        with open(file_path) as f:
            for line in f:
                img_name, w, h = line.split()
                self.ground_truth['image_list'].append(img_name)
                self.ground_truth['image_sizes'][img_name] = (float(w), float(h))

        # get metadata about image gt_bboxes
        with open(bbox_file_path) as f:
            for line in f:
                img_name, x1, y1, x2, y2 = line.split()
                self.ground_truth['gt_bboxes'][img_name] = (float(x1), float(y1), float(x2), float(y2))

    def __len__(self):
        return len(self.ground_truth['image_list'])

    def __getitem__(self, index):

        img_id = self.ground_truth['image_list'][index]
        img_path = os.path.join(self.root, self.IMAGE_FOLDER, img_id + self.IMG_EXTENSIONS)

        img = Image.open(img_path).convert('RGB')
        # img = img.resize((self.resize_factor, self.resize_factor))

        # TODO : Not yet implemented about target
        target = [0.0]

        if self.transform is not None:
            img = self.transform(img)

        if self.with_id:
            return img, target, img_id

        return img, target

    def load_bboxes(self, img_size=224):
        bboxes = {}

        for img_name in self.ground_truth['image_list']:

            x1, y1, x2, y2 = self.ground_truth['gt_bboxes'][img_name]
            w, h = self.ground_truth['image_sizes'][img_name]
            x_scale = img_size / w
            y_scale = img_size / h

            x_new = int(np.round(x1 * x_scale))
            y_new = int(np.round(y1 * y_scale))
            x_max = int(np.round(x2 * x_scale))
            y_max = int(np.round(y2 * y_scale))

            bboxes[img_name] = (x_new, y_new, x_max, y_max)

        return bboxes


if __name__ == '__main__':
    test = AIRCRAFT(root='../data', with_id=True)
    img, target, id = test.__getitem__(0)
    # print(test.ground_truth['gt_bboxes'])
    print(test.load_bboxes())
