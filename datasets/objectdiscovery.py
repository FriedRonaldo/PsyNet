import torch.utils.data as data
import numpy as np
import os
from cub200 import pil_loader
from glob import glob


class ObjectDiscovery(data.Dataset):
    def __init__(self, data_dir, is_train, data_type='ODHORSE', transform=None, with_id=False):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        assert data_type in ['ODHORSE', 'ODCAR', 'ODAIRPLANE']

        self.transform = transform
        self.with_id = with_id
        self.data_dir = os.path.join(data_dir, 'ObjectDiscovery-data', 'Data')

        if data_type == 'ODHORSE':
            data_type_name = 'Horse'
        elif data_type == 'ODCAR':
            data_type_name = 'Car'
        elif data_type == 'ODAIRPLANE':
            data_type_name = 'Airplane'

        self.test_dir = sorted(glob(os.path.join(self.data_dir, data_type_name + 'Clean', '*.jpg')))
        self.tot_dir = sorted(glob(os.path.join(self.data_dir, data_type_name, '*.jpg')))
        self.size_dir = os.path.join(self.data_dir, data_type_name + '_size.txt')
        self.bbox_dir = os.path.join(self.data_dir, data_type_name + '_bbox.txt')

        self.test_tmp = []
        self.tot_tmp = []

        for k in range(len(self.test_dir)):
            self.test_tmp.append(self.test_dir[k].split('/')[-1])
        for k in range(len(self.tot_dir)):
            self.tot_tmp.append(self.tot_dir[k].split('/')[-1])

        self.test_dir_set = set(self.test_tmp)
        self.tot_dir_set = set(self.tot_tmp)
        self.train_dir_set = self.tot_dir_set.difference(self.test_dir_set)

        self.train_dir = sorted(list(self.train_dir_set))

        for k in range(len(self.train_dir)):
            self.train_dir[k] = os.path.join(self.data_dir, data_type_name, self.train_dir[k])

        if is_train:
            self.image_dir = self.train_dir
        else:
            self.image_dir = self.test_dir

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        img = pil_loader(self.image_dir[idx])

        imgid = self.image_dir[idx].split('/')[-1].replace('.jpg', '')

        if self.transform:
            img = self.transform(img)

        if self.with_id:
            return img, 0, imgid
        else:
            return img, 0

    def load_bboxes(self, img_size=224):
        with open(self.size_dir, 'r') as f:
            sizes = f.readlines()
        with open(self.bbox_dir, 'r') as f:
            bbox = f.readlines()
        bboxes = dict()
        for idx in range(len(sizes)):
            id, w, h = sizes[idx].strip().split()
            w, h = int(w), int(h)

            x1, y1, x2, y2 = bbox[idx].strip().split()[1:]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            x_scale = img_size / w
            y_scale = img_size / h

            x_new = int(np.round(x1 * x_scale))
            y_new = int(np.round(y1 * y_scale))
            x_max = int(np.round(x2 * x_scale))
            y_max = int(np.round(y2 * y_scale))

            bboxes[id] = [x_new, y_new, x_max, y_max]
        return bboxes

if __name__ == '__main__':
    db = ObjectDiscovery('../../data/', False, with_id=True)
    print(db.load_bboxes())

