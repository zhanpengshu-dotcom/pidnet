import os
import cv2
import numpy as np
import torch
from .base_dataset import BaseDataset


class custom(BaseDataset):
    def __init__(self, root, list_path, num_samples=None, num_classes=3, multi_scale=True, flip=True, ignore_label=255, base_size=512, crop_size=(512, 512), scale_factor=16, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], **kwargs):

        super(custom, self).__init__(ignore_label, base_size, crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip().split() for line in open(os.path.join(root, list_path))]
        self.files = self.read_files()

        self.label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        self.class_weights = None

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path = item[0]
            label_path = item[1]
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(os.path.join(self.root, item["img"]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(os.path.join(self.root, item["label"]), cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        if image.shape[0] != 512 or image.shape[1] != 512:
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)

        image, label, edge = self.gen_sample(image, label, self.multi_scale, self.flip)
        return image.copy(), label.copy(), edge.copy(), np.array(index), name
