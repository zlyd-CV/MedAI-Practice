import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as nn
from pycocotools.coco import COCO
import swanlab

class COCOSegmentationDataset(Dataset):
    def __init__(self, coco, image_dir, transform=None):
        self.coco = coco
        self.image_dir = image_dir
        self.image_ids = coco.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]  # 获取图像的详细信息
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        # 加载图像并转化为numpy数组
        image = Image.open(image_path)
        image = np.array(image, dtype=np.uint8)

        # 创建掩码
        ann_ids = self.coco.getAnnIds(imgIds=image_id)  # 获取该图像对应的标注 ID 列表
        anns = self.coco.loadAnns(ann_ids)  # 加载标注信息
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        # 转换为张量并预处理
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
            mask = (nn.interpolate(mask.unsqueeze(0), size=(256, 256),
                                   mode='nearest').squeeze(0))  # 将掩码调整为 (256, 256) 的大小,去除通道维度

        return image, mask


def load_coco_datasets(transform):
    # 加载数据集路径和COCO标注文件路径
    train_dir = '../dataset/train'
    val_dir = '../dataset/valid'
    test_dir = '../dataset/test'
    train_annotation_file = '../dataset/train/_annotations.coco.json'
    test_annotation_file = '../dataset/test/_annotations.coco.json'
    val_annotation_file = '../dataset/valid/_annotations.coco.json'

    # 加载COCO标注数据数据集
    train_coco = COCO(train_annotation_file)
    val_coco = COCO(val_annotation_file)
    test_coco = COCO(test_annotation_file)

    # 创建数据集
    train_dataset = COCOSegmentationDataset(train_coco, train_dir, transform=transform)
    val_dataset = COCOSegmentationDataset(val_coco, val_dir, transform=transform)
    test_dataset = COCOSegmentationDataset(test_coco, test_dir, transform=transform)\

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=swanlab.config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=swanlab.config["batch_size"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=swanlab.config["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader
