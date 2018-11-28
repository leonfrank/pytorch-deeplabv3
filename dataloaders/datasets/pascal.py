import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import sys
from PIL import Image
from torchvision import transforms
from dataloaders import transforms as tr
class VOCDataset(Dataset):
    def __init__(self, args, mode, year = '2007'):
        self.root = args.root
        self.mode = mode
        self.year = year
        self.image_path = os.path.join(self.root, 'JPEGImages')
        self.task = 'Segmentation'
        self.image_ids = self.load_ids(self.mode)
        self.images = []
        self.categories = []
        self.args = args
        for image_id in self.image_ids:
            img = os.path.join(self.image_path, image_id + '.jpg')
            cat = os.path.join(self.root, 'SegmentationClass', image_id + '.png')
            assert os.path.isfile(img)
            assert os.path.isfile(cat)
            self.categories.append(cat)
            self.images.append(img)
        
    def __len__(self):
        return len(self.image_ids)        
    
    def __str__(self):
        return 'VOC{year}(split={mode})'.format(year = self.year, mode = self.mode)

    def load_ids(self, mode):
        image_f = os.path.join(self.root, 'ImageSets', self.task, self.mode + '.txt')
        image_ids = []
        with open(image_f, 'r') as f:
            for line in f:
                image_ids += [line.strip('\n')]
        return image_ids
        
    def _get_img_label(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _label = Image.open(self.categories[index])
        
        return _img, _label

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def __getitem__(self, idx):
        idx = idx % self.__len__()
        img, label = self._get_img_label(idx)
        ret = {'image': img, 'label': label}
        if self.mode == 'train':
            ret = self.transform_tr(ret)
        elif self.mode == 'val':
            ret = self.transform_val(ret)
        else:
            ret = tr.ToTensor()(ret)
        return ret
        


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCDataset('/data/mc_data/VOC2007' ,'train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)