import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == True:

            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90,  expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90,  expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed)
        torch.manual_seed(seed)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        print(len(self.images), len(self.gts))
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            try:
                img = Image.open(img_path)
                gt = Image.open(gt_path)
                if img.size == gt.size:
                    images.append(img_path)
                    gts.append(gt_path)
            except Exception as e:
                # 如果PIL无法处理TIFF文件，尝试使用OpenCV检查
                if (img_path.lower().endswith('.tif') or img_path.lower().endswith('.tiff') or 
                    gt_path.lower().endswith('.tif') or gt_path.lower().endswith('.tiff')):
                    try:
                        img_cv = cv2.imread(img_path)
                        gt_cv = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                        if img_cv is not None and gt_cv is not None:
                            if img_cv.shape[:2] == gt_cv.shape[:2]:
                                images.append(img_path)
                                gts.append(gt_path)
                    except:
                        print(f"跳过无法读取的文件: {img_path} 或 {gt_path}")
                else:
                    print(f"跳过无法读取的文件: {img_path} 或 {gt_path}")

        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as e:
            # 如果PIL无法处理，尝试使用OpenCV
            if path.lower().endswith('.tif') or path.lower().endswith('.tiff'):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"无法读取图像文件: {path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img)
            else:
                raise e

    def binary_loader(self, path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('L')
        except Exception as e:
            # 如果PIL无法处理，尝试使用OpenCV
            if path.lower().endswith('.tif') or path.lower().endswith('.tiff'):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"无法读取图像文件: {path}")
                return Image.fromarray(img)
            else:
                raise e

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True, augmentation=False):

    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif') or f.endswith('.tiff')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif') or f.endswith('.tiff')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.jpg'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as e:
            # 如果PIL无法处理，尝试使用OpenCV
            if path.lower().endswith('.tif') or path.lower().endswith('.tiff'):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"无法读取图像文件: {path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img)
            else:
                raise e

    def binary_loader(self, path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('L')
        except Exception as e:
            # 如果PIL无法处理，尝试使用OpenCV
            if path.lower().endswith('.tif') or path.lower().endswith('.tiff'):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"无法读取图像文件: {path}")
                return Image.fromarray(img)
            else:
                raise e
