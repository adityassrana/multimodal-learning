from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance

import numpy as np

import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode

from torch.autograd import Variable


import string
import os
import os.path
import numbers

__all__ = [ "ImageFolder","Compose", "ToTensor", "Normalize", "Resize", "Scale", 
			"CenterCrop", "RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop", "imshow"]



# Data loader for (RGB, HHA) pairs
# Based on torch.utils.data.ImageFolder
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
#     print(dir)
#     print(os.listdir(dir))
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    dirRGB = os.path.join(dir, 'rgb')
#     dirDepth = os.path.expanduser(dirDepth)
    for target in sorted(os.listdir(dirRGB)):
#         print(target)
        dRGB = os.path.join(dirRGB, target)
#         d = os.path.join(dir, target)
        if not os.path.isdir(dRGB):
            continue

        for root, _, fnames in sorted(os.walk(dRGB)):
#             print(root)
            
            for fname in sorted(fnames):
                if is_image_file(fname):
                    pathRGB = os.path.join(root, fname)
                    pathDepth = pathRGB.replace('/rgb/','/hha/') 
                    item = (pathRGB, pathDepth, class_to_idx[target])
#                    item = (pathRGB, class_to_idx[target])
                    images.append(item)
#                    print(type(images))

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFolder(data.Dataset):
    """An RGB-D data loader where the images are arranged in this way: ::
        root/rgb/bedroom/xxx.png
        root/rgb/bedroom/xxy.png
        root/rgb/bedroom/xxz.png
        root/rgb/kitchen/123.png
        root/rgb/kitchen/nsdf3.png
        root/rgb/kitchen/asd932_.png
        ...
        root/hha/bedroom/xxx.png
        root/hha/bedroom/xxy.png
        root/hha/bedroom/xxz.png
        root/hha/kitchen/123.png
        root/hha/kitchen/nsdf3.png
        root/hha/kitchen/asd932_.png
        ...
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        
        rootRGB = os.path.join(root, 'rgb')
        classes, class_to_idx = find_classes(rootRGB) # Use RGB as reference. Depth/HHA must replicate same structure and file names

        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        pathRGB, pathDepth, target = self.imgs[index]
        imgRGB = self.loader(pathRGB)
        imgDepth = self.loader(pathDepth)

        
        
        if self.transform is not None:
            imgRGB, imgDepth = self.transform(imgRGB, imgDepth)
#            img_pair  = self.transform(img_pair)
        if self.target_transform is not None:
            target = self.target_transform(target)

#        return img_pair, target
        return imgRGB, imgDepth, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        # tmp = '    Transforms (if any): '
        # fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


# RGB-D transforms

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgRGB, imgDepth):
        for t in self.transforms:
            imgRGB, imgDepth = t(imgRGB, imgDepth)
        return imgRGB, imgDepth

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor(object):
    def __call__(self, imgRGB, imgDepth):
        return F.to_tensor(imgRGB), F.to_tensor(imgDepth)

    def __repr__(self):
        return self.__class__.__name__ + '()'
        
class Normalize(object):
    def __init__(self, meanRGB, stdRGB, meanDepth, stdDepth):
        self.meanRGB = meanRGB
        self.stdRGB = stdRGB
        self.meanDepth = meanDepth
        self.stdDepth = stdDepth

    def __call__(self, tensorRGB, tensorDepth):
       return F.normalize(tensorRGB, self.meanRGB, self.stdRGB), F.normalize(tensorDepth, self.meanDepth, self.stdDepth)

    def __repr__(self):
        return self.__class__.__name__ + '(meanRGB={0}, stdRGB={1},meanDepth={2}, stdDepth={3})'.format(
                self.meanRGB, self.stdRGB, self.meanDepth, self.stdDepth)
        
class Resize(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgRGB, imgDepth):
        return F.resize(imgRGB, self.size, self.interpolation), F.resize(imgDepth, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)
    
class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgRGB, imgDepth):
        return F.center_crop(imgRGB, self.size), F.center_crop(imgDepth, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, imgRGB, imgDepth):
        if self.padding > 0:
            imgRGB = F.pad(imgRGB, self.padding)
            imgDepth = F.pad(imgDepth, self.padding)

        i, j, h, w = self.get_params(imgRGB, self.size)

        return F.crop(imgRGB, i, j, h, w), F.crop(imgDepth, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgRGB, imgDepth):
        if random.random() < self.p:
            return F.hflip(imgRGB), F.hflip(imgDepth)
        return imgRGB, imgDepth

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgRGB, imgDepth):
        if random.random() < self.p:
            return F.vflip(imgRGB), F.vflip(imgDepth)
        return imgRGB, imgDepth

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, imgRGB, imgDepth):
        i, j, h, w = self.get_params(imgRGB, self.scale, self.ratio)
        return F.resized_crop(imgRGB, i, j, h, w, self.size, self.interpolation), F.resized_crop(imgDepth, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(round(self.scale, 4))
        format_string += ', ratio={0}'.format(round(self.ratio, 4))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

# Visualize RGB-D images
def imshow(imgRGB, imgDepth, title=None, concat_vert=False, mean_RGB=[0.485, 0.456, 0.406], 
               mean_depth=[0.485, 0.456, 0.406], std_RGB=[0.229, 0.224, 0.225], 
               std_depth=[0.229, 0.224, 0.225], show_img=True):
    """Imshow for RGB-D data."""
    imgRGB = imgRGB.numpy().transpose((1, 2, 0))
    imgRGB = np.clip(np.array(std_RGB) * imgRGB + np.array(mean_RGB), 0, 1)
    imgDepth = imgDepth.numpy().transpose((1, 2, 0))
    imgDepth = np.clip(np.array(std_depth) * imgDepth + np.array(mean_depth), 0, 1)
    if concat_vert:
        img = np.concatenate((imgRGB, imgDepth),axis=0)
    else:
        img = np.concatenate((imgRGB, imgDepth),axis=1)
    
    if show_img:
        plt.imshow(img)
        if title is not None:
            plt.title(title)
            
        plt.axis('off')
#        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.show()
    else:
        return img





if __name__ == "__main__":
    # Test RGB-D data loader, transforms and utilities
    RGB_AVG, RGB_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # Default ImageNet ILSRVC2012
    DEPTH_AVG, DEPTH_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # Default ImageNet ILSRVC2012
    
    import matplotlib.pyplot as plt
    import torch
    import torchvision
    from torchvision import datasets, models, transforms
    import numpy as np
    import os
    import utils
    
    data_dir = 'sunrgbd/256'
    data_dir = 'sunrgbd/256_lite'
    
#    data_transforms = Compose([Resize(224),
#                                   RandomHorizontalFlip(),
#                                   ToTensor(), 
#                                   Normalize(RGB_AVG, RGB_STD,DEPTH_AVG, DEPTH_STD)])
#    data_transforms = Compose([RandomResizedCrop(224),
#                                   ToTensor(), 
#                                   Normalize(RGB_AVG, RGB_STD,DEPTH_AVG, DEPTH_STD)])
    data_transforms = Compose([CenterCrop(224),
                                   ToTensor(), 
                                   Normalize(RGB_AVG, RGB_STD,DEPTH_AVG, DEPTH_STD)])

    rgbd_dataset = ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
    dataloader = torch.utils.data.DataLoader(rgbd_dataset, batch_size=4,shuffle=True, num_workers=4)
    class_names = rgbd_dataset.classes

    print(class_names)
    
    rgbd_iter = iter(dataloader)

    
    # Get a batch of training data
    imgsRGB, imgsDepth, labels = next(rgbd_iter)

    # Make a grid from batch
    outRGB = torchvision.utils.make_grid(imgsRGB)
    outDepth = torchvision.utils.make_grid(imgsDepth)
#
    imshow(outRGB, outDepth, concat_vert=True, show_img=True)
#    imshow(imgsRGB[0], imgsDepth[0], concat_vert=False)

    # from torchvision import models
    # inputs = torch.randn(1,3,224,224)
    # resnet18 = models.resnet18()
    # y = resnet18(Variable(inputs))
    # print(y)
