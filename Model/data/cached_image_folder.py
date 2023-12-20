# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import io
import os
import time
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image

from .zipreader import is_zip_path, ZipReader

# has_file_allowed_extension: This function checks if a file has an allowed extension 
# (like .jpg, .png, etc.). It takes a filename and a list of extensions as input and 
# returns True if the  file's extension matches any in the list.

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


# find_classes: This function identifies all class folders in a given directory. 
# It returns two things: a list of class names sorted alphabetically, and a dictionary 
# mapping each class name to its index in the sorted list.

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

# make_dataset: This function creates a dataset from a directory structure where each 
# subfolder represents a class. It compiles a list of tuples, each containing the path 
# to an image file and its corresponding class index.

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

# make_dataset_with_ann: Similar to make_dataset, but it uses an annotation file to create 
# the dataset. The annotation file contains image file paths and their class indices. 
# This function is useful when the dataset structure doesn't follow the standard directory layout.

def make_dataset_with_ann(ann_file, img_prefix, extensions):
    images = []
    with open(ann_file, "r") as f:
        contents = f.readlines()
        for line_str in contents:
            path_contents = [c for c in line_str.split('\t')]
            im_file_name = path_contents[0]
            class_index = int(path_contents[1])

            assert str.lower(os.path.splitext(im_file_name)[-1]) in extensions
            item = (os.path.join(img_prefix, im_file_name), class_index)

            images.append(item)

    return images

# DatasetFolder (class): A generic dataset class for loading data where the structure 
# is root/class_x/xxx.ext. It includes methods for dataset initialization, fetching data samples, 
# and representation. It can work with different image file extensions and supports optional 
# transformations and caching.

class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    # DatasetFolder.__init__: Initializes a dataset where samples are organized by class subfolders within a 
    # root directory. It sets up the sample paths, transformations, and target transformations.

    def __init__(self, root, loader, extensions, ann_file='', img_prefix='', transform=None, target_transform=None,
                 cache_mode="no"):
        # Initialization logic based on whether an annotation file is provided.
        if ann_file == '':
            _, class_to_idx = find_classes(root)
            samples = make_dataset(root, class_to_idx, extensions)
        # zip mode
        else:
            samples = make_dataset_with_ann(os.path.join(root, ann_file),
                                            os.path.join(root, img_prefix),
                                            extensions)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.labels = [y_1k for _, y_1k in samples]
        self.classes = list(set(self.labels))

        self.transform = transform
        self.target_transform = target_transform

        self.cache_mode = cache_mode
        if self.cache_mode != "no":
            self.init_cache()
    
    # init_cache (method of DatasetFolder): Initializes a cache for dataset samples to improve data 
    # loading performance. It supports 'full' and 'part' caching modes.

    def init_cache(self):
        assert self.cache_mode in ["part", "full"]
        n_sample = len(self.samples)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        samples_bytes = [None for _ in range(n_sample)]
        start_time = time.time()
        for index in range(n_sample):
            if index % (n_sample // 10) == 0:
                t = time.time() - start_time
                print(f'global_rank {dist.get_rank()} cached {index}/{n_sample} takes {t:.2f}s per block')
                start_time = time.time()
            path, target = self.samples[index]
            if self.cache_mode == "full":
                samples_bytes[index] = (ZipReader.read(path), target)
            elif self.cache_mode == "part" and index % world_size == global_rank:
                samples_bytes[index] = (ZipReader.read(path), target)
            else:
                samples_bytes[index] = (path, target)
        self.samples = samples_bytes

    # __getitem__: Retrieves a data sample and its corresponding target.

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    # __len__: Returns the total number of samples in the dataset.

    def __len__(self):
        return len(self.samples)

    # __repr__: String representation of the dataset.

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

# pil_loader: Loads an image using PIL, supporting multiple source types.

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if isinstance(path, bytes):
        img = Image.open(io.BytesIO(path))
    elif is_zip_path(path):
        data = ZipReader.read(path)
        img = Image.open(io.BytesIO(data))
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
    return img.convert('RGB')

# accimage_loader: Attempts to use `accimage` for faster image loading, with a fallback to PIL.

# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)

# default_img_loader: Selects the appropriate image loading function based on the backend.

def default_img_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    return pil_loader(path)

# CachedImageFolder: Extends DatasetFolder to handle image datasets, supporting caching for efficient data loading.

class CachedImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

     #CachedImageFolder.__init__: Initializes the image dataset. It sets up the image folder with its path, 
    # annotation file, and image prefix. The method configures transformations, a loader for images, and an optional 
    # cache mode for optimized data loading.

    def __init__(self, root, ann_file='', img_prefix='', transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no"):
        super(CachedImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                                ann_file=ann_file, img_prefix=img_prefix,
                                                transform=transform, target_transform=target_transform,
                                                cache_mode=cache_mode)
        self.imgs = self.samples

    #CachedImageFolder.__getitem__: Retrieves an image and its corresponding class label by index. 
    # This method is essential for iterating over the dataset in data loaders.

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target