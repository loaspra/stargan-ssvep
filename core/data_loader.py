"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
# from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
# from torchvision.datasets import ImageFolder

from matplotlib import pyplot as plt

# Modifying the data loader to work with our dataset (EEG time domain data)
def listdir(dname):
    
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                            for ext in ['npy', 'csv']]))

    return fnames


class DefaultDataset(data.Dataset):
    
    def __init__(self, root, transform=None, img_size = 1024):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.img_size = img_size
        # self.targets should contain the name of the folders inside root
        self.targets = [int(os.path.basename(os.path.dirname(x))) for x in self.samples]

    def __getitem__(self, index):
        fname = self.samples[index]
        img = np.load(fname, allow_pickle=True) # (3, 1024)
        img = img.astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)
        
        img = img.reshape((3, self.img_size))

        return img, self.targets[index]
        

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None, img_size = 1024):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform
        self.img_size = img_size

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels


    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = np.load(fname)
        img2 = np.load(fname2)

        # COnvert both to numpy float32
        img = img.astype(np.float32)
        img2 = img2.astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)

        img = img.reshape((3, self.img_size))
        img2 = img2.reshape((3, self.img_size))

        return img, img2, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

def resize_signal(x):
    # add padding to the signal, so the final shape is a power of 2
    x = np.pad(x, ((0, 0), (0, (2**int(np.ceil(np.log2(x.shape[-1]))) - x.shape[-1]))), 'constant')
    return x

def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    transform = transforms.Compose([
        # resize_signal,

        transforms.ToTensor(),
        transforms.Normalize(mean=[0],
                             std=[1]),
    ])

    # As we dont work with images, we cannot use the ImageFolder class
    if which == 'source':
        dataset = DefaultDataset(root, transform, img_size=img_size)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform, img_size=img_size)
    else:
        raise NotImplementedError

    
    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, img_size=1000, batch_size=32,
                    imagenet_normalize=False, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')

    imagenet_normalize = False # change
    
    if imagenet_normalize:
        img_size = 512
        mean = [0.485]
        std = [0.229]
    else:
        mean = [0]
        std = [1]


    transform = transforms.Compose([
        # resize_signal,

        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform, img_size=img_size)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')


    transform = transforms.Compose([
        # resize_signal,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0],
                             std=[1]),
    ])

    # dataset = ImageFolder(root, transform)
    # As we dont work with images, we cannot use the ImageFolder class
    dataset = DefaultDataset(root, transform, img_size=img_size)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})