'''
Init Dataloader
'''
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
from torch.utils.data import RandomSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import torch
import os

from typing import Any, Callable, Sequence


class Dataset:
    def __init__(self, samples: Sequence, num_classes: int, preprocess: Callable) -> None:
        self.samples = samples
        self.num_classes = num_classes
        self.preprocess = preprocess

    def __getitem__(self, index: int) -> Any:
        data = self.preprocess(self.samples[index])
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        fmt_str = "Number of datapoints: {}\tNumber of classes: {}\n".format(len(self), self.num_classes)
        return fmt_str


def opencv_decoder(data_path, input_size, train=True):
    img = cv2.imread(data_path)
    if train:
        flip_code = np.random.randint(2)
        if flip_code == 1:
            img = cv2.flip(img, flip_code)
    return img


class TestPreprocess(object):
    def __init__(self, input_size, bmk_root):
        self.input_size = input_size
        self.bmk_root = bmk_root

    def __call__(self, sample):
        img = opencv_decoder(os.path.join(self.bmk_root, sample["path"]), self.input_size, train=False)
        return img, 1


def prepare_emore_bmk(root: str) -> [dict]:
    samples_meta = np.load(os.path.join(root, "align5p.npy"))
    samples = []
    for x in samples_meta:
        samples.append({"path": x, "label": 1})
    issame = np.load(os.path.join(root, "list.npy"))
    return (samples, len(samples_meta)), {'issame': issame}


def get_test_dataset(conf, bmk_name):
    test_process = TestPreprocess(conf.input_size, conf.data_path)
    (samples, num_classes), metadata_for_eval = prepare_emore_bmk(root=os.path.join(conf.data_path, bmk_name))
    test_dataset = Dataset(samples, num_classes, test_process)

    test_kwargs = {
        "num_workers": conf.num_workers,
        "pin_memory": True}

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        timeout=14400,
        shuffle=False,  # bmk要顺序读取
        drop_last=False,
        batch_size=conf.batch_size,
        worker_init_fn=None,
        **test_kwargs
    )
    return {"test_dataset_iter": test_loader, "num_class": num_classes, "name": bmk_name,
            'issame': metadata_for_eval['issame']}


def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_train_loader(conf):
    ds, class_num = get_train_dataset(conf.emore_folder / 'imgs')
    random_sampler = RandomSampler(ds, replacement=True)
    print("batchsize = ", conf.batch_size)
    loader = DataLoader(ds, batch_size=conf.batch_size,
                        # shuffle=True,
                        sampler=random_sampler,
                        pin_memory=conf.pin_memory,
                        num_workers=conf.num_workers,
                        drop_last=True)
    return loader, class_num
