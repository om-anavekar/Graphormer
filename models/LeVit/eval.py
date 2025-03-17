# Code modified from LeViT repo

import numpy as np
import torch
import json
import os
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from pathlib import Path

from timm.models import create_model

import sys
sys.path.insert(0, os.path.abspath("LeViT"))


# # from LeVit import *
import LeViT.engine
from LeViT.engine import evaluate
import LeViT.levit
import LeViT.levit_c
import LeViT.utils

device = "cuda"
imagenet_path = "/scratch/graphormer_imagenet/imagenet/val/"
model_name = "LeViT_256"

def build_transform():
    t = []
    if True:
        size = int((256 / 224) * 224)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(224))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def main():

    # dataset_train, argss.nb_classes = build_dataset(is_train=True, args=args)
    # dataset_val, _ = build_dataset(is_train=False, args=args)
    # root = os.path.join(imagenet_path, 'val')
    transform = build_transform()
    dataset_val = datasets.ImageFolder(imagenet_path, transform=transform)

    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    print(f"Creating model: {model_name}")
    model = create_model(
        model_name,
        num_classes=1000,
        distillation=True,
        pretrained=True,
        fuse=True,
    )

    model = model.to(device)

    test_stats = evaluate(data_loader_val, model, device)

    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

   
if __name__ == '__main__':
    main()
