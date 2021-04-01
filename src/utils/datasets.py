from __future__ import print_function

import os

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

mean = {
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4867, 0.4408),
    "tiny_imagenet": (0.485, 0.456, 0.406),
}
std = {
    "cifar10": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2675, 0.2565, 0.2761),
    "tiny_imagenet": (0.229, 0.224, 0.225),
}


def get_data(args):
    """Applies general preprocess transformations to Datasets
    ops -
    transforms.Normalize = Normalizes each channel of the image
    args - mean tensor, standard-deviation tensor
    transforms.RandomCrops - crops the image at random location
    transforms.HorizontalFlip - randomly flips the image
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[args.dataset], std[args.dataset]),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean[args.dataset], std[args.dataset]),
        ]
    )
    imagenet_transform_train = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[args.dataset], std[args.dataset]),
        ]
    )

    imagenet_transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean[args.dataset], std[args.dataset]),
        ]
    )

    if args.dataset == "mnist":
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2
        )

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=4
        )
        args.num_classes = 10

    elif args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=4
        )
        args.num_classes = 100

    elif args.dataset == "tiny_imagenet":
        data_dir = "tiny-imagenet-200/"
        num_workers = {"train": 100, "val": 0, "test": 0}
        data_transforms = {
            "train": imagenet_transform_train,
            "val": imagenet_transform_test,
            "test": imagenet_transform_test,
        }
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
            for x in ["train", "val", "test"]
        }
        dataloaders = {
            x: data.DataLoader(
                image_datasets[x],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers[x],
            )
            for x in ["train", "val", "test"]
        }
        trainloader = dataloaders["train"]
        testloader = dataloaders["test"]
        args.num_classes = 200

    return trainloader, testloader
