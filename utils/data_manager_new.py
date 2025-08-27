import logging
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import os

class DataManager(object):
    def __init__(self, args=None):
        self.args = args
        self.dataset_name = args['dataset']

    def get_dataset(self):
        if self.dataset_name == 'cifar':
            trainset, testset = self.build_dataset_cifar()
            return trainset, testset
        elif self.dataset_name == 'cifar10':
            trainset, testset = self.build_dataset_cifar10()
            return trainset, testset
        elif self.dataset_name == 'cars':
            trainset, testset,valset = self.build_dataset_cars()
            return trainset, testset
        elif self.dataset_name == 'dtd':
            trainset, testset = self.build_dataset_dtd()
            return trainset, testset
        elif self.dataset_name == 'sat':
            trainset, testset = self.build_dataset_sat()
            return trainset, testset
        elif self.dataset_name == 'aircraft':
            trainset, testset = self.build_dataset_aircraft()
            return trainset, testset
        elif self.dataset_name == 'flower':
            trainset, testset = self.build_dataset_flower()
            return trainset, testset
        elif self.dataset_name == 'nwpu':
            trainset, testset = self.build_dataset_nwpu()
            return trainset, testset
        elif self.dataset_name == 'pattern':
            trainset, testset = self.build_dataset_pattern()
            return trainset, testset
        elif self.dataset_name == 'Imagenet':
            trainset, testset = self.build_dataset_imagenet()
            return trainset, testset
        elif self.dataset_name == 'dog':
            trainset, testset = self.build_dataset_dog()
            return trainset, testset
        elif self.dataset_name == 'ucf':
            trainset, testset = self.build_dataset_ucf()
            return trainset, testset
        elif self.dataset_name == 'caltech101':
            trainset, testset = self.build_dataset_caltech101()
            return trainset, testset
        elif self.dataset_name == 'imagenetv2':
            testset = self.build_dataset_imagenetv2()
            return testset

    def build_dataset_cifar(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        trainset = datasets.CIFAR100(root=self.args['data_path'], train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=self.args['data_path'], train=False, download=True, transform=transform_test)

        return trainset, testset

    def build_dataset_cifar10(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        trainset = datasets.CIFAR10(root=self.args['data_path'], train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=self.args['data_path'], train=False, download=True, transform=transform_test)

        return trainset, testset

    def build_dataset_cars(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),

        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        transform_val = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'split_train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'split_test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        val_dir = os.path.join(self.args['data_path'], 'split_val')
        valset = datasets.ImageFolder(val_dir, transform=transform_val)

        return trainset, testset,valset

    def build_dataset_dtd(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),

        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_sat(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_aircraft(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_flower(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset


    def build_dataset_nwpu(self):
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_pattern(self):
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset


    def build_dataset_imagenet(self):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'val')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_dog(self):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_ucf(self):

        interpolation_mode = transforms.InterpolationMode.BICUBIC
        transform_train = transforms.Compose([
            # 将RandomSizedCrop替换为RandomResizedCrop，并指定插值模式
            transforms.RandomResizedCrop(224, interpolation=interpolation_mode),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        # transform_train = transforms.Compose([
        #     transforms.RandomSizedCrop(224, interpolation=BICUBIC),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_caltech101(self):
        interpolation_mode = transforms.InterpolationMode.BICUBIC

        # 定义训练集的转换操作
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=interpolation_mode),  # 随机裁剪并调整大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # 标准化
        ])

        # 定义测试集的转换操作
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=interpolation_mode),  # 调整大小
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # 标准化
        ])

        # 设置数据路径
        train_dir = os.path.join(self.args['data_path'], 'train')
        test_dir = os.path.join(self.args['data_path'], 'test')

        # 加载训练集和测试集
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        testset = datasets.ImageFolder(test_dir, transform=transform_test)

        return trainset, testset

    def build_dataset_imagenetv2(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # test_dir = os.path.join(self.args['data_path'], 'imagenetv2-matched-frequency-format-val')
        test_dir = os.path.join(self.args['data_path'], 'imagenet-a')
        # test_dir = os.path.join(self.args['data_path'], 'images')
        testset = datasets.ImageFolder(test_dir, transform=transform)

        return  testset
