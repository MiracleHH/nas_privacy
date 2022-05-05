from numpy.testing._private.utils import rundocs
import torch
import torchvision
import torchvision.transforms as transforms
from xautodl.datasets import get_datasets
import random
import numpy as np
from attacks.utils.dataloader import UTKFaceDataset, CelebA

dataset_list = ["cifar10", "cifar100", "stl10", "utkface", "celeba", "fmnist"]


class Dataset(object):
    """docstring for Dataset"""

    def __init__(self, dataset, data_path, attr):
        super(Dataset, self).__init__()
        self.full_data, self.num_classes, self.xshape = self.get_full_data(
            dataset, data_path, attr)
        self.full_aug_data, _, _ = self.get_augment_data(dataset,data_path,attr)

    def get_full_data(self, dataset, data_path, attr):
        path=data_path+"/"+dataset
        dataset_name = dataset.lower()
        if dataset_name == "cifar10":
            num_classes = 10
            xshape = (1, 3, 64, 64)
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
            ])
            trainset = torchvision.datasets.CIFAR10(
                root=path, train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(
                root=path, train=False, download=True, transform=transform)
            full_data = trainset + testset
        elif dataset_name == "cifar100":
            num_classes = 100
            xshape = (1, 3, 64, 64)
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
            ])
            trainset = torchvision.datasets.CIFAR100(
                root=path, train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR100(
                root=path, train=False, download=True, transform=transform)
            full_data = trainset + testset
        elif dataset_name=="stl10":
            num_classes = 10
            xshape = (1, 3, 64, 64)
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            trainset = torchvision.datasets.STL10(
                root=path, split='train', download=True, transform=transform)
            testset = torchvision.datasets.STL10(
                root=path, split='test', download=True, transform=transform)
            full_data = trainset + testset
        elif dataset_name=="fmnist":
            num_classes = 10
            xshape = (1, 1, 64, 64)
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
            trainset = torchvision.datasets.FashionMNIST(
                root=path, train=True, download=True, transform=transform)
            testset = torchvision.datasets.FashionMNIST(
                root=path, train=False, download=True, transform=transform)
            full_data = trainset + testset
        elif dataset_name=="utkface":
            if isinstance(attr, list):
                num_classes = []
                for a in attr:
                    if a == "age":
                        num_classes.append(117)
                    elif a == "gender":
                        num_classes.append(2)
                    elif a == "race":
                        num_classes.append(4)
                    else:
                        raise ValueError("Target type \"{}\" is not recognized.".format(a))
            else:
                if attr == "age":
                    num_classes = 117
                elif attr == "gender":
                    num_classes = 2
                elif attr == "race":
                    num_classes = 4
                else:
                    raise ValueError("Target type \"{}\" is not recognized.".format(attr))

            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            full_data = UTKFaceDataset(root=data_path, attr=attr, transform=transform)
            xshape = (1, 3, 64, 64)
        elif dataset_name=="celeba":
            if isinstance(attr, list):
                for a in attr:
                    if a != "attr":
                        raise ValueError("Target type \"{}\" is not recognized.".format(a))

                    num_classes = [8, 4]
                    # heavyMakeup MouthSlightlyOpen Smiling, Male Young
                    attr_list = [[18, 21, 31], [20, 39]]
            else:
                if attr == "attr":
                    num_classes = 8
                    attr_list = [[18, 21, 31]]
                else:
                    raise ValueError("Target type \"{}\" is not recognized.".format(attr))

            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            full_data = CelebA(root=data_path, attr_list=attr_list, target_type=attr, transform=transform)
            xshape = (1, 3, 64, 64)
        else:
            print("Error! Dataset not available! Please select among the following datasets:\n{}".format(
                dataset_list))
            exit(-1)
        return full_data, num_classes, xshape

    def get_augment_data(self, dataset, data_path, attr):
        path=data_path+"/"+dataset
        dataset_name = dataset.lower()
        if dataset_name == "cifar10":
            num_classes = 10
            xshape = (1, 3, 64, 64)
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
            ])
            trainset = torchvision.datasets.CIFAR10(
                root=path, train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(
                root=path, train=False, download=True, transform=transform)
            full_data = trainset + testset
        elif dataset_name == "cifar100":
            num_classes = 100
            xshape = (1, 3, 64, 64)
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
            ])
            trainset = torchvision.datasets.CIFAR100(
                root=path, train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR100(
                root=path, train=False, download=True, transform=transform)
            full_data = trainset + testset
        elif dataset_name=="stl10":
            num_classes = 10
            xshape = (1, 3, 64, 64)
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            trainset = torchvision.datasets.STL10(
                root=path, split='train', download=True, transform=transform)
            testset = torchvision.datasets.STL10(
                root=path, split='test', download=True, transform=transform)
            full_data = trainset + testset
        elif dataset_name=="fmnist":
            num_classes = 10
            xshape = (1, 1, 64, 64)
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
            trainset = torchvision.datasets.FashionMNIST(
                root=path, train=True, download=True, transform=transform)
            testset = torchvision.datasets.FashionMNIST(
                root=path, train=False, download=True, transform=transform)
            full_data = trainset + testset
        elif dataset_name=="utkface":
            if isinstance(attr, list):
                num_classes = []
                for a in attr:
                    if a == "age":
                        num_classes.append(117)
                    elif a == "gender":
                        num_classes.append(2)
                    elif a == "race":
                        num_classes.append(4)
                    else:
                        raise ValueError("Target type \"{}\" is not recognized.".format(a))
            else:
                if attr == "age":
                    num_classes = 117
                elif attr == "gender":
                    num_classes = 2
                elif attr == "race":
                    num_classes = 4
                else:
                    raise ValueError("Target type \"{}\" is not recognized.".format(attr))

            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            full_data = UTKFaceDataset(root=data_path, attr=attr, transform=transform)
            xshape = (1, 3, 64, 64)
        elif dataset_name=="celeba":
            if isinstance(attr, list):
                for a in attr:
                    if a != "attr":
                        raise ValueError("Target type \"{}\" is not recognized.".format(a))

                    num_classes = [8, 4]
                    # heavyMakeup MouthSlightlyOpen Smiling, Male Young
                    attr_list = [[18, 21, 31], [20, 39]]
            else:
                if attr == "attr":
                    num_classes = 8
                    attr_list = [[18, 21, 31]]
                else:
                    raise ValueError("Target type \"{}\" is not recognized.".format(attr))

            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            full_data = CelebA(root=data_path, attr_list=attr_list, target_type=attr, transform=transform)
            xshape = (1, 3, 64, 64)
        else:
            print("Error! Dataset not available! Please select among the following datasets:\n{}".format(
                dataset_list))
            exit(-1)
        return full_data, num_classes, xshape

    def get_split_data(self, seed=-1, augment=False):
        if augment:
            total_data=self.full_aug_data
        else:
            total_data=self.full_data
        len_total = len(total_data)
        len_target_tr_nas_tr=len_total//8
        
        torch.manual_seed(seed)
        target_out, target_nas_tr, target_nas_val, shadow_train, shadow_out = torch.utils.data.random_split(total_data,
                                                                                           [len_target_tr_nas_tr*2, len_target_tr_nas_tr, len_target_tr_nas_tr, len_target_tr_nas_tr*2, len_total-len_target_tr_nas_tr*6])
        
        return target_out, target_nas_tr, target_nas_val, shadow_train, shadow_out

    def get_iid_split_data(self, num_classes, seed=-1, augment=False):
        if augment:
            total_data=self.full_aug_data
        else:
            total_data=self.full_data
        np.random.seed(seed)
        random_seeds=np.random.randint(1,1000000,num_classes)
        #data=self.trainset+self.testset
        class_indices={i:[] for i in range(num_classes)}
        for i,sample in enumerate(total_data):
            class_indices[sample[1]].append(i)
        
        len_class=len(total_data)//num_classes
        len_target_tr_nas_tr=len_class//8
        target_out_idx, target_nas_tr_idx, target_nas_val_idx, shadow_train_idx, shadow_out_idx=[],[],[],[],[]

        for i in range(num_classes):
            random.seed(random_seeds[i])
            random.shuffle(class_indices[i])

            target_out_idx.extend(class_indices[i][:len_target_tr_nas_tr*2])
            target_nas_tr_idx.extend(class_indices[i][len_target_tr_nas_tr*2:len_target_tr_nas_tr*3])
            target_nas_val_idx.extend(class_indices[i][len_target_tr_nas_tr*3:len_target_tr_nas_tr*4])
            shadow_train_idx.extend(class_indices[i][len_target_tr_nas_tr*4:len_target_tr_nas_tr*6])
            shadow_out_idx.extend(class_indices[i][len_target_tr_nas_tr*6:])

        target_out=torch.utils.data.Subset(total_data,target_out_idx)
        target_nas_tr=torch.utils.data.Subset(total_data,target_nas_tr_idx)
        target_nas_val=torch.utils.data.Subset(total_data,target_nas_val_idx)
        shadow_train=torch.utils.data.Subset(total_data,shadow_train_idx)
        shadow_out=torch.utils.data.Subset(total_data,shadow_out_idx)

        return target_out, target_nas_tr, target_nas_val, shadow_train, shadow_out
    
    