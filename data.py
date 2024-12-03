import torch 
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Dataset
import random
import time
from steganography import *
from datasets import load_dataset

class CIFAR10WithOneHot(Dataset):
    def __init__(self, dataset, embed_prob, num_bits, steg):
        self.dataset = dataset
        self.embed_prob = embed_prob
        self.num_bits = num_bits
        self.steg = steg

    def __getitem__(self, index):
        data, label = self.dataset[index]
        
        if self.steg and random.random() < self.embed_prob:
            secret_idx = random.randint(0, len(self.dataset) - 1)
            secret_data, secret_label = self.dataset[secret_idx]
            embedded_data = lsb_embed(data, secret_data, self.num_bits)
            one_hot = torch.zeros(10)
            one_hot[label] = 1
            one_hot[secret_label] = 1
            return embedded_data, one_hot
        
        one_hot = torch.zeros(10)
        one_hot[label] = 1
        return data, one_hot
        
    def __len__(self):
        return len(self.dataset)

def process_data_cifar10(
        batch_size, 
        seed, 
        num_workers, 
        download, 
        embed_prob, 
        steg,
        num_bits,
    ):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5)
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=download,
        transform=transform
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=download,
        transform=transform
    )
    
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    trainset, valset = random_split(trainset, [train_size, val_size], generator)
    
    trainset = CIFAR10WithOneHot(trainset, embed_prob=embed_prob, steg=steg, num_bits=num_bits)
    valset = CIFAR10WithOneHot(valset, embed_prob=0, steg=False, num_bits=0)
    testset = CIFAR10WithOneHot(testset, embed_prob=0, steg=False, num_bits=0)
    
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return trainloader, valloader, testloader

class TinyImagenetWithOneHot(Dataset):
    def __init__(self, dataset, embed_prob, num_bits, steg):
        self.dataset = dataset
        self.embed_prob = embed_prob
        self.num_bits = num_bits
        self.steg = steg

    def __getitem__(self, index):
        data, label = self.dataset[index]
        
        if self.steg and random.random() < self.embed_prob:
            secret_idx = random.randint(0, len(self.dataset) - 1)
            secret_data, secret_label = self.dataset[secret_idx]
            embedded_data = lsb_embed(data, secret_data, self.num_bits)
            one_hot = torch.zeros(200)
            one_hot[label] = 1
            one_hot[secret_label] = 1
            return embedded_data, one_hot
        
        one_hot = torch.zeros(200)
        one_hot[label] = 1
        return data, one_hot
        
    def __len__(self):
        return len(self.dataset)

class TinyImagenetBase(Dataset):
    def __init__(self, split, transform):
        if split == 'train':
            ds = load_dataset("zh-plus/tiny-imagenet", split='train')
        else:
            ds = load_dataset("zh-plus/tiny-imagenet", split='valid')
        
        self.transform = transform
        self.X = []
        self.Y = []

        for idx in range(len(ds)):
            label = ds[idx]["label"]
            if label == -1:  
                continue

            img = ds[idx]["image"]  
            if img.mode == 'L':  
                img = img.convert('RGB') 
            elif img.mode != 'RGB':
                print(f"Unexpected image mode: {img.mode}. Converting to RGB.")
                img = img.convert('RGB')  

            self.X.append(self.transform(img))
            self.Y.append(label) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def process_data_tinyimagenet(
        batch_size, 
        seed, 
        num_workers, 
        embed_prob, 
        steg,
        num_bits,
        download,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    
    trainset = TinyImagenetBase(split='train', transform = transform)
    testset = TinyImagenetBase(split='valid', transform = transform)      
    
    trainset = TinyImagenetWithOneHot(trainset, embed_prob=embed_prob, steg=steg, num_bits=num_bits)
    testset = TinyImagenetWithOneHot(testset, embed_prob=0, steg=False, num_bits=0)
    
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return trainloader, testloader, testloader

class STL10WithOneHot(Dataset):
    def __init__(self, dataset, embed_prob, num_bits, steg):
        self.dataset = dataset
        self.embed_prob = embed_prob
        self.num_bits = num_bits
        self.steg = steg

    def __getitem__(self, index):
        data, label = self.dataset[index]
        
        if self.steg and random.random() < self.embed_prob:
            secret_idx = random.randint(0, len(self.dataset) - 1)
            secret_data, secret_label = self.dataset[secret_idx]
            embedded_data = lsb_embed(data, secret_data, self.num_bits)
            one_hot = torch.zeros(10)
            one_hot[label] = 1
            one_hot[secret_label] = 1
            return embedded_data, one_hot
        
        one_hot = torch.zeros(10)
        one_hot[label] = 1
        return data, one_hot
        
    def __len__(self):
        return len(self.dataset)

def process_data_stl10(
        batch_size, 
        seed, 
        num_workers, 
        download, 
        embed_prob, 
        steg,
        num_bits,
    ):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5)
    ])
    
    trainset = torchvision.datasets.STL10(
        root='./data', 
        split='train',
        download=download,
        transform=transform
    )
    
    testset = torchvision.datasets.STL10(
        root='./data',
        split='test',
        download=download,
        transform=transform
    )
    
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    trainset, valset = random_split(trainset, [train_size, val_size], generator)
    
    trainset = STL10WithOneHot(trainset, embed_prob=embed_prob, steg=steg, num_bits=num_bits)
    valset = STL10WithOneHot(valset, embed_prob=0, steg=False, num_bits=0)
    testset = STL10WithOneHot(testset, embed_prob=0, steg=False, num_bits=0)
    
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return trainloader, valloader, testloader

def process_data(
        batch_size, 
        seed, 
        num_workers, 
        embed_prob, 
        steg,
        num_bits,
        download,
        dataset,
    ):

    dataset_processors = {
        'cifar10': process_data_cifar10,
        'tinyimagenet': process_data_tinyimagenet,
        'stl10': process_data_stl10
    }

    if dataset not in dataset_processors:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return dataset_processors[dataset](
        batch_size=batch_size, 
        seed=seed, 
        download=download,
        num_workers=num_workers,
        steg=steg,
        num_bits=num_bits,
        embed_prob=embed_prob
    )

def main():
    trainloader, valloader,testloader = process_data_tinyimagenet(
        batch_size=64, 
        seed=42, 
        num_workers=0,
        steg=True,
        num_bits=4,
        embed_prob=0.75,
        download=False,
    )

    # for i, (x,y) in enumerate(trainloader):
    #     print(x.shape)
    #     print(y)

    # for i, (x,y) in enumerate(valloader):
    #     print(x.shape)
    #     print(y.shape)

    # for i, (x,y) in enumerate(testloader):
    #     print(x.shape)
    #     print(y.shape)

if __name__ == '__main__':
    main()

