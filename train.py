import torch
import torch.nn as nn
from data import *
import pytorch_lightning as pl
from model import *
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import random
import numpy as np

class MetricsPrinter(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"\nEpoch {trainer.current_epoch}:")
        print(f"Train Acc: {trainer.logged_metrics['train_acc']} | Train Loss: {trainer.logged_metrics['train_loss']:.4f}")
        
    def on_validation_epoch_end(self, trainer, pl_module):
        print(f"\nVal Acc: {trainer.logged_metrics['val_acc']:.4f} | Val Loss: {trainer.logged_metrics['val_loss']:.4f}")
        print("-" * 50)

def train(
    name: str,
    num_classes: int,
    model = None,
    dataset = str,
    transform = None,
    num_epochs: int=10,
    seed: int=random.randint(0, 2**32 - 1),
    num_workers: int=5,
    download: bool=False,
    batch_size: int=64,   
    lr: float=2e-4,
    num_bits: int=4,
    steg: bool=False,
    embed_prob: float=0.5
):

    trainloader, valloader, testloader = process_data(
        batch_size=batch_size, 
        seed=seed, 
        download=download,
        num_workers=num_workers,
        steg=steg,
        num_bits=num_bits,
        embed_prob=embed_prob,
        dataset=dataset,
        num_classes=num_classes,
        transform=transform
    )
    
    model = torch.compile(model)

    model = plmodel(model=model, num_classes=num_classes, lr=lr, num_epochs=num_epochs)

    logger = TensorBoardLogger(
        save_dir="tensorboard_logs",
        name=name,
        default_hp_metric=False 
    )
    
    metrics_printer = MetricsPrinter()

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='cuda', 
        devices=1,
        log_every_n_steps=50,
        logger=logger,
        strategy='auto',
        inference_mode='True',
        callbacks=[metrics_printer],
        enable_progress_bar=True,
    )

    trainer.fit(model, trainloader, valloader)
    trainer.test(model, testloader)

def main():

    # transforms.ToTensor(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    # transforms.ToTensor(),
    # transforms.GaussianBlur(kernel_size=3)
    # transforms.RandomHorizontalFlip(p=0.5)
    # transforms.RandomErasing(p=0.5) 

    lr = 2e-4
    num_classes=10
    dataset = 'cifar10'
    num_epochs = 50
    training_runs = 100
    names = [
        'cifar10_steg',
    ]
    transformz = [
        transforms.Compose([
            transforms.ToTensor(),
        ]),  
    ]

    for _ in range(training_runs):
        for name,trans in zip(names, transformz):
            train(
                lr=lr, 
                model=make_net(output_dim=num_classes), 
                num_classes=num_classes, 
                num_epochs=num_epochs, 
                name=name, 
                steg=True,
                transform=trans,
                dataset=dataset,
            )

if __name__ == '__main__':
    main()