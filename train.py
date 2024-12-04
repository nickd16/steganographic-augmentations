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
        num_classes=num_classes
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
    lr = 2e-4
    num_classes=10
    dataset = 'stl10'
    #model = make_net(output_dim=num_classes)
    #model = RESNET18(output_dim=num_classes)
    model = ViT(output_dim=num_classes, img_size=96)
    num_epochs = 50
    for _ in range(4):
        train(
            lr=lr, 
            model=model, 
            num_classes=num_classes, 
            num_epochs=num_epochs, 
            name='stl10_d_vit', 
            steg=False,
            dataset=dataset,
        )
    for _ in range(5):
        train(
            lr=lr, 
            model=model, 
            num_classes=num_classes, 
            num_epochs=num_epochs, 
            name='stl_s_vit', 
            steg=True,
            dataset=dataset,
        )
    


if __name__ == '__main__':
    main()