from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch.nn.functional as F
import torchvision
import torch

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

def conv(ch_in, ch_out):
    return nn.Conv2d(ch_in, ch_out, kernel_size=3, 
                     padding='same', bias=False)

def make_net(output_dim=10):
    act = lambda: nn.GELU()
    bn = lambda ch: nn.BatchNorm2d(ch)
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True),
            act(),
        ),
        nn.Sequential(
            conv(24, 64),
            nn.MaxPool2d(2),
            bn(64), act(),
            conv(64, 64),
            bn(64), act(),
        ),
        nn.Sequential(
            conv(64, 256),
            nn.MaxPool2d(2),
            bn(256), act(),
            conv(256, 256),
            bn(256), act(),
        ),
        nn.Sequential(
            conv(256, 256),
            nn.MaxPool2d(2),
            bn(256), act(),
            conv(256, 256),
            bn(256), act(),
        ),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(256, output_dim, bias=False),
        Mul(1/9)
    )

class RESNET18(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class plmodel(pl.LightningModule):
    def __init__(self, model, num_classes, lr, num_epochs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        pred_class = torch.argmax(logits, dim=1)
        y_class = torch.argmax(y, dim=1)
        self.train_acc(pred_class, y_class)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        loss = F.binary_cross_entropy_with_logits(logits, y)
    
        pred_class = torch.argmax(logits, dim=1)
        y_class = torch.argmax(y, dim=1)
        self.val_acc(pred_class, y_class)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        pred_class = torch.argmax(logits, dim=1)
        y_class = torch.argmax(y, dim=1)
        self.test_acc(pred_class, y_class)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.90, 0.95), weight_decay=0.01)
        return optimizer

def main():
    def print_shape_hook(module, input, output):
        print(f"Shape after {module}: {output.shape}")


    x = torch.randn((64, 3, 64,64)).cuda()
    model = make_net(output_dim=200).cuda()
    for layer in model:
        if isinstance(layer, nn.AdaptiveAvgPool2d) or isinstance(layer, nn.MaxPool2d):
            layer.register_forward_hook(print_shape_hook)
    print(model(x).shape)

if __name__ == '__main__':
    main()

