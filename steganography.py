import torch 
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Dataset
import cv2
import numpy as np
import random
import torch.fft
import time
import sys
import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor
from typing import Tuple

def lsb_embed(cover, secret, num_bits):
    cover = (cover * 255).byte()
    secret = (secret * 255).byte()
    mask = 0xFF << num_bits
    cover_cleared = cover & mask
    secret_shifted = secret >> (8 - num_bits)
    embedded = cover_cleared | secret_shifted
    return embedded.float() / 255.0

class LSBEmbed:
    def __init__(self, num_bits):
        self.num_bits = num_bits
    
    def __call__(self, data):
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError("Input must be a tuple of (cover_image, secret_image)")
        cover, secret = data
        return lsb_embed(cover, secret, self.num_bits)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(num_bits={self.num_bits})"

def palette_embed(cover, secret, num_bits):
    palette_size = 256
    palette = torch.randint(0, 256, (palette_size, 3), dtype=torch.uint8)
    cover = (cover * 255).byte()
    secret = (secret * 255).byte()
    mask = 0xFF << num_bits
    secret_bits = (secret >> (8 - num_bits)).byte()
    palette_cleared = palette & mask
    secret_extended = secret_bits.view(-1, 1)
    palette_embedded = palette_cleared + secret_extended[:len(palette)]
    cover_flat = cover.view(-1, 3)
    diff = (cover_flat.unsqueeze(1) - palette_embedded.unsqueeze(0)).abs().sum(dim=2)
    _, indices = diff.min(dim=1)
    embedded = palette_embedded[indices]
    embedded = embedded.view(cover.shape)
    return embedded.float() / 255.0


def pvd_embed(cover, secret, num_bits):
    threshold = 16
    cover = (cover * 255).byte()
    secret = (secret * 255).byte()
    diff = torch.abs(cover[:, :, 1:] - cover[:, :, :-1])
    mask = diff < threshold
    secret_mod = secret[:, :, 1:] % threshold
    embedded = cover.clone()
    embedded[:, :, 1:][mask] += secret_mod[mask]
    return embedded.float() / 255.0

def dct_embed(cover, secret, num_bits):
    cover = cover * 255.0
    secret = secret * 255.0
    cover_dct = torch.fft.rfft2(cover, norm='ortho')
    secret_dct = torch.fft.rfft2(secret, norm='ortho')
    max_coeffs = 64
    coeff_scale = 0.1
    mask = torch.ones_like(cover_dct, dtype=torch.bool)
    mask[..., max_coeffs:] = False
    cover_dct[mask] = cover_dct[mask]
    cover_dct[~mask] = cover_dct[~mask] + secret_dct[~mask] * coeff_scale
    embedded = torch.fft.irfft2(cover_dct, s=cover.shape[-2:], norm='ortho')
    return torch.clamp(embedded / 255.0, 0.0, 1.0)

def save_lsb_steganography_plot(cover_path, secret_path, save_path, num_bits=6):
    cover_img = np.array(Image.open(cover_path).convert('RGB').resize((224, 224)))
    secret_img = np.array(Image.open(secret_path).convert('RGB').resize((224, 224)))
    
    mask = 0xFF << num_bits
    cover_cleared = cover_img & mask
    secret_shifted = secret_img >> (8 - num_bits)
    embedded = cover_cleared | secret_shifted
    
    secret_msb = secret_img & (0xFF << (8 - num_bits))
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(cover_img)
    axs[0].set_title("Cover Image (256x256)")
    axs[0].axis('off')

    axs[1].imshow(secret_img)
    axs[1].set_title("Secret Image (256x256)")
    axs[1].axis('off')

    axs[2].imshow(secret_msb)
    axs[2].set_title("Secret (MSBs Only)")
    axs[2].axis('off')

    axs[3].imshow(embedded)
    axs[3].set_title("Embedded Image (224x224)")
    axs[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    save_lsb_steganography_plot("img1.jpg", "img2.jpg", "img3.jpg")

if __name__ == '__main__':
    main()