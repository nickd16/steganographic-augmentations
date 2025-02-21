# steganographic-augmentations


![stegimage](https://github.com/user-attachments/assets/a0a8800e-2476-4baf-9ed7-4b7515732074)


All results can be reproduced in train.py:
```bash
pip3 install -r requirements.txt
python3 train.py

A naive implementation of the steganographic embeddings can be found in steganography.py

```python
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


![colors](https://github.com/user-attachments/assets/d5d730d7-a3a3-4682-898e-de1406904cb7)


![distributions](https://github.com/user-attachments/assets/35ef0b51-8fff-4684-bf00-28ac1c75ad6d)


![finaldata](https://github.com/user-attachments/assets/e7db6f4a-fadb-499b-8bae-d2ce1e79e993)
