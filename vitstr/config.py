import torch
import numpy as np

# ViTSTR (tiny)
patch_length = 16
embedding_dim = 192
num_attention_heads = 3
num_encoder_blocks = 12
max_seq_length = 25
characters = '0123456789abcdefghijklmnopqrstuvwxyz'
loss_type = 'CE'

train_data_path = '/users/nrousell/data/nrousell/cv-project/mjsynth2/mnt/ramdisk/max/90kDICT32px'

print_freq = 100
batch_size = 100
num_epochs = 300
save_interval = 1
pths_path = "pths"
learning_rate = 0.0001

grad_clip = 5.0
