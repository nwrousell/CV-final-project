import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from time import time
from torch.optim import lr_scheduler
import os
import argparse
import numpy as np
import time

from .network import ViTSTR
from .data import RawDataset, filter_collate_fn
from .converter import TokenLabelConverter

from .config import print_freq, batch_size, num_epochs, save_interval, pths_path, learning_rate, loss_type, grad_clip, train_data_path, characters

def infer(image, vitstr_model, converter, device):
    image = torch.permute(image, (0,3,1,2)).to(device)

    # Forward pass
    pred = vitstr_model(image, converter.max_seq_length, device)[:,1:,:]

    pred_indices = torch.argmax(pred, dim=-1)

    lengths = [pred_indices.shape[1]] * pred_indices.shape[0]
    pred_texts = converter.decode(pred_indices, lengths)

    print(pred_texts)

    return pred_texts

def main():
    print("CUDA:", torch.cuda.is_available())
    epoch_num = 0
    
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--checkpoint', help='Checkpoint path to load')
    parser.add_argument('--name', help='name of run')
    args = parser.parse_args()

    print("name:", args.name)
    print("# characters:", len(characters))

    # create model
    vitstr_model = ViTSTR()

    pytorch_total_params = sum(p.numel() for p in vitstr_model.parameters() if p.requires_grad)
    print("# trainable parameters:", pytorch_total_params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize optimizer
    optimizer = Adam(vitstr_model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        vitstr_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_num = checkpoint["epoch"]
        print(f"loaded model from {args.checkpoint}")

    # set up data loaders
    train_dataset = RawDataset(train_data_path)
    print("num train samples:", len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=filter_collate_fn)

    converter = TokenLabelConverter()

    data_parallel = False
    if torch.cuda.device_count() > 1:
        vitstr_model = torch.nn.DataParallel(vitstr_model)
        data_parallel = True
        print("Using multiple devices")
    vitstr_model.to(device)

    for i, (image, label) in enumerate(train_dataloader):
        pred_text = infer(image, device)[0]
        print(f"Sample {i}: {pred_text}/{label[0]}")
        time.sleep(1)


if __name__ == "__main__":
    main()