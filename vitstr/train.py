import torch
from torch.utils.data import DataLoader
from network import ViTSTR
from torch.optim import Adam
from time import time
from torch.optim import lr_scheduler
import os
import argparse
import numpy as np

from .data import RawDataset, filter_collate_fn
from .converter import TokenLabelConverter

from .config import print_freq, batch_size, num_epochs, save_interval, pths_path, learning_rate, loss_type, grad_clip, train_data_path, characters


def train_ViTSTR(model, train_loader, criterion, optimizer, epoch_num, device, scheduler, converter):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to GPU
        images = torch.permute(images, (0,3,1,2)).to(device)

        gt_labels, target_lengths = converter.encode(labels)
        gt_labels = gt_labels.to(device)
        target_lengths = torch.tensor(target_lengths).to(device)

        # Forward pass
        pred_text = model(images, converter.max_seq_length, device)
        
        # Compute loss
        if loss_type == 'CTC':
            pred_text = nn.log_softmax(pred_text).permute((1, 0, 2)) # (input length, batch size, num classes)
            loss = criterion(pred_text, gt_labels, converter.max_seq_length, target_lengths)
        else: 
            loss = criterion(pred_text.view(-1, pred_text.shape[-1]), gt_labels.contiguous().view(-1))


        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        # Logging
        total_loss += loss
        if i % print_freq == 0:
            print(f"[Epoch {epoch_num}] Batch {i}/{len(train_loader)} Loss: {total_loss/(i+1)}")

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vitstr_model = ViTSTR()

    data_parallel = False
    if torch.cuda.device_count() > 1:
        vitstr_model = torch.nn.DataParallel(vitstr_model)
        data_parallel = True
        print("Using multiple devices")
    
    vitstr_model.to(device)

    pytorch_total_params = sum(p.numel() for p in vitstr_model.parameters() if p.requires_grad)
    print("# trainable parameters:", pytorch_total_params)


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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=filter_collate_fn)

    converter = TokenLabelConverter()

    # set up loss
    if loss_type == 'CTC':
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

    start_time = time()

    for epoch_num in range(epoch_num, num_epochs):
        train_ViTSTR(vitstr_model, train_dataloader, criterion, optimizer, epoch_num, device, scheduler, converter)
        minutes_elapsed = (time() - start_time) // 60
        print(f"finished epoch {epoch_num}. time elapsed: {minutes_elapsed}m")

        if (epoch_num+1) % save_interval == 0:
            state_dict = vitstr_model.module.state_dict() if data_parallel else vitstr_model.state_dict()
            save_path = f"{args.name}_model_epoch_{epoch_num+1}.pth"
            torch.save({
                "epoch": epoch_num,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict()
            }, os.path.join(pths_path, save_path))
            print(f"saved checkpoint at {save_path}")

if __name__ == "__main__":
    main()
