import torch
from torch.utils.data import DataLoader
from networks import EAST
# from my_data import EastDataset
from torch.optim import Adam
from time import time
from torch.optim import lr_scheduler
import os
import argparse
import numpy as np

from config import print_freq, geometry, batch_size, geometry_loss_weight, angle_loss_weight, num_epochs, save_interval, pths_path, learning_rate
from data_gen import EastDataset
from losses import EastLoss


def train_EAST(model, train_loader, criterion, optimizer, epoch_num, device, scheduler):
    total_loss = 0
    for i, (img, score_label, geo_label, training_mask) in enumerate(train_loader):
        # Move tensors to GPU
        img = img.to(device)
        score_label = score_label.to(device)
        geo_label = geo_label.to(device)
        training_mask = training_mask.to(device)

        # Make channel dimension along axis 2 (to match up with the network outputs)
        score_label = torch.permute(score_label, (0, 3, 1, 2))
        geo_label = torch.permute(geo_label, (0, 3, 1, 2))
        training_mask = torch.permute(training_mask, (0, 3, 1, 2))

        # Forward pass
        pred_score, pred_geometry = model(img)
        
        # Compute loss
        loss = criterion(score_label, pred_score, geo_label, pred_geometry, training_mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        # torch.nn.utils.clip_grad_norm_(model.parameters(), )
        optimizer.step()
        scheduler.step()

        # Logging
        total_loss += loss
        if i % print_freq == 0:
            print(f"[Epoch {epoch_num}] Batch {i}/{len(train_loader)} Loss: {total_loss/(i+1)}")



if __name__ == "__main__":
    print("CUDA:", torch.cuda.is_available())
    epoch_num = 0
    
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--checkpoint', help='Checkpoint path to load')
    parser.add_argument('--name', help='name of run')
    args = parser.parse_args()

    print("name:", args.name)

    # create model
    east_model = EAST(geometry=geometry)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize optimizer
    optimizer = Adam(east_model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        east_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_num = checkpoint["epoch"]
        print(f"loaded model from {args.checkpoint}")

    # set up data loaders
    train_dataset = EastDataset('train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # test_dataset = EastDataset('test')
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # set up criterion
    criterion = EastLoss(geometry_loss_weight, angle_loss_weight)

    data_parallel = False
    if torch.cuda.device_count() > 1:
        east_model = torch.nn.DataParallel(east_model)
        data_parallel = True
        print("Using multiple devices")
    east_model.to(device)

    start_time = time()

    for epoch_num in range(epoch_num, num_epochs):
        train_EAST(east_model, train_dataloader, criterion, optimizer, epoch_num, device, scheduler)
        minutes_elapsed = (time() - start_time) // 60
        print(f"finished epoch {epoch_num}. time elapsed: {minutes_elapsed}m")

        if (epoch_num+1) % save_interval == 0:
            state_dict = east_model.module.state_dict() if data_parallel else east_model.state_dict()
            save_path = f"{args.name}_model_epoch_{epoch_num+1}.pth"
            torch.save({
                "epoch": epoch_num,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict()
            }, os.path.join(pths_path, save_path))
            print(f"saved checkpoint at {save_path}")

# Non-maxima suppression
# eval