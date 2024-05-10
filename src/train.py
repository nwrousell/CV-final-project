import torch
from torch.utils.data import DataLoader
from networks import EAST
from my_data import EastDataset
from torch.optim import Adam

from config import print_freq, geometry, training_data_path, testing_data_path, batch_size

def train_EAST(model, train_loader, criterion, optimizer, epoch_num):

    total_loss = 0
    for i, (img, score_label, geo_label, training_mask) in enumerate(train_loader):
        # Move to GPU?

        # Forward pass
        pred_score, pred_geometry = model(img)

        # Compute loss
        loss = criterion(score_label, pred_score, geo_label, pred_geometry, training_mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Logging
        total_loss += loss
        if i % print_freq == 0:
            print("f[Epoch {epoch_num}] Batch {i}/{len(train_loader)} Loss: {total_loss/(i+1)}")



if __name__ == "__main__":
    print("CUDA:", torch.cuda.is_available())
    
    # create model
    east_model = EAST(geometry=geometry)

    # set up data loaders
    train_dataset = EastDataset(training_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = EastDataset(testing_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # set up criterion
    criterion = ...

    # initialize optimizer
    optimizer = Adam()

    for epoch_num in range(1,11):
        train_EAST(east_model, train_dataloader, criterion, optimizer, epoch_num)