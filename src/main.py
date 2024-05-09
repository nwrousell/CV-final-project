import torch
from networks import EAST

# download dataset

# preprocessing images

# skim Mask R-CNNs, EAST, and other two papers

# document vs. natural scene?

# implement architecture (choose fast one)

# train on Oscar

# once good enough, create GUI, add google translate call, re-render translation onto image

# then, if time, use fast correspondence matching to enable psuedo-live detection

def train_EAST():
    east_model = EAST(geometry="quad")

    # random input
    rand_input = torch.rand(1,3,512,512)
    print("input shape:", rand_input.shape)

    # forward pass
    output = east_model.forward(rand_input)
    print("feature shape:", output.shape)
    



if __name__ == "__main__":
    print("CUDA:", torch.cuda.is_available())
    train_EAST()