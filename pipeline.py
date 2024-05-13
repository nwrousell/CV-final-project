import torch

from east.networks import EAST
from vitstr.network import ViTSTR

from east.deploy import detect, place_boxes_on_image, validate_boxes
from vitstr.converter import TokenLabelConverter

east_checkpoint_path = "east/pths/resume_w_grad_clip_model_epoch_600.pth"
vitstr_checkpoint_path = "vitstr/pths/..."

def load_models(device):
    east_model = EAST()
    east_checkpoint = torch.load(east_checkpoint_path, map_location=device)
    east_model.load_state_dict(east_checkpoint["model_state_dict"])

    vitstr_model = ViTSTR()
    vitstr_checkpoint = torch.load(vitstr_checkpoint_path, map_location=device)
    vitstr_model.load_state_dict(vitstr_checkpoint["model_state_dict"])

    east_model.to(device)
    vitstr_model.to(device)

    return east_model, vistr_model

def text_detection(image, east):
    pass


# test east with documents
# test ViTSTR with spanish/french/etc. 

# hook up full pipeline with webcam