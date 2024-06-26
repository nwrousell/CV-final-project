import torch
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
import cv2
import numpy as np

from .deploy import detect, validate_boxes, place_boxes_on_image
from .data_gen import EastDataset, data_transforms
from .networks import EAST
from .icdar import get_images

from .config import geometry, test_data_path

transformer = data_transforms['test']

def predict(model, im, device):
    h, w, _ = im.shape

    # resize to be multiples of 32
    resize_h = h if h % 32 == 0 else (h // 32 - 1) * 32
    resize_w = w if w % 32 == 0 else (w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    # Process image
    im = im[..., ::-1]  # RGB
    im = transforms.ToPILImage()(im)
    processed_im = transformer(im)[np.newaxis, :, :, :]

    processed_im = processed_im.to(device)

    # forward pass
    pred_score, pred_geometry = model(processed_im)

    # restore boxes
    text_boxes = detect(pred_score, pred_geometry)
    
    if text_boxes is None:
        return None

    text_boxes = text_boxes[:, :8].reshape((-1, 4, 2))
    text_boxes[:, :, 0] /= ratio_w
    text_boxes[:, :, 1] /= ratio_h    
    valid_boxes = validate_boxes(text_boxes)

    return valid_boxes


def test_east(model, image_files, device):
    for i, im_path in enumerate(image_files):
        img = cv2.imread(im_path)
        boxes = predict(model, img, device)

        img = img[:,:,::-1]

        img_with_boxes = place_boxes_on_image(img, boxes)

        im_path = f"../results/img_{i}.jpg"
        success = cv2.imwrite(im_path, img_with_boxes)
        if success:
            print(f"wrote image to {im_path}")
        else:
            print("failed to write image")
                

def main():
    print("CUDA:", torch.cuda.is_available())

    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--checkpoint', help='Checkpoint path to load')
    args = parser.parse_args()

    # create model
    east_model = EAST(geometry=geometry)
    east_model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    east_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"loaded model from {args.checkpoint}")

    test_files = get_images(test_data_path)

    test_east(east_model, test_files, device)


if __name__ =="__main__":
    main()