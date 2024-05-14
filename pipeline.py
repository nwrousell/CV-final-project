import torch
import cv2
import numpy as np
import argparse
import os

from east.networks import EAST
from vitstr.network import ViTSTR
from vitstr.converter import TokenLabelConverter
from east.test import predict
from east.deploy import place_boxes_on_image
from vitstr.test import infer
from video import expand_box, get_image_from_bounding_box, render_text_box
from affine_matrix import find_keypoints_and_descriptors
from translator import Translator

east_checkpoint_path = "east/pths/resume_w_grad_clip_model_epoch_600.pth"
vitstr_checkpoint_path = "vitstr/pths/continue_model_epoch_18.pth"

# pip install numba natsort opencv-python shapely argparse transformers lanms-neo

def load_models(device):
    east_model = EAST(geometry="RBOX")
    east_checkpoint = torch.load(east_checkpoint_path, map_location=device)
    east_model.load_state_dict(east_checkpoint["model_state_dict"])

    vitstr_model = ViTSTR()
    vitstr_checkpoint = torch.load(vitstr_checkpoint_path, map_location=device)
    vitstr_model.load_state_dict(vitstr_checkpoint["model_state_dict"])

    east_model.to(device)
    vitstr_model.to(device)

    return east_model, vitstr_model


def text_recognition(sub_images, vitstr, converter, device):
    # resize each text-image to 224x224
    resized = [cv2.resize(sub_im, (224, 224), interpolation=cv2.INTER_CUBIC) for sub_im in sub_images]

    # stack images into "batch" and convert to tensor
    stack = torch.tensor(np.stack(resized, axis=0))

    # feed-forward / decode
    pred_text = infer(stack, vitstr, converter, device)

    return pred_text

def render_boxes_on_image(base_img, boxes, translations):
    image_with_translation = base_img.copy()
    for box, text in zip(boxes, translations):
        image_with_translation = render_text_box(image_with_translation, (box[0][0], box[0][1]), (box[3][0], box[3][1]), (box[2][0], box[2][1]), (box[1][0], box[1][1]), text)
    
    return image_with_translation

def pipeline(img, east, vitstr, converter, device, target_lang, translator):

    image_with_translation = img.copy()
    # run east to get boxes
    boxes = predict(east, img, device)
    if boxes is None: # no text detected
        return None, None, None, None, None
        
    boxes = [expand_box(box, 0.15) for box in boxes]
    # boxes_on_image = place_boxes_on_image(img, boxes)

    # cut out boxes
    # img_rgb = img[:,:,::-1] # BGR -> RGB
    sub_images = [np.array(get_image_from_bounding_box(img, (box[0][0], box[0][1]), (box[3][0], box[3][1]), (box[2][0], box[2][1]), (box[1][0], box[1][1]))) for box in boxes]

    # run vitstr to get text
    text_preds = text_recognition(sub_images, vitstr, converter, device)

    # call to translate API
    translations = translator.translate_text(text_preds, target_lang)
    # translations = [t.text for t in translator.translate(text_preds, dest=target_lang)]

    # use ORB to generate feature descriptors
    keypoints, descriptors = find_keypoints_and_descriptors(img)

    return boxes, text_preds, translations, keypoints, descriptors

def process_file(im_path, east, vitstr, converter, device, target_lang, translator):
    img = cv2.imread(im_path)
    print("read img", img.shape)

    boxes, text_preds, translations, _, _ = pipeline(img, east, vitstr, converter, device, target_lang, translator)

    save_path = rf"results/{im_path.split('/')[-1]}"

    # cv2.imwrite(rf"results/bounding_{im_path.split('/')[-1]}", img_with_boxes)

    rendered_img = render_boxes_on_image(img, boxes, translations)

    success = cv2.imwrite(save_path, rendered_img)
    if success:
        print(f"processed file {im_path} and wrote result to {save_path}\n\tDetected {len(text_preds)} word(s): {', '.join(text_preds)}")
    else:
        print(f"failed to write {save_path}")


def live_demo(east, vitstr, converter, device, target_lang, translator):
    vid = cv2.VideoCapture(0) 
  
    # Slow update data
    boxes = None
    text_preds = None
    translations = None
    keypoints = None
    feature_descriptors = None
  
    while True: 
        
        # Read a video frame
        ret, frame = vid.read() 
        
        boxes, text_preds, translations, keypoints, descriptors = pipeline(frame, east, vitstr, converter, device, target_lang, translator)
        if boxes is not None:
            rendered_img = render_boxes_on_image(frame, boxes, translations)
            
            # Display the resulting frame 
            cv2.imshow('frame', rendered_img) 
        else:
            cv2.imshow('frame', frame)
        
        # quit when q is pressed
        if cv2.waitKey(1) == ord('q'): 
            break
        
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path',
                        help='path to file(s) to run the pipeline on. Results are written to the results directory. If passed a folder --> will process each.')

    args = parser.parse_args()

    target_lang = "es"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    east, vitstr = load_models(device)
    
    converter = TokenLabelConverter()
    translator = Translator()

    if args.path is not None:
        # parse path flag to get file(s)
        if "." in args.path: # file, not dir
            im_paths = [args.path]
        else:
            im_paths = [os.path.join(args.path, p) for p in os.listdir(args.path)]
        print(f"processing {im_paths}")
        for p in im_paths:
            process_file(p, east, vitstr, converter, device, target_lang, translator)
    else:
        live_demo(east, vitstr, converter, device, target_lang, translator)