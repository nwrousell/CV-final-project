import torch
import cv2
import numpy as np
import argparse
import os
from threading import Thread, Lock
import time

from east.networks import EAST
from vitstr.network import ViTSTR
from vitstr.converter import TokenLabelConverter
from east.test import predict
from east.deploy import place_boxes_on_image
from vitstr.test import infer
from video import expand_box, get_image_from_bounding_box, render_with_text
from affine_matrix import find_keypoints_and_descriptors, match_descriptors, compute_affine_matrix
from translator import Translator

east_checkpoint_path = "east/pths/resume_w_grad_clip_model_epoch_600.pth"
vitstr_checkpoint_path = "vitstr/pths/continue_model_epoch_2.pth"

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


def pipeline(img, east, vitstr, converter, device, translator):

    image_with_translation = img.copy()
    # run east to get boxes
    boxes = predict(east, img, device)
    if boxes is None: # no text detected
        return None, None, None, None, None
        
    # boxes = [expand_box(box, 0.05) for box in boxes]
    # boxes_on_image = place_boxes_on_image(img, boxes)

    # cut out boxes
    # img_rgb = img[:,:,::-1] # BGR -> RGB
    sub_images = [np.array(get_image_from_bounding_box(img, (box[0][0], box[0][1]), (box[3][0], box[3][1]), (box[2][0], box[2][1]), (box[1][0], box[1][1]))) for box in boxes]

    # run vitstr to get text
    text_preds = text_recognition(sub_images, vitstr, converter, device)

    # call to translate API
    # corrected_preds = [translator.get_spellchecked_word(word) for word in text_preds]
    # print(corrected_preds)
    translations = translator.translate_text(text_preds)

    # use ORB to generate feature descriptors
    keypoints, descriptors = find_keypoints_and_descriptors(img)

    return boxes, text_preds, translations, keypoints, descriptors

def process_file(im_path, east, vitstr, converter, device, translator):
    img = cv2.imread(im_path)
    print("read img", img.shape)

    boxes, text_preds, translations, _, _ = pipeline(img, east, vitstr, converter, device, translator)

    save_path = rf"results/{im_path.split('/')[-1]}"

    # cv2.imwrite(rf"results/bounding_{im_path.split('/')[-1]}", img_with_boxes)

    rendered_img = render_with_text(img, boxes, translations)

    success = cv2.imwrite(save_path, rendered_img)
    if success:
        print(f"processed file {im_path} and wrote result to {save_path}\n\tDetected {len(text_preds)} word(s): {', '.join(text_preds)}")
    else:
        print(f"failed to write {save_path}")


def live_demo(east, vitstr, converter, device, translator):
    vid = cv2.VideoCapture(0) 
  
    FPS = 24
  
    shared_data = {
        "boxes": None,
        "text_preds": None,
        "translations": None,
        "keypoints": None,
        "feature_descriptors": None,
        "current_frame": None,
        "mtx": Lock()
    }
  
    def slow_update_thread_func():
        while True:
            with shared_data["mtx"]:
                current_frame = shared_data["current_frame"]
            
            if current_frame is None:
                continue
            
            print("[slow] computing slow forward pass")
            
            b, pred, trans, kps, desc = pipeline(current_frame, east, vitstr, converter, device, translator)
            
            with shared_data['mtx']:
                shared_data["boxes"] = b
                shared_data["text_preds"] = pred
                shared_data["translations"] = trans
                shared_data["keypoints"] = kps
                shared_data["feature_descriptors"] = desc
            
            # force thread to yield
            # time.sleep(5)

    def fast_update_thread_func():
        transformed_boxes = None
        
        while True: 
            # Read a video frame
            ret, current_frame = vid.read() 
            
            print("[fast] read current frame")
            
            with shared_data["mtx"]:
                shared_data["current_frame"] = current_frame
                boxes = shared_data["boxes"]
                translations = shared_data["translations"]
                keypoints = shared_data["keypoints"]
                feature_descriptors = shared_data["feature_descriptors"]
                
            # Fast update
            if feature_descriptors is not None:
                current_keypoints, current_descriptors = find_keypoints_and_descriptors(current_frame)
                matches = match_descriptors(feature_descriptors, current_descriptors)
            
                if len(matches) >= 3 and boxes is not None:
                    M = compute_affine_matrix(keypoints, current_keypoints, matches)
                    points = np.array(boxes).reshape(-1, 2).T
                    transformed_points = (np.matmul(M[:2,:2], points) + M[:, 2, np.newaxis]).T
                    transformed_boxes = [box for box in transformed_points.reshape(len(boxes), 4, 2)]
                    print("[fast] computed affine matrix")
             
            
        
        
            if boxes is not None:
                if transformed_boxes is None:
                    transformed_boxes = boxes.copy()

                try:
                    rendered_img = render_with_text(current_frame, transformed_boxes, translations)
                    # Display the resulting frame 
                    cv2.imshow('frame', rendered_img) 
                    print("[fast] rendered frame with text!")
                except:
                    cv2.imshow('frame', current_frame)
                    print("[fast] rendered frame with no text")
            else:
                cv2.imshow('frame', current_frame)
                print("[fast] rendered frame with no text")
            
            # quit when q is pressed
            if cv2.waitKey(1) == ord('q'): 
                break
                
            time.sleep(1 / FPS)
            
        # After the loop release the cap object 
        vid.release() 
        # Destroy all the windows 
        cv2.destroyAllWindows() 
    
    # Create two threads
    thread1 = Thread(target=slow_update_thread_func)
    thread2 = Thread(target=fast_update_thread_func)

    # Start the threads
    thread2.start() # start this one first cause it needs to grab the first image
    thread1.start()

    # Wait for the threads to finish (which they won't in this case)
    thread1.join()
    thread2.join()
    
    
# cd OneDrive/Documents/Coding/cs1430/CV-final-project
# conda activate cv-project
# python pipeline.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path',
                        help='path to file(s) to run the pipeline on. Results are written to the results directory. If passed a folder --> will process each.')

    args = parser.parse_args()

    source_lang = "en"
    target_lang = "es"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    east, vitstr = load_models(device)
    
    converter = TokenLabelConverter()
    translator = Translator(source_lang="en", target_lang=target_lang)

    if args.path is not None:
        # parse path flag to get file(s)
        if "." in args.path: # file, not dir
            im_paths = [args.path]
        else:
            im_paths = [os.path.join(args.path, p) for p in os.listdir(args.path)]
        print(f"processing {im_paths}")
        for p in im_paths:
            process_file(p, east, vitstr, converter, device, translator)
    else:
        live_demo(east, vitstr, converter, device, translator)