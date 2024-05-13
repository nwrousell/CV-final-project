import torch
import numpy as np
import cv2
from lanms import merge_quadrangle_n9 as la_nms

from .icdar import restore_rectangle

'''
    Given model outputs, apply Thresholding and Non-Maximum Suppression to 
    get the detected bounding boxes

    Args:
        score_map: tensor of shape (1, 1, 128, 128), converted to (128, 128)
        geometry_map: tensor of shape (batch_size, 5, 128, 128), converted to (5, 128, 128)
        score_map_thresh (optional): scalar, threshold for score map
        nms_thresh (optional): scalar, threshold for NMS
'''
def detect(score_map, geometry_map, score_map_thresh=0.8, nms_thresh=0.2):

    score_map = torch.squeeze(score_map)
    geometry_map = torch.squeeze(geometry_map)
    geometry_map = torch.permute(geometry_map, (1,2,0))

    score_map = score_map.detach().numpy()
    geometry_map = geometry_map.detach().numpy()

    # apply score map threshold
    xy_text = np.argwhere(score_map > score_map_thresh)

    # sort the text boxes along the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    # restore rectangles from maps
    text_boxes_restored = restore_rectangle(xy_text[:, ::-1] * 4, geometry_map[xy_text[:, 0], xy_text[:, 1], :])
    print(f"text boxes before NMS: {len(text_boxes_restored)}")

    if text_boxes_restored.shape[0] == 0:
        return None

    # set up candidates for NMS
    boxes = np.zeros((text_boxes_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_boxes_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

    # NMS
    boxes = la_nms(boxes.astype(np.float32), nms_thresh)
    print(f"text boxes after NMS: {len(boxes)}")


    return boxes

def place_boxes_on_image(image, boxes):
    im = image.copy()
    for box in boxes:
        cv2.polylines(im[:, :], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
    
    return im

def validate_boxes(boxes):
    valid_boxes = []
    for i, box in enumerate(boxes):
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            # print('wrong direction')
            continue

        if box[0, 0] < 0 or box[0, 1] < 0 or box[1, 0] < 0 or box[1, 1] < 0 or box[2, 0] < 0 or box[
            2, 1] < 0 or box[3, 0] < 0 or box[3, 1] < 0:
            continue
        
        valid_boxes.append(box[np.newaxis,:,:])
    
    if len(valid_boxes) > 1:
        return np.concatenate(valid_boxes, axis=0)
    elif len(valid_boxes) == 1:
        return valid_boxes[0]
    else:
        return None