import math
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageTransform
import numba as nb
import os
from pathlib import Path


# define display text parameters
ascii_font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.50
text_color = (0, 0, 0)
text_thickness = 0

# define bounding box parameters
box_color = (0, 0, 0)
box_thickness = 2

# Get font
# roboto = ImageFont.truetype("fonts/Robot-Regular.ttf",10)

# expands the box by the specified amount in the long axis
# box is a (4,2) numpy array
# scale is a float, e.g. 0.1 = 10% larger
def expand_box(box, scale):
    top_left = box[0]
    top_right = box[1]
    bottom_left = box[3]
    bottom_right = box[2]
    width = top_right[0] - top_left[0]
    height = bottom_left[1] - top_left[1]
    dx = width * scale
    dy = height * scale
    slope = (top_right[1] - top_left[1]) / (top_right[0] - top_left[0])

    if slope > 0:
        top_left[0] = top_left[0] - dx
        top_left[1] = top_left[1] - dy
        top_right[0] = top_right[0] + dx
        top_right[1] = top_right[1] + dy
        bottom_left[0] = bottom_left[0] - dx
        bottom_left[1] = bottom_left[1] - dy
        bottom_right[0] = bottom_right[0] + dx
        bottom_right[1] = bottom_right[1] + dy
    else:
        top_left[0] = top_left[0] - dx
        top_left[1] = top_left[1] + dy
        top_right[0] = top_right[0] + dx
        top_right[1] = top_right[1] - dy
        bottom_left[0] = bottom_left[0] - dx
        bottom_left[1] = bottom_left[1] + dy
        bottom_right[0] = bottom_right[0] + dx
        bottom_right[1] = bottom_right[1] - dy

    box[0] = top_left
    box[1] = top_right
    box[2] = bottom_right
    box[3] = bottom_left
    return box

def get_image_from_bounding_box(frame, top_left, bottom_left, bottom_right, top_right):
    # Open starting image and ensure RGB
    im = Image.fromarray(frame)
    width = int(math.dist(top_left, top_right))
    height = int(math.dist(top_left, bottom_left))

    # Define 8-tuple with x,y coordinates of top-left, bottom-left, bottom-right and top-right corners and apply
    transform= top_left + bottom_left + bottom_right + top_right
    result = im.transform((width,height), ImageTransform.QuadTransform(transform))

    # Save the result
    # result.save('result.png')
    return result

# given bounding box defined by top-left and bottom-right corners and text to be translated,
# translates the text and prints it back to the image with the bounding box
def render_text_box(frame, top_left, bottom_left, bottom_right, top_right, text: str):    
    save_intermediates = False

    width = int(math.dist(top_left, top_right))
    height = int(math.dist(top_left, bottom_left))

    # Workaround for non-ASCII text to display text
    # translation = translator.get_translation(source_language, target_language, text)

    # Create mask using Numpy and convert from BGR (OpenCV) to RGB (PIL)
    image = np.full((height, width, 3), (150, 150, 150), dtype=np.uint8)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # portion of image width you want text width to be
    img_fraction = 0.8

    fontsize = 10
    roboto = ImageFont.truetype("fonts/Robot-Regular.ttf", fontsize)
    while roboto.getbbox(text)[3] < img_fraction*image.shape[0]:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        roboto = ImageFont.truetype("fonts/Robot-Regular.ttf", fontsize)

    draw.text((0,0), text, font=roboto, fill=(255,255,255))

    # Convert back to Numpy array and switch back from RGB to BGR
    image = np.asarray(pil_image)
    
    if save_intermediates:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(bgr_image)
        img.save('text.png')

    # attempt to rotate
    # compute rotation angle
    dx = top_right[0] - top_left[0]
    dy = bottom_left[0] - top_left[0]
    radians = np.arctan2(dy, dx)
    # print("radians: ", radians)
    rotation_angle = math.degrees(-radians)
    # print("rotation_angle: ", rotation_angle)
    # rotation_angle = 45

    width = image.shape[1]
    # pivot_point = (width/2, offset_from_center)
    # pivot_point = (bottom_right[0] - top_left[0], top_left[1] - bottom_right[1])
    pivot_point = (width / 2, height / 2)
    # print("pivot_point: ", pivot_point)

    rotation_mat = cv2.getRotationMatrix2D(pivot_point, -rotation_angle, 1.)

    canvas_height = frame.shape[0]
    canvas_width = frame.shape[1]

    rotation_mat[0, 2] += canvas_width/2 - pivot_point[0]
    rotation_mat[1, 2] += canvas_height/2 - pivot_point[1]

    rotated_image = cv2.warpAffine(image,
                                rotation_mat,
                                (canvas_width, canvas_height))
    
    if save_intermediates:
        rot_img = Image.fromarray(rotated_image)
        rot_img.save('rotated_text.png')

    # add translation
    canvas_center = (canvas_width / 2, canvas_height / 2)
    # print("canvas_center: ", canvas_center)
    # image_center = (bottom_right[0] - top_left[0], top_left[1] - bottom_right[1])
    avg_x = int((top_left[0] + top_right[0] ) / 2)
    avg_y = int((top_left[1] + bottom_left[1] ) / 2)
    image_center = (avg_x, avg_y)
    # print("image_center: ", image_center)
    dx = image_center[0] - canvas_center[0]
    dy = image_center[1] - canvas_center[1]
    # print("dx, dy: ", dx, ", ", dy)
    M = np.float32([
        [1,0,dx],
        [0,1,dy]
    ])
    
    transformed_image = cv2.warpAffine(rotated_image, M, (canvas_width, canvas_height))

    if save_intermediates:
        trans_img = Image.fromarray(transformed_image)
        trans_img.save('transformed_text.png')

    canvas = frame.astype(np.uint8)
    compute(transformed_image, canvas)
    if save_intermediates:
        bgr_image = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(bgr_image)
        img.save('result.png')

    return canvas

@nb.njit('(uint8[:,:,::1], uint8[:,:,::1])', parallel=True)
def compute(img, canvas):
    for i in nb.prange(img.shape[0]):
        for j in range(img.shape[1]):
            ir = np.float32(img[i, j, 0])
            ig = np.float32(img[i, j, 1])
            ib = np.float32(img[i, j, 2])
            cr = np.float32(canvas[i, j, 0])
            cg = np.float32(canvas[i, j, 1])
            cb = np.float32(canvas[i, j, 2])
            alpha = np.float32((ir + ig + ib) > 0)
            inv_alpha = np.float32(1.0) - alpha
            cr = inv_alpha * cr + alpha * ir
            cg = inv_alpha * cg + alpha * ig
            cb = inv_alpha * cb + alpha * ib
            canvas[i, j, 0] = np.uint8(cr)
            canvas[i, j, 1] = np.uint8(cg)
            canvas[i, j, 2] = np.uint8(cb)