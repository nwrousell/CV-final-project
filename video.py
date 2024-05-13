import math
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageTransform

# from translator import Translator 

# define display text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.75
text_color = (0, 0, 0)
text_thickness = 0

# define bounding box parameters
box_color = (0, 0, 0)
box_thickness = 2

# handles translations
# translator = Translator()

# define a video capture object 
# vid = cv2.VideoCapture(0)

# Get font
# arial = ImageFont.truetype("Arial.ttf", 30)

def get_image_from_bounding_box(frame, top_left, bottom_left, bottom_right, top_right):
    # Open starting image and ensure RGB
    im = Image.fromarray(frame)
    width = int(math.dist(top_left, top_right))
    height = int(math.dist(top_left, bottom_left))
    # print("width ", width)
    # print("height ", height)

    # Define 8-tuple with x,y coordinates of top-left, bottom-left, bottom-right and top-right corners and apply
    transform = top_left + bottom_left + bottom_right + top_right
    # print("transform: ", transform)
    result = im.transform((width,height), ImageTransform.QuadTransform(transform))

    # Save the result
    # result.save('result.png')
    return result