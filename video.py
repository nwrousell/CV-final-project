import math
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageTransform
import numba as nb

from translator import Translator 

# define display text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.75
text_color = (0, 0, 0)
text_thickness = 0

# define bounding box parameters
box_color = (0, 0, 0)
box_thickness = 2

# handles translations
translator = Translator()

# define a video capture object 
vid = cv2.VideoCapture(0)

# Get font
arial = ImageFont.truetype("Arial.ttf",30)

def get_image_from_bounding_box(frame, top_left, bottom_left, bottom_right, top_right):
    # Open starting image and ensure RGB
    im = Image.fromarray(frame)
    width = int(math.dist(top_left, top_right))
    height = int(math.dist(top_left, bottom_left))
    print("width ", width)
    print("height ", height)

    # Define 8-tuple with x,y coordinates of top-left, bottom-left, bottom-right and top-right corners and apply
    transform= top_left + bottom_left + bottom_right + top_right
    print("transform: ", transform)
    result = im.transform((width,height), ImageTransform.QuadTransform(transform))

    # Save the result
    result.save('result.png')
    return result


# given bounding box defined by top-left and bottom-right corners and text to be translated,
# translates the text and prints it back to the image with the bounding box
def translate_and_show(frame, top_left, bottom_left, bottom_right, top_right, source_language: str, target_language: str, text: str):    
    # place text in bounding box
    # x, y = top_left[0], top_left[1]
    # w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]

    save_intermediates = False

    width = int(math.dist(top_left, top_right))
    height = int(math.dist(top_left, bottom_left))
    print("width ", width)
    print("height ", height)

    margin = 5
    line_buffer = 10

    # # Blur background slightly
    # ROI = frame[y:y+h, x:x+w]
    # blur = cv2.GaussianBlur(ROI, (51,51), 0) 
    # frame[y:y+h, x:x+w] = blur

    # Workaround for non-ASCII text to display text
    # text_start = (top_left[0] + margin, top_left[1] + 2 * margin)
    translation = translator.get_translation(source_language, target_language, text)

    # Create black mask using Numpy and convert from BGR (OpenCV) to RGB (PIL)
    # image = np.zeros((height, width, 3), dtype=np.uint8)
    image = np.full((height, width, 3), (100, 100, 100), dtype=np.uint8)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    text_size = cv2.getTextSize(translation, font, font_scale, text_thickness)
    ((text_width, text_height), baseline) = text_size
    line_start = (margin, 0)
    if text_width > width:
        line_width = 0
        line = ""
        for word in translation.split(" "):
            ((word_width, _), _) = cv2.getTextSize(word + " ", font, font_scale, text_thickness)
            if word_width + line_width < width - 2 * margin:
                line += word + " "
                line_width += word_width
            else:
                line_start = (line_start[0], line_start[1] + text_height + line_buffer)
                draw.text(line_start, line, font=arial, fill=(255,255,255))
                line = word + " "
                line_width = word_width
        # handle last word if it breaks lines
        if line != "":
            line_start = (line_start[0], line_start[1] + text_height + line_buffer)
            draw.text(line_start, line, font=arial, fill=(255,255,255))
    else:
        line_start = (line_start[0], line_start[1] + text_height)
        draw.text(line_start, translation, font=arial, fill=(255,255,255))

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
    print("radians: ", radians)
    rotation_angle = math.degrees(radians)
    print("rotation_angle: ", rotation_angle)
    # rotation_angle = 45

    width = image.shape[1]
    # pivot_point = (width/2, offset_from_center)
    # pivot_point = (bottom_right[0] - top_left[0], top_left[1] - bottom_right[1])
    pivot_point = (width / 2, height / 2)
    print("pivot_point: ", pivot_point)

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
    print("canvas_center: ", canvas_center)
    # image_center = (bottom_right[0] - top_left[0], top_left[1] - bottom_right[1])
    image_center = (bottom_right[0], bottom_right[1])
    print("image_center: ", image_center)
    dx = image_center[0] - canvas_center[0]
    dy = image_center[1] - canvas_center[1]
    print("dx, dy: ", dx, ", ", dy)
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
    # tmp = transformed_image.astype(np.uint16)
    # alpha = (tmp[:,:,0] + tmp[:,:,1] + tmp[:,:,2]) > 0

    # alpha = alpha.astype(np.float32)

    # alpha = np.dstack((alpha, alpha, alpha))

    # transformed_image = transformed_image.astype(np.float32)
    # canvas = frame.astype(np.float32)

    # foreground = np.multiply(alpha, transformed_image)
    # np.subtract(1.0, alpha, out=alpha)
    # np.multiply(alpha, canvas, out=canvas)

    # np.add(foreground, canvas, out=canvas)
    # canvas = canvas.astype(np.uint8)

    # if save_intermediates:
    #     bgr_image = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    #     img = Image.fromarray(bgr_image)
    #     img.save('final.png')
    # return canvas

    # overlay translation box
    # frame[x:x+w,y:y+h,:] = image[0:w,0:h,:]

    # display bounding box
    # cv2.rectangle(frame, top_left, bottom_right, box_color, box_thickness)

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

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    
    # display text
    # translate_and_show(frame, (200, 200), (500, 230), (200, 260), (500, 290), None, "es", "Really long string to test line breaks and bounding box fitting! How cool?!! ")
    frame = translate_and_show(frame, (31*2, 146*2), (88*2, 226*2), (252*2, 112*2), (195*2, 31*2), None, "es", "Really long string to test line breaks and bounding box fitting! How cool?!! ")


    # frame = translate_and_show(frame, (31*2 + 300, 146*2 + 300), (88*2 + 300, 226*2 + 300), (252*2 + 300, 112*2 + 300), (195*2 + 300, 31*2 + 300), None, "es", "Really long string to test line breaks and bounding box fitting! How cool?!! ")
    
    # failed attempt to test rotation
    # angle = math.radians(0)

    # px = 1000
    # py = 1000
    # ox = frame.shape[1] / 2
    # oy = frame.shape[0] / 2

    # qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    # qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    # print("angle: ", angle)
    # print("qx, qy: (", qx, ",", qy, ")")

    # frame = translate_and_show(frame, (px, py), (px, qy), (qx, qy), (qx, py), None, "es", "Really long string to test line breaks and bounding box fitting! How cool?!! ")
  
    # testing
    # get_image_from_bounding_box(frame, (31*3, 146*3), (88*3, 226*3), (252*3, 112*3), (195*3, 31*3))

    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

    # 'p' to print currently stored translations
    if cv2.waitKey(1) & 0xFF == ord('p'): 
        print(translator.translations)
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 