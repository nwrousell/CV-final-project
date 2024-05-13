import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

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


# given bounding box defined by top-left and bottom-right corners and text to be translated,
# translates the text and prints it back to the image with the bounding box
def translate_and_show(frame, top_left: tuple[int, int], bottom_right: tuple[int, int], source_language: str, target_language: str, text: str):
    # place text in bounding box
    x, y = top_left[0], top_left[1]
    w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]

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
    image = np.zeros((w, h, 3), dtype=np.uint8)
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    # draw.text((margin, margin), translation, font=kbd, fill=(255,255,255))

    text_size = cv2.getTextSize(translation, font, font_scale, text_thickness)
    ((text_width, text_height), baseline) = text_size
    # print("text_size ", text_size)
    line_start = (margin, 0)
    if text_width > w:
        line_width = 0
        line = ""
        for word in translation.split(" "):
            ((word_width, _), _) = cv2.getTextSize(word + " ", font, font_scale, text_thickness)
            if word_width + line_width < w - 2 * margin:
                line += word + " "
                line_width += word_width
            else:
                line_start = (line_start[0], line_start[1] + text_height + line_buffer)
                draw.text(line_start, line, font=arial, fill=(255,255,255))
                # cv2.putText(frame, line, line_start, font, font_scale, text_color, text_thickness)
                line = word + " "
                line_width = word_width
        # handle last word if it breaks lines
        if line != "":
            line_start = (line_start[0], line_start[1] + text_height + line_buffer)
            draw.text(line_start, line, font=arial, fill=(255,255,255))
            # cv2.putText(frame, line, line_start, font, font_scale, text_color, text_thickness)
    else:
        line_start = (line_start[0], line_start[1] + text_height)
        draw.text(line_start, translation, font=arial, fill=(255,255,255))
        # cv2.putText(frame, translation, line_start, font, font_scale, text_color, text_thickness)

    # Convert back to Numpy array and switch back from RGB to BGR
    image = np.asarray(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # overlay translation box
    frame[x:x+w,y:y+h,:] = image[0:w,0:h,:]

    # display bounding box
    # cv2.rectangle(frame, top_left, bottom_right, box_color, box_thickness)


while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    
    # display text
    translate_and_show(frame, (200, 200), (500, 500), None, "es", "Really long string to test line breaks and bounding box fitting! How cool?!! ")
  
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

