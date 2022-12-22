import numpy as np
import cv2 
import matplotlib.pyplot as plt

# constants
intro_images_path='./Images/Intro_images/'

def adaptative_histogram_equalization(image, clip_limit=2.0, grid_size=(8,8)):
    ''' 
    Given an image, this function applies an adaptative histogram equalization
    @inputs: image, clip_limit, grid_size
    @outputs: image
    '''
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)  

def gaussian_pyramid(image, scale=2, minSize=(200, 200)):
    '''
    Given an image, this function applies a gaussian pyramid
    @inputs: image, scale, minSize
    @outputs: image
    '''
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image


# Read the template and image
tuna_template = cv2.imread(intro_images_path+'Tuna_template.jpg', cv2.IMREAD_GRAYSCALE)
tuna_sizes = cv2.imread(intro_images_path+'Tuna_Relative_Sizes.jpg', cv2.IMREAD_GRAYSCALE)

tuna_sizes_eq = adaptative_histogram_equalization(tuna_sizes, clip_limit=4.0, grid_size=(128,128))
tuna_template_eq = adaptative_histogram_equalization(tuna_template, clip_limit=4.0, grid_size=(128,128))

# create a list of all the pyramid layers
tuna_template_pyramid = list(gaussian_pyramid(tuna_template_eq, scale=1.5))
tuna_image_rectangles = tuna_sizes_eq.copy()

# loop over the layers of the tuna image pyramid 
for tuna_template_layer in tuna_template_pyramid:
   # match the tuna template to the tuna image using cv2.TM_CCOEFF_NORMED
   result = cv2.matchTemplate(tuna_sizes_eq, tuna_template_layer, cv2.TM_CCOEFF_NORMED)
   # find the location of the best match
   min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
   # draw a rectangle around the best match
   cv2.rectangle(tuna_image_rectangles, max_loc, (max_loc[0] + tuna_template_layer.shape[1], max_loc[1] + tuna_template_layer.shape[0]), (0, 255, 0), 10)

# save rectangles image
cv2.imwrite('tuna_image_rectangles.jpg', tuna_image_rectangles)