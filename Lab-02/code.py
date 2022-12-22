import numpy as np
from matplotlib import pyplot as plt
import cv2

# constants
circle_radius = 10
key_delay     = 20
src_color  = (0, 0, 255)
trg_color  = (255, 0, 0)
intro_images_path = './Images/Intro_images/'
window_name = 'Image tuner'


def fit_on_dimensions(image, target_y, target_x):
    '''
    Given an image and target dimensions, this function resizes the image
    to fit on the dimensions while keeping aspect ratio.
    
    @inputs:  image, target_height, target_width
    
    @outputs: image resized keeping aspect ratio scaled 
              to fit in the desired dimensions
    '''
    image_y, image_x, _ = image.shape
    image_ratio = image_x/image_y

    if (image_y < target_y) or (image_x < target_x):
        if (target_y < target_x):
            final_y = int(target_y)
            final_x = int(image_ratio * final_y)
        else:
            final_y = int(image_ratio * target_x)
            final_x = int(target_x)
    else:
        if (target_y > target_x):
            final_y = int(target_y)
            final_x = int(image_ratio * final_y)
        else:
            final_y = int(image_ratio * target_x)
            final_x = int(target_x)
            
    image=cv2.resize(image, (final_y, final_x))
    return image

def fit_on_position(image, frame_y, frame_x, pos_y, pos_x):
    '''
    Given a image, a frame dimensions and a starting position
    this function will generate a black frame of (frame_y, frame_x) dimensions
    and plot the image on it placing the image's top-left corner on position
    (pos_y, pos_x). Overflowing parts from the image will be trimmed out
    
    @inputs:  image, frame_dimensions, plot_point.
    
    @outputs: black frame with image plotted at plot_point. 
    '''
    image_y, image_x, _ = image.shape
    frame = np.zeros((frame_y,frame_x,3),np.uint8)

    ## fits x

    if pos_x < 0:
        frame_start_x = 0
        frame_end_x   = pos_x + image_x
        image_start_x = abs(pos_x)
        image_end_x   = image_x
    
    elif pos_x+image_x > frame_x:
        frame_start_x = pos_x
        frame_end_x   = frame_x
        image_start_x = 0
        image_end_x   = frame_x - pos_x
    
    else:
        frame_start_x = pos_x
        frame_end_x   = pos_x + image_x
        image_start_x = 0
        image_end_x   = image_x


    ## fits y

    if pos_y+image_y < 0:
        frame_start_y = 0
        frame_end_y   = pos_y + image_y
        image_start_y = abs(pos_y)
        image_end_y   = image_y
    
    elif pos_y+image_y > frame_y:
        frame_start_y = pos_y
        frame_end_y   = frame_y
        image_start_y = 0
        image_end_y   = frame_y - pos_y

    else:
        frame_start_y = pos_y
        frame_end_y   = pos_y + image_y
        image_start_y = 0
        image_end_y   = image_y

    frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x] = image[image_start_y:image_end_y, image_start_x:image_end_x]
    return frame


def draw_circle(event,x,y,flags,param):
    '''
    Callback for the get_points function. Draws a point
    on image and stores it on the params.src_points
    '''
    image = param[0]
    src_points = param[1]

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image,(x,y),circle_radius,src_color,-1)
        src_points.append((x,y))

def get_points(image):
    '''
    Function that given an image will display it until the user
    inputs 4 points on it. If the user presses the R key the points
    will be resetted
    '''
    src_points = []
    pointed_image = image.copy()
    
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, pointed_image)
    cv2.setMouseCallback(window_name, draw_circle,param=[pointed_image, src_points])

    while(1):
        cv2.imshow(window_name, pointed_image)
        if len(src_points)==4:
            return np.float32(src_points)
        if cv2.waitKey(key_delay)==ord('r'):
            src_points = []
            pointed_image = image.copy()
            cv2.setMouseCallback(window_name, draw_circle,param=[pointed_image, src_points])

def get_image_roi(source_image):
    '''
    Interactive loop that given an image will display it and wait for the user
    to input 4 points on it. After that, it will perform a trim using those points.
    '''
    src_y, src_x, _ = source_image.shape
    
    while(1):
        trim_points = get_points(source_image)
        min_y, max_y, min_x, max_x = src_y,src_x,0,0
        for point in trim_points:
            y, x = point
            min_y = int(y) if y<min_y else min_y
            min_x = int(x) if x<min_x else min_x
            max_y = int(y) if y>max_y else max_y
            max_x = int(x) if x>max_x else max_x

        img_roi = source_image[min_y:max_y, min_x:max_x]
        cv2.imshow(window_name, img_roi)
        while(1):
            if cv2.waitKey(key_delay) == ord('s'):       
                print("Saving the image")
                cv2.imwrite('roi_image.jpg', img_roi)
                exit(1)
            if cv2.waitKey(key_delay) == ord('c'):  
                print("Performing another trim")     
                get_image_roi(img_roi)
            if cv2.waitKey(key_delay) == ord('r'):       
                print("Repeating the trim")
                get_image_roi(source_image)

def tune_image_perspective(source_image, target_image):
    '''
    Interactive loop that given two images will display them and wait for the user
    to input 4 points on them. After that, it will perform a warp using those points
    as homography.
    
    @inputs:  source_image, target_image
    
    @outputs: None; Saves source_image warped. 
    '''
    while(1):
        src_points = get_points(source_image)
        trg_points = get_points(target_image)
        
        print("Source points:",src_points)
        print("Target points:",trg_points)
        
        h, _ = cv2.findHomography(src_points, trg_points, cv2.RANSAC) 

        trg_y, trg_x, _ = target_image.shape
        img_morphed = cv2.warpPerspective(source_image, h, (trg_x, trg_y))

        cv2.imshow(window_name, img_morphed)
        while(1):
            if cv2.waitKey(key_delay) == ord('s'):       
                print("Saving the image")
                cv2.imwrite('morphed_image.jpg', img_morphed)
                exit(1)
            if cv2.waitKey(key_delay) == ord('c'):  
                print("Performing another tuning")     
                main_loop(img_morphed, target_image)
            if cv2.waitKey(key_delay) == ord('r'):       
                print("Repeating the perspective tuning")
                tune_image_perspective(source_image, target_image)
            if cv2.waitKey(key_delay) == ord('t'):
                print("Trimming the image")
                get_image_roi(img_morphed)

def main_loop(source_image, target_image):
    src_y, src_x, _ = source_image.shape
    trg_y, trg_x, _ = target_image.shape

    print(source_image.shape)
    print(target_image.shape)

    source_image_resize = fit_on_dimensions(source_image, trg_y, trg_x)
    src_r_y, src_r_x, _ = source_image_resize.shape


    target_image_moved = fit_on_position(target_image, src_r_y, src_r_x, 85, -75)
    target_image_overlap = cv2.addWeighted(source_image_resize,1, target_image_moved, 1, 0)

    tune_image_perspective(source_image_resize, target_image_overlap)


# 1) Read the target and source images
trg = cv2.imread(intro_images_path+'torre_hercules_2.jpg')
src = cv2.imread(intro_images_path+'torre_hercules_1.jpg')
main_loop(src, trg)