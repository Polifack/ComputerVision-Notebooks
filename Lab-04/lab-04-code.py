import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
Function library employed in jupyter notebook
'''

def median_canny_edge(img, g_ksize, median_diff, a_size, equalize=False):
    if equalize:
        img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img,(g_ksize, g_ksize),0)
    
    # compute threshold values from median
    median = np.median(img)
    th_low = median*(1-median_diff)
    th_high = median*(1+median_diff)

    edges = cv2.Canny(img, 
                      threshold1=th_low, 
                      threshold2=th_high, 
                      apertureSize=a_size,
                      L2gradient=True)
    return edges


def otsu_canny_edge(img, g_ksize, a_size, equalize=False):
    if equalize:
        img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img,(g_ksize, g_ksize),0)
    
    # compute threshold values from otsu 
    th_high, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_low = 0.5*th_high

    edges = cv2.Canny(img, 
                      threshold1=th_low, 
                      threshold2=th_high, 
                      apertureSize=a_size,
                      L2gradient=True)
    return edges


def gaussian_pyramid(image, scale=2, minSize=(200, 200)):
   yield image
   while True:
     w = int(image.shape[1] / scale)
     h = int(image.shape[0] / scale)
     image = cv2.resize(image, (w, h))
     if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
        break
     yield image


def draw_hough_lines(image, c_k, c_th_u, c_th_l, c_op):
    smooth = cv2.GaussianBlur(image,(c_k,c_k),0)
    edges = cv2.Canny(smooth,c_th_u,c_th_l,apertureSize=c_op,L2gradient=True)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,
                 minLineLength=50,maxLineGap=10)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv2.LINE_AA)
    return image


def count_hough_circles(img, smoothing_kernel, dp, min_dist, param1, param2, min_r, max_r):
    img = cv2.GaussianBlur(img,(smoothing_kernel,smoothing_kernel),2)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,dp,min_dist, param1=param1,param2=param2,minRadius=min_r,maxRadius=max_r)

    if circles is None:
        return img, 0

    circles = circles[0,:]
    # remove circles that are too close to each other
    for i in range(len(circles)):
        for j in range(len(circles)):
            if i != j:
                # compute euclidean distance between centers
                dist = np.sqrt((circles[i,0]-circles[j,0])**2+(circles[i,1]-circles[j,1])**2)
                if dist < 2*circles[i,2]:
                    circles[j] = None

    # remove nan
    circles_final = []
    for i in range(len(circles)):
        if not np.isnan(circles[i]).any():
            circles_final.append(circles[i])
    
    # draw circles
    if circles_final is not None:
        circles_final = np.uint16(np.around(circles_final))
        for i in circles_final:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
    return cimg, len(circles_final)
