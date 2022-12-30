import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner_detector(img, k=0.04, window_size=3, threshold=0.01, draw_circles=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    gray = cv2.GaussianBlur(gray, (window_size, window_size), 0)

    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=window_size)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=window_size)
    
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    Sxx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)
    Syy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)
    
    det = (Sxx * Syy) - (Sxy * Sxy)
    trace = Sxx + Syy
    R = det - k * (trace ** 2)
    R[R < threshold * R.max()] = 0

    if draw_circles:
        coords = np.where(R > 0.01 * R.max())
        for i in range(len(coords[0])):
            cv2.circle(img, (coords[1][i], coords[0][i]), 5, (0, 255, 0), -1)
        return img
    
    else:
        return R

def harris_corner_borders(img, l=[100,200]):
    # Harris corner detector that detects edges on corners
    coords = []
    for i in range(len(l)):
        b_img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[l[i], l[i], l[i]])
        h_img = harris_corner_detector(img)
        coords += np.where(h_img > 0)
    
    dst = np.zeros(img.shape, dtype=np.uint8)
    for i in range(len(coords[0])):
        cv2.circle(dst, (coords[1][i], coords[0][i]), 10, (255, 255, 255), -1)
    
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst.astype(np.uint8))
    return len(labels-1)
