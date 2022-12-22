import numpy as np
import cv2 
import matplotlib.pyplot as plt

#  function to calculate histogram of an image
def histogram(img):
    hist = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i,j]] += 1
    return hist

#  implement multilevel threhsolding algorithm
def multilevel_thresholding(img, threshold_array):
    img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for t in range(1, len(threshold_array)-1):
                if img[i,j] < threshold_array[t+1]:
                    img[i,j] = threshold_array[t]
                    break
    return img

#  implement distance-based thresholding algorithm
def multilevel_thresholding_distance(img, dist):
    th_array = np.arange(0, 255, dist, dtype=np.uint8)
    return multilevel_thresholding(img, th_array)

# implement otsu algortihm
def otsu(img):
    n_pixels = (img.shape[0] * img.shape[1])
    his, bins = np.histogram(img, np.arange(0,257))
    final_thresh = -1
    final_value = -1
    
    # create the intensity array and the variation array
    intensity_arr = np.arange(256)
    
    for t in bins[1:-1]:
        # compute the class variation and store it
        pixels_background, pixels_foreground = np.sum(his[:t]), np.sum(his[t:])
        w_b, w_f = pixels_background / n_pixels, pixels_foreground / n_pixels
        m_b, m_f = np.sum(intensity_arr[:t] * his[:t]) / pixels_background, np.sum(intensity_arr[t:] * his[t:]) / pixels_foreground
        icd = w_b * w_f * (m_b - m_f) ** 2

        if icd > final_value:
            final_thresh = t
            final_value = icd
    
    # perform thresholdization
    final_img = img.copy()
    final_img[img > final_thresh] = 255
    final_img[img < final_thresh] = 0

    return final_img

def region_growing(img, seed, threshold):
    img = img.copy()
    segmented = np.zeros(img.shape)
    segmented[seed[0], seed[1]] = 255
    current_region = [seed]
    while current_region:
        new_region = []
        for pixel in current_region:
            # fill region using neighbors
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if pixel[0]+i >= 0 and pixel[0]+i < img.shape[0] and pixel[1]+j >= 0 and pixel[1]+j < img.shape[1]:
                        # if the pixel is not segmented and the difference between the pixel and the seed is less than the threshold
                        if segmented[pixel[0]+i, pixel[1]+j] == 0 and abs(img[pixel[0], pixel[1]] - img[pixel[0]+i, pixel[1]+j]) < threshold:
                            segmented[pixel[0]+i, pixel[1]+j] = 255
                            # add the pixel to the new region
                            new_region.append([pixel[0]+i, pixel[1]+j])
        current_region = new_region
    return segmented


## dos platos multi-level th
image = cv2.imread('dosPlatos.png',0)
cv2.imwrite('./results/dosPlatos_th.png', multilevel_thresholding_distance(image, 64))


## aguacate ct1 otsu cv2
aguacate = cv2.imread('aguacateBW.png',0)
ct1 = cv2.imread('CT_1.png',0)

ret1, th1 = cv2.threshold(aguacate, 0, 255, cv2.THRESH_OTSU)
ret2, th2 = cv2.threshold(ct1, 0, 255, cv2.THRESH_OTSU)

cv2.imwrite('./results/aguacate_otsu_cv2.png', th1)
cv2.imwrite('./results/ct1_otsu_cv2.png', th2)

# apply otsu algorithm
th1 = otsu(aguacate)
th2 = otsu(ct1)

cv2.imwrite('./results/aguacate_otsu_custom.png', th1)
cv2.imwrite('./results/ct1_otsu_custom.png', th2)


# region growing

seed = [int(image.shape[0]/2), int(image.shape[1]/2)]
segmented = region_growing(th2, seed, ret2)
cv2.imwrite('./results/artifact_removal.png', segmented)