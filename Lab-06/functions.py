import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth

def v3_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

def filter_colors_mean_shift(q, img):
    flat_image = img.reshape((-1,3))
    flat_image = np.float32(flat_image)

    bandwidth = estimate_bandwidth(flat_image, quantile=q, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth, max_iter=10, bin_seeding=True)
    ms.fit(flat_image)
    labeled=ms.labels_
    
    colors_to_map = ms.cluster_centers_
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i,j]
            lowest_distance = 999
            nearest_color = None
            for color in colors_to_map:
                distance = v3_distance(color, pixel)
                if distance < lowest_distance:
                    lowest_distance = distance
                    nearest_color = color
            img[i,j] = nearest_color    
    return img

def region_growing(img, seed, threshold, color):
    img = img.copy()
    segmented = np.zeros(img.shape)
    black = np.array([0, 0, 0], dtype=np.float64)
    segmented[seed[0], seed[1]] = color
    current_region = [seed]
    while current_region:
        new_region = []
        for pixel in current_region:
            # fill region using neighbors
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if pixel[0]+i >= 0 and pixel[0]+i < img.shape[0] and pixel[1]+j >= 0 and pixel[1]+j < img.shape[1]:
                        # if the pixel is not segmented and the difference between the pixel and the seed is less than the threshold
                        is_segmented = (segmented[pixel[0]+i, pixel[1]+j] == black).all()
                        is_similar = (v3_distance(img[pixel[0], pixel[1]], img[pixel[0]+i, pixel[1]+j]) < threshold).all()

                        if (is_segmented and is_similar):
                            segmented[pixel[0]+i, pixel[1]+j] = color
                            new_region.append([pixel[0]+i, pixel[1]+j])

        current_region = new_region
    return segmented

def multi_seed_region_growing(img, seeds, threshold, colors):
    if len(seeds)!=len(colors):
        print("Error: seeds and colors must have the same length")
        return
    segmented = np.zeros(img.shape, dtype=np.float64)
    for i in range(len(seeds)):
        segmented += region_growing(img, seeds[i], threshold, colors[i])
    return segmented

def filter_colors_k_means(n_clusters, img):
    clt = KMeans(n_clusters=n_clusters)
    clt.fit(img.reshape(-1,3))
    colors_to_map = clt.cluster_centers_
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i,j]
            lowest_distance = 999
            nearest_color = None
            for color in colors_to_map:
                distance = v3_distance(color, pixel)
                if distance < lowest_distance:
                    lowest_distance = distance
                    nearest_color = color
            img[i,j] = nearest_color    
    return img