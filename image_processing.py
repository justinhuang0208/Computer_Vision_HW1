# image_processing.py
import cv2
import numpy as np

def color_separation(image):
    """分離 BGR 三個通道，並返回 B、G、R 通道影像。"""
    b, g, r = cv2.split(image)
    zeros = np.zeros_like(b)
    b_image = cv2.merge([b, zeros, zeros])
    g_image = cv2.merge([zeros, g, zeros])
    r_image = cv2.merge([zeros, zeros, r])
    return b_image, g_image, r_image

def color_transformation(image):
    """將影像轉換為灰階並返回 cv_gray 與 avg_gray 影像。"""
    cv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)
    avg_gray = (b / 3 + g / 3 + r / 3).astype(np.uint8)
    return cv_gray, avg_gray

def color_extraction(image):
    """提取黃綠色範圍並返回 mask、mask_inverse 與提取後的影像。"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([18, 0, 25])
    upper_bound = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    mask_inverse = cv2.bitwise_not(mask)
    extracted_image = cv2.bitwise_and(image, image, mask=mask_inverse)
    return mask, mask_inverse, extracted_image