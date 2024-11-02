# transforms.py

import cv2
import numpy as np

def rotate_image(image, angle):
    """
    旋轉影像指定的角度。

    Args:
        image (numpy.ndarray): 輸入影像。
        angle (float): 旋轉角度（度）。

    Returns:
        numpy.ndarray: 旋轉後的影像。
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 取得旋轉矩陣
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 執行旋轉
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def scale_image(image, scale_factor):
    """
    縮放影像指定的比例。

    Args:
        image (numpy.ndarray): 輸入影像。
        scale_factor (float): 縮放比例。

    Returns:
        numpy.ndarray: 縮放後的影像。
    """
    if scale_factor <= 0:
        raise ValueError("縮放比例必須大於0")
    
    scaled = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    return scaled

def translate_image(image, tx, ty):
    """
    平移影像指定的像素。

    Args:
        image (numpy.ndarray): 輸入影像。
        tx (int): 水平平移像素數。
        ty (int): 垂直平移像素數。

    Returns:
        numpy.ndarray: 平移後的影像。
    """
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def apply_transform(image, angle=0, scale=1.0, tx=0, ty=0):
    """
    結合旋轉、縮放和平移，對影像進行轉換。

    Args:
        image (numpy.ndarray): 輸入影像。
        angle (float): 旋轉角度（度）。
        scale (float): 縮放比例。
        tx (int): 水平平移像素數。
        ty (int): 垂直平移像素數。

    Returns:
        numpy.ndarray: 轉換後的影像。
    """
    transformed_image = image.copy()
    
    # 執行縮放
    if scale != 1.0:
        transformed_image = scale_image(transformed_image, scale)
    
    # 執行旋轉
    if angle != 0:
        transformed_image = rotate_image(transformed_image, angle)
    
    # 執行平移
    if tx != 0 or ty != 0:
        transformed_image = translate_image(transformed_image, tx, ty)
    
    return transformed_image
