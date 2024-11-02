import cv2
import numpy as np

def sobel_x(image):
    """
    執行自定義的 Sobel X 邊緣檢測。先將影像轉換為灰階，進行高斯模糊，再應用 Sobel X 濾波器。
    """
    # 將影像轉為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0, sigmaY=0)
    
    # 定義 Sobel X 濾波器
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    
    # 應用濾波器
    sobel_x = cv2.filter2D(blurred, -1, sobel_x_kernel)
    cv2.imshow("Sobel X", sobel_x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return sobel_x

def sobel_y(image):
    """
    執行自定義的 Sobel Y 邊緣檢測。先將影像轉換為灰階，進行高斯模糊，再應用 Sobel Y 濾波器。
    """
    # 將影像轉為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0, sigmaY=0)
    
    # 定義 Sobel Y 濾波器
    sobel_y_kernel = np.array([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]])
    
    # 應用濾波器
    sobel_y = cv2.filter2D(blurred, -1, sobel_y_kernel)
    cv2.imshow("Sobel Y", sobel_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return sobel_y

def combine_and_threshold(sobel_x, sobel_y, threshold1=128, threshold2=28):
    """
    結合 Sobel X 和 Sobel Y 的結果，並進行閾值化處理。
    """
    # 計算 Sobel X 和 Sobel Y 的組合
    combined = np.sqrt(sobel_x**2 + sobel_y**2)
    combined = np.uint8(combined)
    
    # 歸一化至 0~255 範圍
    normalized = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
    
    # 應用第一個閾值
    _, thresholded1 = cv2.threshold(normalized, threshold1, 255, cv2.THRESH_BINARY)
    
    # 應用第二個閾值
    _, thresholded2 = cv2.threshold(normalized, threshold2, 255, cv2.THRESH_BINARY)
    
    # 顯示結果
    cv2.imshow("Combined", normalized)
    cv2.imshow(f"Thresholded (T={threshold1})", thresholded1)
    cv2.imshow(f"Thresholded (T={threshold2})", thresholded2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return normalized, thresholded1, thresholded2

def gradient_angle(sobel_x, sobel_y):
    """
    計算梯度角度，並基於特定的角度範圍生成兩個遮罩，最終顯示符合範圍的結果。
    """
    # 計算梯度角度（弧度轉度數）
    angle = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)
    angle = angle % 360  # 確保角度在 0~360 度之間
    
    # 創建兩個遮罩
    mask1 = np.zeros_like(angle, dtype=np.uint8)
    mask2 = np.zeros_like(angle, dtype=np.uint8)
    
    # 遮罩1：角度範圍 170° ~ 190°
    mask1[(angle >= 170) & (angle <= 190)] = 255
    
    # 遮罩2：角度範圍 260° ~ 280°
    mask2[(angle >= 260) & (angle <= 280)] = 255
    
    # 將 Sobel X 和 Sobel Y 組合的圖像
    combined_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    combined_magnitude = np.uint8(combined_magnitude)
    
    # 應用遮罩
    result1 = cv2.bitwise_and(combined_magnitude, mask1)
    result2 = cv2.bitwise_and(combined_magnitude, mask2)
    
    # 顯示結果
    cv2.imshow("Gradient Angle Mask 170°~190°", result1)
    cv2.imshow("Gradient Angle Mask 260°~280°", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result1, result2
