import cv2
import numpy as np

def gaussian_blur(image):
    """
    顯示一個帶有 Trackbar 的窗口，用於動態調整高斯模糊的核大小。
    核大小： (2 * m + 1) x (2 * m + 1)，範圍： m = [1, 5]
    """
    def on_trackbar(val):
        m = max(val, 1)
        ksize = (2 * m + 1, 2 * m + 1)
        blurred_image = cv2.GaussianBlur(image, ksize, sigmaX=0)
        cv2.imshow("Gaussian Blur", blurred_image)

    cv2.namedWindow("Gaussian Blur")
    cv2.createTrackbar("Kernel Radius (m)", "Gaussian Blur", 1, 5, on_trackbar)
    on_trackbar(1)  # 初始化時顯示核大小為 (2 * 1 + 1) x (2 * 1 + 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bilateral_filter(image):
    """
    顯示一個帶有 Trackbar 的窗口，用於動態調整雙邊濾波的核大小。
    核大小： (2 * m + 1) x (2 * m + 1)，範圍： m = [1, 5]
    sigmaColor 和 sigmaSpace 固定為 90。
    """
    def on_trackbar(val):
        m = max(val, 1)
        d = 2 * m + 1
        filtered_image = cv2.bilateralFilter(image, d=d, sigmaColor=90, sigmaSpace=90)
        cv2.imshow("Bilateral Filter", filtered_image)

    cv2.namedWindow("Bilateral Filter")
    cv2.createTrackbar("Kernel Radius (m)", "Bilateral Filter", 1, 5, on_trackbar)
    on_trackbar(1)  # 初始化時顯示核大小為 (2 * 1 + 1) x (2 * 1 + 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def median_filter(image):
    """
    顯示一個帶有 Trackbar 的窗口，用於動態調整中值濾波的核大小。
    核大小： (2 * m + 1) x (2 * m + 1)，範圍： m = [1, 5]
    """
    def on_trackbar(val):
        m = max(val, 1)
        ksize = 2 * m + 1
        filtered_image = cv2.medianBlur(image, ksize)
        cv2.imshow("Median Filter", filtered_image)

    cv2.namedWindow("Median Filter")
    cv2.createTrackbar("Kernel Radius (m)", "Median Filter", 1, 5, on_trackbar)
    on_trackbar(1)  # 初始化時顯示核大小為 (2 * 1 + 1) x (2 * 1 + 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
