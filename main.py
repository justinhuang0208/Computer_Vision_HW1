import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QGroupBox, QFileDialog)
from PyQt5.QtCore import Qt
import cv2
import image_processing
import image_smoothing
import edge_detection
import transforms

class ImageProcessingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image1 = None
        self.image2 = None
        self.sobel_x_result = None
        self.sobel_y_result = None

    def initUI(self):
        # 設置主視窗
        self.setWindowTitle('Image Processing')
        self.setGeometry(100, 100, 800, 600)  # 調整窗口大小以適應右側的變換參數

        # 建立中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 創建主布局
        main_layout = QHBoxLayout(central_widget)

        # 左側布局
        left_layout = QVBoxLayout()

        # 添加載入圖片按鈕
        load_btn1 = QPushButton('Load Image 1')
        load_btn1.clicked.connect(self.load_image1)
        load_btn2 = QPushButton('Load Image 2')
        load_btn2.clicked.connect(self.load_image2)
        left_layout.addWidget(load_btn1)
        left_layout.addWidget(load_btn2)

        # 創建圖像處理群組
        processing_group = QGroupBox("1. Image Processing")
        processing_layout = QVBoxLayout()
        processing_buttons = [
            ("1.1 Color Separation", self.run_color_separation),
            ("1.2 Color Transformation", self.run_color_transformation),
            ("1.3 Color Extraction", self.run_color_extraction)
        ]
        for btn_text, btn_function in processing_buttons:
            btn = QPushButton(btn_text)
            btn.clicked.connect(btn_function)
            processing_layout.addWidget(btn)
        processing_group.setLayout(processing_layout)

        # 創建圖像平滑群組
        smoothing_group = QGroupBox("2. Image Smoothing")
        smoothing_layout = QVBoxLayout()
        smoothing_buttons = [
            ("2.1 Gaussian blur", self.run_gaussian_blur),
            ("2.2 Bilateral filter", self.run_bilateral_filter),
            ("2.3 Median filter", self.run_median_filter)
        ]
        for btn_text, btn_function in smoothing_buttons:
            btn = QPushButton(btn_text)
            btn.clicked.connect(btn_function)
            smoothing_layout.addWidget(btn)
        smoothing_group.setLayout(smoothing_layout)

        # 創建邊緣檢測群組
        edge_group = QGroupBox("3. Edge Detection")
        edge_layout = QVBoxLayout()
        edge_buttons = [
            ("3.1 Sobel X", self.run_sobel_x),
            ("3.2 Sobel Y", self.run_sobel_y),
            ("3.3 Combination and Threshold", self.run_combination_and_threshold),
            ("3.4 Gradient Angle", self.run_gradient_angle)
        ]
        for btn_text, btn_function in edge_buttons:
            btn = QPushButton(btn_text)
            btn.clicked.connect(btn_function)
            edge_layout.addWidget(btn)
        edge_group.setLayout(edge_layout)

        # 添加所有群組到左側布局
        left_layout.addWidget(processing_group)
        left_layout.addWidget(smoothing_group)
        left_layout.addWidget(edge_group)
        left_layout.addStretch()

        # 右側布局
        right_layout = QVBoxLayout()

        # 創建變換參數群組
        transform_group = QGroupBox("4. Transform")
        transform_layout = QVBoxLayout()

        # 添加參數輸入欄位
        params = [
            ("Rotation:", "deg"),
            ("Scaling:", ""),
            ("Tx:", "pixel"),
            ("Ty:", "pixel")
        ]

        self.transform_inputs = {}  # 儲存 QLineEdit 控件的字典

        for param_name, unit in params:
            param_layout = QHBoxLayout()
            label = QLabel(param_name)
            input_field = QLineEdit()
            self.transform_inputs[param_name] = input_field  # 儲存控件
            param_layout.addWidget(label)
            param_layout.addWidget(input_field)
            if unit:
                unit_label = QLabel(unit)
                param_layout.addWidget(unit_label)
            transform_layout.addLayout(param_layout)

        # 添加變換按鈕並連接到 run_transform 方法
        transform_btn = QPushButton("4. Transforms")
        transform_btn.clicked.connect(self.run_transform)
        transform_layout.addWidget(transform_btn)

        transform_group.setLayout(transform_layout)

        # 將變換群組添加到右側布局
        right_layout.addWidget(transform_group)
        right_layout.addStretch()

        # 將左右布局添加到主布局
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

    # 載入圖片
    def load_image1(self):
        """載入第一張影像並儲存到 self.image1"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image1 = cv2.imread(file_path)
            if self.image1 is not None:
                print("Image 1 loaded")
                cv2.imshow("Loaded Image 1", self.image1)
            else:
                print("Failed to load Image 1")

    def load_image2(self):
        """載入第二張影像並儲存到 self.image2"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image2 = cv2.imread(file_path)
            if self.image2 is not None:
                print("Image 2 loaded")
                cv2.imshow("Loaded Image 2", self.image2)
            else:
                print("Failed to load Image 2")

    # 圖像處理函數
    def run_color_separation(self):
        if self.image1 is not None:
            b_image, g_image, r_image = image_processing.color_separation(self.image1)
            cv2.imshow("Blue Channel", b_image)
            cv2.imshow("Green Channel", g_image)
            cv2.imshow("Red Channel", r_image)
        else:
            print("No image loaded for Color Separation")

    def run_color_transformation(self):
        if self.image1 is not None:
            cv_gray, avg_gray = image_processing.color_transformation(self.image1)
            cv2.imshow("Grayscale (cvtColor)", cv_gray)
            cv2.imshow("Grayscale (Average)", avg_gray)
        else:
            print("No image loaded for Color Transformation")

    def run_color_extraction(self):
        if self.image1 is not None:
            mask, mask_inverse, extracted_image = image_processing.color_extraction(self.image1)
            cv2.imshow("Yellow-Green Mask", mask)
            cv2.imshow("Mask Inverse", mask_inverse)
            cv2.imshow("Extracted Image", extracted_image)
        else:
            print("No image loaded for Color Extraction")

    # 圖像平滑函數
    def run_gaussian_blur(self):
        if self.image1 is not None:
            blurred = image_smoothing.gaussian_blur(self.image1)
            cv2.imshow("Gaussian Blurred Image", blurred)
        else:
            print("No image loaded for Gaussian Blur")

    def run_bilateral_filter(self):
        if self.image1 is not None:
            bilateral = image_smoothing.bilateral_filter(self.image1)
            cv2.imshow("Bilateral Filtered Image", bilateral)
        else:
            print("No image loaded for Bilateral Filter")

    def run_median_filter(self):
        if self.image1 is not None:
            median = image_smoothing.median_filter(self.image1)
            cv2.imshow("Median Filtered Image", median)
        else:
            print("No image loaded for Median Filter")

    # 邊緣檢測函數
    def run_sobel_x(self):
        """執行 Sobel X 邊緣檢測"""
        if self.image1 is not None:
            # 保存結果到類屬性
            self.sobel_x_result = edge_detection.sobel_x(self.image1)
            if self.sobel_x_result is not None:
                cv2.imshow("Sobel X Result", self.sobel_x_result)
        else:
            print("No image loaded for Sobel X")

    def run_sobel_y(self):
        """執行 Sobel Y 邊緣檢測"""
        if self.image1 is not None:
            # 保存結果到類屬性
            self.sobel_y_result = edge_detection.sobel_y(self.image1)
            if self.sobel_y_result is not None:
                cv2.imshow("Sobel Y Result", self.sobel_y_result)
        else:
            print("No image loaded for Sobel Y")

    def run_combination_and_threshold(self):
        """執行 Sobel 結果的組合和閾值化"""
        if self.image1 is not None:
            # 確保有 Sobel X 和 Y 的結果
            if self.sobel_x_result is None:
                self.sobel_x_result = edge_detection.sobel_x(self.image1)
            if self.sobel_y_result is None:
                self.sobel_y_result = edge_detection.sobel_y(self.image1)
                
            # 使用保存的結果執行組合和閾值化
            combined = edge_detection.combine_and_threshold(self.sobel_x_result, self.sobel_y_result)
            if combined is not None:
                cv2.imshow("Combined and Thresholded Edge", combined)
        else:
            print("No image loaded for Combination and Threshold")

    def run_gradient_angle(self):
        """執行梯度角度計算並應用遮罩"""
        if self.image1 is not None:
            # 確保有 Sobel X 和 Y 結果
            if self.sobel_x_result is None:
                self.sobel_x_result = edge_detection.sobel_x(self.image1)
            if self.sobel_y_result is None:
                self.sobel_y_result = edge_detection.sobel_y(self.image1)

            # 呼叫 gradient_angle 並獲取結果
            gradient_angle_image = edge_detection.gradient_angle(self.sobel_x_result, self.sobel_y_result)
            if gradient_angle_image is not None:
                cv2.imshow("Gradient Angle", gradient_angle_image)
        else:
            print("No image loaded for Gradient Angle")

    # 變換函數
    def run_transform(self):
        """
        從輸入欄位獲取參數，並應用轉換。
        """
        if self.image1 is None:
            print("No image loaded for transformation")
            return

        try:
            # 獲取並解析輸入參數
            angle_text = self.transform_inputs["Rotation:"].text()
            scale_text = self.transform_inputs["Scaling:"].text()
            tx_text = self.transform_inputs["Tx:"].text()
            ty_text = self.transform_inputs["Ty:"].text()

            angle = float(angle_text) if angle_text else 0.0
            scale = float(scale_text) if scale_text else 1.0
            tx = int(tx_text) if tx_text else 0
            ty = int(ty_text) if ty_text else 0

            # 應用轉換
            transformed_image = transforms.apply_transform(self.image1, angle=angle, scale=scale, tx=tx, ty=ty)

            # 顯示轉換後的影像
            cv2.imshow("Transformed Image", transformed_image)
            print("Transformation applied successfully")

        except ValueError as ve:
            print(f"Invalid input: {ve}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessingGUI()
    ex.show()
    sys.exit(app.exec_())