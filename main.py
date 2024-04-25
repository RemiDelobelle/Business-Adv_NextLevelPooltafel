import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
import cv2
from mainwindow_ui import Ui_MainWindow
from concurrent.futures import ThreadPoolExecutor
import mainProgram

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # slider
        self.ui.canny_thres_slider.valueChanged.connect(self.slider_changed)
        self.ui.run_camera_feed_btn.clicked.connect(self.run_camera_feed)
        self.ui.run_main_btn.clicked.connect(self.run_main)
        
        # Create a thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Open the camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Failed to open camera")
        
        self.label = self.ui.label_Camera
        initial_value = self.ui.canny_thres_slider.value()
        self.ui.canny_thres_label.setText(f"canny_thres: {initial_value}")

    # Slider reading
    def slider_changed(self, value):
        self.ui.canny_thres_label.setText(f"canny_thres: {value}")
        # Submit the tracking function to the executor
        self.executor.submit(self.run_camera_feed)

    def run_camera_feed(self):
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break

            # Image processing
            imgBlur = cv2.GaussianBlur(frame, (7, 7), 1)
            imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
            imgCanny = cv2.Canny(imgGray, 100, 100)
            kernel = np.ones((5, 5), np.uint8)
            imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
            
            rgb_image = cv2.cvtColor(imgDil, cv2.COLOR_GRAY2RGB)
            
            # Convert to QImage
            h, w, c = rgb_image.shape
            q_img = QImage(rgb_image.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Set the pixmap to the QLabel
            self.ui.label_Camera.setPixmap(pixmap)
            
            # Resize QLabel to fit the image
            self.ui.label_Camera.setScaledContents(True)
            
            # Update the UI
            QtWidgets.QApplication.processEvents()
            
            # Break the loop if the window is closed
            if not self.isVisible():
                break

    def run_main(self):
        self.executor.submit(mainProgram.run_tracking_module(10))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
