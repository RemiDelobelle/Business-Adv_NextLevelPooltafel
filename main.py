import sys
import numpy as np
from PyQt5 import QtWidgets
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

        # Create a thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=1)
        print(">>> Created thread pool executor")
        # Connect signals and slots
        self.ui.canny_thres_slider.valueChanged.connect(self.slider_changed)
        self.ui.run_camera_feed_btn.clicked.connect(self.run_camera_feed)
        self.ui.run_main_btn.clicked.connect(self.run_main)
        print(">>> Connected signals and slots")

        # Open the camera
        try:
            self.cap = cv2.VideoCapture(1) # external camera
            if not self.cap.isOpened():
                print("Failed to open camera")
        except:
            print("[ERROR] Failed to open camera")
            exit(1)

        # Set initial slider value
        self.slider_changed(500)
        

    def slider_changed(self, value):
        self.ui.canny_thres_label.setText(f"canny_thres: {value}")
        self.canny_thres = value

    def run_camera_feed(self):
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Convert the frame to QImage for unfiltered display
            h, w, c = frame.shape
            q_img_original = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap_original = QPixmap.fromImage(q_img_original)
            
            # Set the pixmap to the QLabel for unfiltered display
            self.ui.label_Camera.setPixmap(pixmap_original)
            self.ui.label_Camera.setScaledContents(True)
            
            # Image processing
            imgBlur = cv2.GaussianBlur(frame, (7, 7), 1)
            imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
            try:
                imgCanny = cv2.Canny(imgGray, self.canny_thres, 0)
            except:
                print("Error in Canny: ", self.canny_thres, type(self.canny_thres))
            
            kernel = np.ones((5, 5), np.uint8)
            imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
            
            rgb_image = cv2.cvtColor(imgDil, cv2.COLOR_GRAY2RGB)
            
            # Convert to QImage for processed display
            h, w, c = rgb_image.shape
            q_img = QImage(rgb_image.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Set the pixmap to the QLabel for processed display
            self.ui.label_Canny.setPixmap(pixmap)
            self.ui.label_Canny.setScaledContents(True)
            
            # Update the UI
            QtWidgets.QApplication.processEvents()
            
            # Break the loop if the window is closed
            if not self.isVisible():
                break


    def run_main(self):
        try:
            self.executor.submit(mainProgram.run_tracking_module, self.canny_thres)
            print("Tried starting main program with: ", self.canny_thres)
        except:
            print("Error in running main program")     

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())