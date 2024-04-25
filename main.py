import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
import cv2
from mainwindow_ui import Ui_MainWindow
from concurrent.futures import ThreadPoolExecutor

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # slider
        self.ui.Slider.valueChanged.connect(self.slider_changed)

        # Button
        self.ui.btn.clicked.connect(self.run_tracking)

        # Create a thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Open the camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Failed to open camera")

        # Create a label to display the video feed
        self.label = self.ui.label_Camera

    # Slider reading
    def slider_changed(self, value):
        self.ui.label.setText(f"Value: {value}")
        # Submit the tracking function to the executor
        self.executor.submit(self.run_tracking)

    def run_tracking(self):
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            h, w, c = rgb_image.shape
            q_img = QImage(rgb_image.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.label.setPixmap(pixmap)
            
            # Resize QLabel to fit the image
            self.label.setScaledContents(True)
            
            # Update the UI
            QtWidgets.QApplication.processEvents()
            
            # Break the loop if the window is closed
            if not self.isVisible():
                break

    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
