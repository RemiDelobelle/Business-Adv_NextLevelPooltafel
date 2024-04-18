import numpy as np
import sys
from PyQt5 import QtWidgets

from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtWidgets, QtCore
from mainwindow_ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
from mainProgram import run_tracking_module

import numpy as np
import cv2
import tensorflow as tf
from shapely.geometry import Polygon, Point
from concurrent.futures import ThreadPoolExecutor

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_second_window()

        # slider
        self.ui.Slider.valueChanged.connect(self.slider_changed)
        screen = QDesktopWidget().screenGeometry(0)
        self.setGeometry(screen)
        self.showMaximized()

        #Button
        self.ui.btn.clicked.connect(run_tracking_module)
    
    def setup_second_window(self):
        self.second_window = QtWidgets.QWidget()
        self.second_layout = QVBoxLayout()
        self.second_label = QLabel(self.second_window)
        self.second_layout.addWidget(self.second_label)
        self.second_window.setLayout(self.second_layout)
        self.second_window.setWindowTitle("CV2 Window")

        primary_screen_geometry = QDesktopWidget().screenGeometry(1)
        self.second_window.setGeometry(primary_screen_geometry)
        self.second_window.showFullScreen()


        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_cv_window)
        self.timer.start(30)

    def update_cv_window(self):
        run_tracking_module()

    def update_label_image(self, pixmap):
        self.second_label.setPixmap(pixmap)
        self.second_label.repaint()

    # slider uitlezen
    def slider_changed(self):
        value = self.ui.Slider.value()
        self.ui.label.setText(f"Value: {value}")

    # def run_tracking_and_display(self):
    #     run_tracking_module()
        # height, width, channel = img.shape
        # bytesPerLine = 3 * width
        # qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        # pixmap = QPixmap.fromImage(qImg)
        # self.update_label_image(pixmap)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    #window.run_tracking_and_display()  # Run tracking and display the image
    window.show()
    sys.exit(app.exec_())