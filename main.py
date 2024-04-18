import cv2
import numpy as np
import sys
import threading
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from mainwindow_ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt
from mainProgram import run_tracking_module

import numpy as np
import cv2
import tensorflow as tf
from shapely.geometry import Polygon, Point
from concurrent.futures import ThreadPoolExecutor

class CV2Thread(threading.Thread):
    def __init__(self, main_window):
        threading.Thread.__init__(self)
        self.main_window = main_window

    def run(self):
        self.main_window.run_cv2()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.btn.clicked.connect(self.run)
        self.setup_second_window()
        
        # slider
        self.ui.Slider.valueChanged.connect(self.slider_changed)
        screen = QDesktopWidget().screenGeometry(1)
        self.setGeometry(screen)
        self.showMaximized()
        
    def setup_second_window(self):
        self.second_window = QtWidgets.QWidget()
        self.second_layout = QVBoxLayout()
        self.second_label = QLabel(self.second_window)
        self.second_layout.addWidget(self.second_label)
        self.second_window.setLayout(self.second_layout)
        self.second_window.setWindowTitle("CV2 Window")
        

        primary_screen_geometry = QDesktopWidget().screenGeometry(0)
        self.second_window.setGeometry(primary_screen_geometry)
        
        self.second_window.showFullScreen()


    # slider uitlezen   
    def slider_changed(self):
        value = self.ui.Slider.value()
        self.ui.label.setText(f"Value: {value}")
    
    # Tabbladen instellen
    def setup_standard_tab(self):
        layout = QtWidgets.QVBoxLayout(self.standard_tab)
        label = QtWidgets.QLabel("Standard Mode Tab Content")
        layout.addWidget(label)

    def setup_developer_tab(self):
        layout = QtWidgets.QVBoxLayout(self.developer_tab)
        label = QtWidgets.QLabel("Developer Mode Tab Content")
        layout.addWidget(label)

    # Run-methode voor detectieprogramma
    def run(self):
        run_tracking_module()
                
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())