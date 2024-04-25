import numpy as np
import cv2
import tensorflow as tf
import time
from shapely.geometry import Polygon, Point
from concurrent.futures import ThreadPoolExecutor

from Mod_Constants import BBOXSIZE, PRINTS, PRINTS_DEBUG
import Mod_ArUco
import Mod_Bbox
import Mod_Preprocess

# Allow memory growth on all GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


aruco_type = "DICT_4X4_50"
arucoDict = cv2.aruco.getPredefinedDictionary(Mod_ArUco.ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters()
start_timer_markers = time.time()
both_marker_found = False

rect_in_frame: bool = [False, False]

frame_width = 1920
frame_height = 1080

# === FOR WEBCAM ===
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
# cap.set(cv2.CAP_PROP_FPS, 60)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# #Run HP Webcam software to autofocus
# focus = 1
# cap.set(28, focus)

path = "Dependencies/RealPool_Cut2.mp4"
cap = cv2.VideoCapture(path)

# interpreter = tf.lite.Interpreter(model_path="Dependencies/V5_FOMO_FLOAT.lite")
interpreter = tf.lite.Interpreter(model_path="Dependencies/V5_FOMO_FLOAT.lite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print("Output: ", output_details)

current_middle_points = np.array([])
last_middle_points = np.array([])
prev_count_Balls = [0, 0]
score = [0, 0]

# Set interval for ArUco marker detection (in seconds)
aruco_detection_interval = 5  
stop_time = 1
last_detection_time = time.time()
working = True

while True:
    ret, clean_img = cap.read()
    timer = cv2.getTickCount()
    img = clean_img.copy()

    if clean_img is None:
        print("Error: Unable to load the image.")
        cv2.destroyAllWindows()
        exit()

    img = clean_img.copy()
    mask_stripedBalls = np.zeros_like(clean_img)
    projection_MASK = np.zeros_like(clean_img)

    # Preprocess image + get contours from model
    contours, output_heatmap = Mod_Preprocess.processImg(img, interpreter, input_details, output_details)

    # Calculate centers of contours
    centers = Mod_Bbox.calc_centers(contours)

    # Draw bounding boxes
    bbox_coor = Mod_Bbox.calc_bboxes(centers, last_middle_points, BBOXSIZE, img, projection_MASK, drawing=True)

    # ==========================================================================
    circ_img = clean_img.copy()

    # Define the minimum size of circle (minimum radius)
    min_circle_radius = 10  # Adjust this value as needed

    # Detect circles within bounding boxes
    for bbox in bbox_coor:
        xbox1, ybox1, xbox2, ybox2 = bbox
        roi = clean_img[ybox1:ybox2, xbox1:xbox2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY)  # Adjust the threshold value as needed

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=30)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                
                if radius >= min_circle_radius:
                    # Adjust circle center coordinates based on bounding box coordinates
                    adjusted_center = (center[0] + xbox1, center[1] + ybox1)
                    
                    cv2.circle(img, adjusted_center, radius, (0, 255, 0), 2) 
                    cv2.circle(projection_MASK, adjusted_center, radius, (0, 255, 0), 2)  
                    circ_img = cv2.circle(circ_img, adjusted_center, radius + 10, (255, 255, 255), 2)  

    # ==========================================================================

    # Perform ArUco marker detection periodically
    current_time = time.time()
    
    if current_time - last_detection_time >= aruco_detection_interval or working:
        if not working:
            last_detection_time = current_time
            working = True
        if current_time - last_detection_time >= stop_time:
            working = False

        # ArUco detection
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        corners, ids, rejected = detector.detectMarkers(img)
        result = Mod_ArUco.aruco_display(corners, ids, rejected, img, rect_in_frame)
        if result is not None:
            current_middle_points, marker_centers = result
            if len(marker_centers) > 7 :
                working = False

    if current_middle_points is not None and len(current_middle_points) > 0:
        last_middle_points = current_middle_points

    if last_middle_points is not None and len(last_middle_points) > 0:
        cv2.polylines(img, [last_middle_points], True, (255, 0, 255), 5)
        cv2.polylines(projection_MASK, [last_middle_points], True, (255, 0, 255), 5)
        # # PROBLEM: comment this line if you want to test fps problem
        cv2.polylines(mask_stripedBalls, [last_middle_points], True, (255, 0, 255), 5)


    # cv2 window settings
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, f"FPS: {int(fps)}", (frame_width-220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    heatmap = cv2.applyColorMap(np.uint8(255 * output_heatmap), cv2.COLORMAP_JET)
    # overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    # Display windows
    focus = 1
    cap.set(28, focus)

    # Scale all images to 480p
    img = cv2.resize(img, (640, 480))
    circ_img = cv2.resize(circ_img, (640, 480))
    # overlay = cv2.resize(overlay, (640, 480))
    # projection_MASK = cv2.resize(projection_MASK, (640, 480))
    
    cv2.imshow('Original', img)
    cv2.imshow('Circles', circ_img)
    # cv2.imshow('Overlay', overlay)
    # cv2.imshow('Original boxes', projection_MASK)

    heatmap = cv2.applyColorMap(np.uint8(255 * output_heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (640, 480))
    cv2.imshow('Heatmap', heatmap)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
exit()
