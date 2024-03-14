# Striped balls detection with ArUco markers

import numpy as np
import cv2
import tensorflow as tf
import time
from shapely.geometry import Polygon, Point
from concurrent.futures import ThreadPoolExecutor

from Mod_Constants import THRESHOLD_STRIPED_BALLS, MARGIN_STRIPED_BALLS, THRESHOLD_CONTOUR, BBOXSIZE
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
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

interpreter = tf.lite.Interpreter(model_path="Dependencies/V2_Float.lite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print("Output: ", output_details)

last_middle_points = np.array([])


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
    print(f"Bounding boxes: {bbox_coor}") if len(bbox_coor) > 0 else None
    mask_stripedBalls = np.zeros_like(clean_img)
    mask_stripedBalls, projection_MASK = Mod_Bbox.process_bboxes(bbox_coor, clean_img, mask_stripedBalls, projection_MASK)

    # ArUco detection
    corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    current_middle_points = Mod_ArUco.aruco_display(corners, ids, rejected, img, rect_in_frame)
    if len(current_middle_points) > 0:
        last_middle_points = current_middle_points
    if len(last_middle_points) > 0:
        cv2.polylines(img, [last_middle_points], True, (255, 0, 255), 5)
        cv2.polylines(projection_MASK, [last_middle_points], True, (255, 0, 255), 5)
        cv2.polylines(mask_stripedBalls, [last_middle_points], True, (255, 0, 255), 5)
        min_x = np.min(last_middle_points[:, 0])
        min_y = np.min(last_middle_points[:, 1])
        max_x = np.max(last_middle_points[:, 0])
        max_y = np.max(last_middle_points[:, 1])
        warped_img = clean_img[min_y:max_y, min_x:max_x]

    # cv2 window settings
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, f"FPS: {int(fps)}", (frame_width-320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    heatmap = cv2.applyColorMap(np.uint8(255 * output_heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    # Display windows
    cv2.imshow('Original', img)
    # cv2.imshow('Heatmap', heatmap)
    # cv2.imshow('Overlay', overlay)
    cv2.imshow('Original boxes', projection_MASK)
    cv2.imshow('Striped balls', mask_stripedBalls)
    cv2.imshow("Warped Image", warped_img) if len(last_middle_points) > 0 else None
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
exit()
