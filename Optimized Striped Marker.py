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
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
# cap.set(cv2.CAP_PROP_FPS, 60)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Run HP Webcam software to autofocus
# focus = 1
# cap.set(28, focus)

path = "AiTracking_Module_Video\Dependencies\RealPool_Cutted2.mp4"
cap = cv2.VideoCapture(path)

interpreter = tf.lite.Interpreter(model_path="AiTracking_Module_Video\Dependencies\V5_FOMO_FLOAT.lite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print("Output: ", output_details)

last_middle_points = np.array([])
prev_count_Balls = [0, 0]
score = [0, 0]




# --------------------------------------------------------------------------------------------------------


# Calculate scaling factor
scaling_factor = 0.5  # Adjust this value to fit the window within your screen's resolution

# Set window names
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Original boxes', cv2.WINDOW_NORMAL)
cv2.namedWindow('Striped balls', cv2.WINDOW_NORMAL)
cv2.namedWindow('Solid balls', cv2.WINDOW_NORMAL)
cv2.namedWindow('Heatmap', cv2.WINDOW_NORMAL)

# Set window sizes
window_width = int(frame_width * scaling_factor)
window_height = int(frame_height * scaling_factor)
cv2.resizeWindow('Original', window_width, window_height)
cv2.resizeWindow('Original boxes', window_width, window_height)
cv2.resizeWindow('Striped balls', window_width, window_height)
cv2.resizeWindow('Solid balls', window_width, window_height)
cv2.resizeWindow('Heatmap', window_width, window_height)


#--------------------------------------------------------------------------------------------------------




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
    print(f"[DEBUG] Bounding boxes: {bbox_coor}") if len(bbox_coor) and PRINTS_DEBUG > 0 else None
    mask_stripedBalls = np.zeros_like(clean_img)
    mask_solidBalls = np.zeros_like(clean_img)
    coor_stripedBalls, coor_solidBalls = [], []
    mask_stripedBalls, mask_solidBalls, projection_MASK, coor_stripedBalls, coor_solidBalls = Mod_Bbox.process_bboxes(bbox_coor, clean_img, mask_stripedBalls, mask_solidBalls, projection_MASK, coor_stripedBalls, coor_solidBalls)
    curr_count_Balls = [len(coor_stripedBalls), len(coor_solidBalls)]   # TODO: score
    if (curr_count_Balls != prev_count_Balls) and prev_count_Balls[0] is not None and prev_count_Balls[1] is not None:
        score[0] = curr_count_Balls[0] - prev_count_Balls[0]
        score[1] = curr_count_Balls[1] - prev_count_Balls[1]
        prev_count_Balls = curr_count_Balls
    cv2.putText(img, f"#Striped balls: {curr_count_Balls[0]}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(img, f"#Solid balls: {curr_count_Balls[1]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(img, f"Score: I:{score[0]} - O:{score[1]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # ArUco detection
    corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    current_middle_points = Mod_ArUco.aruco_display(corners, ids, rejected, img, rect_in_frame)
    if len(current_middle_points) > 0:
        last_middle_points = current_middle_points
    if len(last_middle_points) > 0:
        cv2.polylines(img, [last_middle_points], True, (255, 0, 255), 5)
        cv2.polylines(projection_MASK, [last_middle_points], True, (255, 0, 255), 5)
        cv2.polylines(mask_stripedBalls, [last_middle_points], True, (255, 0, 255), 5)

    # cv2 window settings
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, f"FPS: {int(fps)}", (frame_width-220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    heatmap = cv2.applyColorMap(np.uint8(255 * output_heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    # Display windows
    focus = 1
    cap.set(28, focus)
    
    cv2.imshow('Original', img)
    # cv2.imshow('Heatmap', heatmap)
    # cv2.imshow('Overlay', overlay)
    cv2.imshow('Original boxes', projection_MASK)
    cv2.imshow('Striped balls', mask_stripedBalls)
    cv2.imshow('Solid balls', mask_solidBalls)
    heatmap = cv2.applyColorMap(np.uint8(255 * output_heatmap), cv2.COLORMAP_JET)
    cv2.imshow('Heatmap', heatmap)



    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
exit()
