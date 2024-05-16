import numpy as np
import cv2
import tensorflow as tf
import time
from shapely.geometry import Polygon, Point
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtWidgets import QDesktopWidget

from Mod_Constants import BBOXSIZE, PRINTS_DEBUG, CUE_DETECTION, MAX_BOUNCES, CIRC_MARGIN, MIN_CIRC_RADIUS, BUFFER_SIZE
import Mod_ArUco
import Mod_Bbox
import Mod_Preprocess
import Mod_CueDetect
from Mod_CircBuff import CircularBuffer

def run_tracking_module(canny_threshold1):
    print("run_tracking_module called with canny_threshold1:", canny_threshold1)
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
    # start_timer_markers = time.time()
    # both_marker_found = False

    rect_in_frame: bool = [False, False]

    frame_width = 1920
    frame_height = 1080

    # === FOR WEBCAM ===
    print("Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # # Run HP Webcam software to autofocus
    focus = 1
    cap.set(28, focus)

    # path = "Dependencies\RealPool_Cut2.mp4"
    # cap = cv2.VideoCapture(path)

    interpreter = tf.lite.Interpreter(model_path="Dependencies\V5_FOMO_FLOAT.lite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print("Output: ", output_details)

    current_middle_points = np.array([])
    last_middle_points = np.array([])
    cue_polygon = np.array([[0,0],[0,0],[0,0],[0,0]])
    # prev_count_Balls = [0, 0]
    # score = [0, 0]

    # Set interval for ArUco marker detection (in seconds)
    aruco_detection_interval = 5  
    stop_time = 1
    last_detection_time = time.time()
    working = True

    ball_buffer = CircularBuffer(BUFFER_SIZE)

    while True:
        ret, clean_img = cap.read()
        timer = cv2.getTickCount()
        img = clean_img.copy()

        if ret is False:
            print("Error: Unable to capture frame.")
            cv2.destroyAllWindows()
            exit()

        if clean_img is None:
            print("Error: Unable to load the image.")
            cv2.destroyAllWindows()
            exit()
        
        hue = 10
        contrast = 1
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] += hue
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

        circ_img = clean_img.copy()

        # mask_stripedBalls = np.zeros_like(clean_img)
        projection_MASK = np.zeros_like(clean_img)

        # Preprocess image + get contours from model
        contours, output_heatmap = Mod_Preprocess.processImg(img, interpreter, input_details, output_details)

        # Calculate centers of contours
        centers = Mod_Bbox.calc_centers(contours)

        # Draw bounding boxes
        bbox_coor = Mod_Bbox.calc_bboxes(centers, last_middle_points, BBOXSIZE, img, circ_img, projection_MASK, drawing=True)

        polygon = Polygon(last_middle_points)
        for bbox in bbox_coor:
            xbox1, ybox1, xbox2, ybox2 = bbox
            roi = clean_img[ybox1:ybox2, xbox1:xbox2]

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] += 9

            roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            roi = cv2.convertScaleAbs(roi, alpha=1, beta=0)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY) 

            circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=40, param2=20, minRadius=5, maxRadius=30)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    center = (circle[0], circle[1])
                    radius = circle[2]
                if polygon.contains(Point(center[0] + xbox1, center[1] + ybox1)):
                    if not ball_buffer.exists_within_margin(center, CIRC_MARGIN): 
                        if radius >= MIN_CIRC_RADIUS:
                            adjusted_center = (center[0] + xbox1, center[1] + ybox1)
                            
                            cv2.circle(img, adjusted_center, radius + 10, (0, 255, 0), 2) 
                            cv2.circle(projection_MASK, adjusted_center, radius + 10, (255, 255, 255), 2)  
                            circ_img = cv2.circle(circ_img, adjusted_center, radius + 10, (255, 255, 255), 2)  
                            ball_buffer.add(adjusted_center)

        # Perform ArUco marker detection periodically
        current_time = time.time()
        
        if current_time - last_detection_time >= aruco_detection_interval or working:
            if not working:
                last_detection_time = current_time
                working = True
            if current_time - last_detection_time >= stop_time:
                working = False

            detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
            corners, ids, rejected = detector.detectMarkers(img)
            current_middle_points, marker_centers, cue_polygon_test = Mod_ArUco.aruco_display(corners, ids, rejected, img, circ_img, rect_in_frame)
            if len(marker_centers) > 7:
                working = False
            if len(cue_polygon_test) == 4:
                cue_polygon = cue_polygon_test

        if current_middle_points is not None and len(current_middle_points) > 0:
            last_middle_points = current_middle_points

        if last_middle_points is not None and len(last_middle_points) > 0:
            cv2.polylines(circ_img, [last_middle_points], True, (255, 0, 255), 5)
            cv2.polylines(projection_MASK, [last_middle_points], True, (255, 0, 255), 5)

        if CUE_DETECTION:
            print("Cue polygon", cue_polygon) if PRINTS_DEBUG else None
            x, y, w, h = cv2.boundingRect(cue_polygon)
            x1_roi, y1_roi, x2_roi, y2_roi = x, y, x + w, y + h
            cv2.polylines(circ_img, [cue_polygon], isClosed=True, color=(255, 0, 255), thickness=2)
            cv2.rectangle(circ_img, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 255, 255), 2)

            if x2_roi > x1_roi and y2_roi > y1_roi:  # Check if the rectangle has valid coordinates
                hsv = cv2.cvtColor(clean_img, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] += 9
                hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                hsv = cv2.convertScaleAbs(hsv, alpha=1, beta=0)
                roi = hsv[y1_roi:y2_roi, x1_roi:x2_roi]
                if not roi.size == 0:  # Check if the ROI is not empty
                    x_offset, y_offset = x1_roi, y1_roi  # Offset used for extracting ROI
                    imgBlur = cv2.GaussianBlur(roi, (7, 7), 1)
                    roi_gray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

                    #threshold1 = 500
                    threshold2 = 0
                    minLineLength = 50
                    imgCanny = cv2.Canny(roi_gray, canny_threshold1, threshold2)
                    kernel = np.ones((5, 5))
                    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

                    # mask_cue = np.zeros_like(roi)
                    img_cueLines = roi.copy()
                    cue_x1, cue_y1, cue_x2, cue_y2 = Mod_CueDetect.findCue(imgDil, minLineLength, img_cueLines)

                    if cue_x1 is not None and cue_y1 is not None and cue_x2 is not None and cue_y2 is not None:
                        cue_x1_orig, cue_y1_orig = cue_x1 + x_offset, cue_y1 + y_offset
                        cue_x2_orig, cue_y2_orig = cue_x2 + x_offset, cue_y2 + y_offset
                        cv2.line(img_cueLines, (cue_x1_orig, cue_y1_orig), (cue_x2_orig, cue_y2_orig), (0, 255, 0), 2)
                        cv2.line(circ_img, (cue_x1_orig, cue_y1_orig), (cue_x2_orig, cue_y2_orig), (0, 255, 0), 2)
                        cv2.line(projection_MASK, (cue_x1_orig, cue_y1_orig), (cue_x2_orig, cue_y2_orig), (0, 255, 0), 2)

                        cueTip_mask = np.zeros_like(clean_img)
                        tape_margin = 50
                        cv2.line(cueTip_mask, (cue_x1_orig, cue_y1_orig), (cue_x2_orig, cue_y2_orig), (255, 255, 255), tape_margin, cv2.LINE_AA)
                        cueTip_mask = cv2.bitwise_and(clean_img, cueTip_mask)
                        cueTip_mask = cv2.cvtColor(cueTip_mask, cv2.COLOR_BGR2HSV)

                        cue_x1_orig, cue_y1_orig, cue_x2_orig, cue_y2_orig, tape_mask = Mod_CueDetect.find_tipCue_tape(cueTip_mask, cue_x1_orig, cue_y1_orig, cue_x2_orig, cue_y2_orig, circ_img, clean_img)
                        cv2.circle(circ_img, (cue_x1_orig, cue_y1_orig), 5, (0, 255, 255), cv2.FILLED)
                        cv2.circle(circ_img, (cue_x2_orig, cue_y2_orig), 5, (255, 0, 0), cv2.FILLED)

                        cue_sec_coor = np.array([cue_x2_orig, cue_y2_orig])
                        cue_angle = np.degrees(np.arctan2(cue_y2_orig - cue_y1_orig, cue_x2_orig - cue_x1_orig))
                        # cue_length = np.sqrt((cue_x2_orig - cue_x1_orig) ** 2 + (cue_y2_orig - cue_y1_orig) ** 2)
                        Mod_CueDetect.draw_trajectory(img, circ_img, projection_MASK, cue_sec_coor, cue_polygon, cue_angle, MAX_BOUNCES)

                        cv2.imshow("CueTip Mask", cueTip_mask)
                        cv2.imshow("Tape Mask", tape_mask)

                    cv2.imshow("Cue img Dilate", imgDil)
                    cv2.imshow("Cue lines", img_cueLines)
        
        cv2.imshow('Original', img)
        screen_geometry = QDesktopWidget().screenGeometry(0)
        cv2.resizeWindow('Original', screen_geometry.width(), screen_geometry.height())

        # cv2 window settings
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(img, f"FPS: {int(fps)}", (frame_width-220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(circ_img, f"FPS: {int(fps)}", (screen_geometry.width()-220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        heatmap = cv2.applyColorMap(np.uint8(255 * output_heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

        focus = 1
        cap.set(28, focus)
        
        img = cv2.resize(img, (1280, 720))
        circ_img = cv2.resize(circ_img, (1920, 1080))

        # Scale all images to 480p
        overlay = cv2.resize(overlay, (640, 480))
        projection_MASK = cv2.resize(projection_MASK, (1920, 1080))
        cv2.imshow('Projection Mask', projection_MASK)
        cv2.imshow('Circles', circ_img)

        heatmap = cv2.applyColorMap(np.uint8(255 * output_heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (1920, 1080))
        cv2.imshow('Heatmap', heatmap)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    exit()
