# aruco_module.py
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from Mod_Constants import PRINTS, PRINTS_DEBUG, MARGIN_POLYGON

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def aruco_display(corners, ids, rejected, img, rect_in_frame):
    middle_points = np.array([])
    if len(corners) > 0:
        ids = ids.flatten()
        marker_centers = []

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each point to coordinate
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            
            # draw lines connecting the corners of the marker
            cv2.line(img, topLeft, topRight, (0, 255, 255), 2)
            cv2.line(img, topRight, bottomRight, (0, 255, 255), 2)
            cv2.line(img, bottomRight, bottomLeft, (0, 255, 255), 2)
            cv2.line(img, bottomLeft, topLeft, (0, 255, 255), 2)

            # draw a circle at the center of the marker            
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            marker_centers.append((markerID,(cX, cY)))
            print(f"Marker coor of ID ({markerID}): ({cX}, {cY})") if PRINTS else None
            cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)
            
            cv2.putText(img, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # print("ArUco marker ID: {}".format(markerID)) if PRINTS else None

        # Display count of detected markers
        cv2.putText(img, f"#Detected markers: {len(marker_centers)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        print("IDs:", ids) if PRINTS else None
        print("[DEBUG] Marker centers:", marker_centers) if PRINTS_DEBUG else None
        # print("IDs:", ids, type(ids))

        # Draw rectangles
        values1 = [0, 1, 2, 3]
        rect_in_frame[0] = draw_rectangle_markers(img, ids, marker_centers, values1)     
        values2 = [4, 5, 6, 7]
        rect_in_frame[1] = draw_rectangle_markers(img, ids, marker_centers, values2)

        if rect_in_frame[0] and rect_in_frame[1]:
            print("[INFO] Both rectangles are in frame!") if PRINTS else None

            middle_cX1, middle_cY1 = None, None
            middle_cX2, middle_cY2 = None, None
            middle_cX3, middle_cY3 = None, None
            middle_cX4, middle_cY4 = None, None
            # middle_points = np.array([])

            for id, (cX, cY) in marker_centers:
                if id == 4:
                    middle_cY1 = cY + MARGIN_POLYGON
                elif id == 0:
                    middle_cX1 = cX + MARGIN_POLYGON
                elif id == 5:
                    middle_cY2 = cY + MARGIN_POLYGON
                elif id == 1:
                    middle_cX2 = cX - MARGIN_POLYGON
                elif id == 6:
                    middle_cY3 = cY - MARGIN_POLYGON
                elif id == 2:
                    middle_cX3 = cX - MARGIN_POLYGON
                elif id == 7:
                    middle_cY4 = cY - MARGIN_POLYGON
                elif id == 3:
                    middle_cX4 = cX + MARGIN_POLYGON
            
            middle_points = np.array([[middle_cX1, middle_cY1], [middle_cX2, middle_cY2], [middle_cX3, middle_cY3], [middle_cX4, middle_cY4]])
            print(f"Corners coor playfield: {middle_points}") if PRINTS else None
            if marker_centers is not None:
                return middle_points, marker_centers

def draw_rectangle_markers(img, ids, marker_centers, values):
    mask = np.isin(values, [x[0] for x in marker_centers])
    
    if np.all(mask):
        print("[INFO] All values present, drawing rectangle...") if PRINTS else None
        min_value = min(values)
        ordered_centers = np.empty((len(values), 2), dtype=np.int32)
        
        for i, marker_id in enumerate(ids):
            if marker_id in values:
                corner_index = marker_id - min_value
                ordered_centers[corner_index] = marker_centers[i][1]

        cv2.polylines(img, [ordered_centers], True, (0, 255, 0), 2)

        return True
    else:
        return False