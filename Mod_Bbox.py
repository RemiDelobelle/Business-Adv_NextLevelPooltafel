import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from concurrent.futures import ThreadPoolExecutor
from Mod_Constants import PRINTS, PRINTS_DEBUG, THRESHOLD_STRIPED_BALLS, MARGIN_STRIPED_BALLS

# def process_bbox(bbox, clean_img, mask_stripedBalls, mask_solidBalls, projection_MASK, coor_stripedBalls, coor_solidBalls):
#     xbox1, ybox1, xbox2, ybox2 = bbox
#     xbox1 = max(0, xbox1)
#     ybox1 = max(0, ybox1)
#     xbox2 = max(0, xbox2)
#     ybox2 = max(0, ybox2)

#     roi = clean_img[ybox1:ybox2, xbox1:xbox2]
    
#     if roi is not None:
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
#         white_pixel_count = cv2.countNonZero(binary)
#         total_pixel_count = roi.size
#         white_percentage = ((white_pixel_count / total_pixel_count) * 100) + MARGIN_STRIPED_BALLS
#         print(f"White percentage: {white_percentage}%") if PRINTS else None
        
#         mask = np.zeros_like(clean_img)
#         cv2.rectangle(mask, (xbox1, ybox1), (xbox2, ybox2), (255, 255, 255), -1)
#         cv2.rectangle(projection_MASK, (xbox1, ybox1), (xbox2, ybox2), (255, 255, 255), 2)
        
#         if white_percentage > THRESHOLD_STRIPED_BALLS:
#             mask_stripedBalls = cv2.bitwise_and(mask, clean_img)
#             coor_stripedBalls.append([xbox1, ybox1, xbox2, ybox2])
#         else:
#             mask_solidBalls = cv2.bitwise_and(mask, clean_img)
#             cv2.rectangle(projection_MASK, (xbox1, ybox1), (xbox2, ybox2), (0, 255, 255), 2)
#             coor_solidBalls.append([xbox1, ybox1, xbox2, ybox2])

#     return mask_stripedBalls, mask_solidBalls, projection_MASK, coor_stripedBalls, coor_solidBalls


def process_bbox(bbox, clean_img, mask_stripedBalls, mask_solidBalls, projection_MASK, coor_stripedBalls, coor_solidBalls):
    print(f"[DEBUG] Processing bbox: {bbox}") if PRINTS_DEBUG else None
    print(f"[DEBUG] Clean image: {type(clean_img)}, {clean_img}") if PRINTS_DEBUG else None

    xbox1, ybox1, xbox2, ybox2 = bbox
    xbox1 = max(0, xbox1)
    ybox1 = max(0, ybox1)
    xbox2 = max(0, xbox2)
    ybox2 = max(0, ybox2)

    print(f"[DEBUG] Xbox1: {xbox1}, Ybox1: {ybox1}, Xbox2: {xbox2}, Ybox2: {ybox2}") if PRINTS_DEBUG else None
    roi = clean_img[ybox1:ybox2, xbox1:xbox2]
    print(f"[DEBUG] ROI: {roi.shape}, {roi}") if PRINTS_DEBUG else None
    if roi is not None:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_pixel_count = cv2.countNonZero(binary)
        total_pixel_count = roi.shape[0] * roi.shape[1]
        white_percentage = round(((white_pixel_count / total_pixel_count) * 100 + MARGIN_STRIPED_BALLS), 2)
        print(f"White percentage: {white_percentage}%") if PRINTS else None
        if white_percentage > THRESHOLD_STRIPED_BALLS:
            mask = np.zeros_like(clean_img)
            clean_img_bboxes = clean_img.copy()
            mask_stripedBalls_img = cv2.rectangle(mask, (xbox1, ybox1), (xbox2, ybox2), (255, 255, 255), -1)
            mask_stripedBalls_img = cv2.bitwise_and(mask_stripedBalls_img, clean_img_bboxes)
            mask_stripedBalls = cv2.addWeighted(mask_stripedBalls, 1, mask_stripedBalls_img, 1, 0)
            cv2.rectangle(projection_MASK, (xbox1, ybox1), (xbox2, ybox2), (255, 255, 255), 2)
            coor_stripedBalls.append([xbox1, ybox1, xbox2, ybox2])
            
        if white_percentage <= THRESHOLD_STRIPED_BALLS:
            mask = np.zeros_like(clean_img)
            clean_img_bboxes = clean_img.copy()
            mask_solidBalls_img = cv2.rectangle(mask, (xbox1, ybox1), (xbox2, ybox2), (255, 255, 255), -1)
            mask_solidBalls_img = cv2.bitwise_and(mask_solidBalls_img, clean_img_bboxes)
            mask_stripedBalls = cv2.addWeighted(mask_solidBalls_img, 1, mask_solidBalls_img, 1, 0)
            cv2.rectangle(projection_MASK, (xbox1, ybox1), (xbox2, ybox2), (0, 255, 255), 2)
            coor_solidBalls.append([xbox1, ybox1, xbox2, ybox2])

    return mask_stripedBalls, mask_solidBalls, projection_MASK, coor_stripedBalls, coor_solidBalls

def process_bboxes(bbox_coor, clean_img, mask_stripedBalls, mask_solidBalls, projection_MASK, coor_stripedBalls, coor_solidBalls):
    with ThreadPoolExecutor() as executor:
        for bbox_args in bbox_coor:
            future = executor.submit(process_bbox, bbox_args, clean_img, mask_stripedBalls, mask_solidBalls, projection_MASK, coor_stripedBalls, coor_solidBalls)
            mask_stripedBalls, mask_solidBalls, projection_MASK, coor_stripedBalls, coor_solidBalls = future.result()

    return mask_stripedBalls, mask_solidBalls, projection_MASK, coor_stripedBalls, coor_solidBalls


def calc_centers(contours):
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            centers.append(center)
        else:
            centers.append(None)
    return centers


def calc_bboxes(centers, last_middle_points, box_size, img, projection_MASK, drawing = True):
    bbox_coor = []
    polygon = Polygon(last_middle_points)
    for center in centers:
        if center is not None:
            point = Point(center[0], center[1])
            if polygon.contains(point):
                cv2.circle(projection_MASK, center, 10, (0,0,255), 3) if drawing else None

                xbox1 = center[0] - box_size
                ybox1 = center[1] - box_size
                xbox2 = center[0] + box_size
                ybox2 = center[1] + box_size
                bbox_coor.append([xbox1, ybox1, xbox2, ybox2])
                cv2.rectangle(img, (xbox1, ybox1), (xbox2, ybox2), (0, 255, 0), 2) if drawing else None
    return bbox_coor