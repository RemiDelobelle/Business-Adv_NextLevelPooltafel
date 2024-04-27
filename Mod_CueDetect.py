import cv2
import numpy as np
from Mod_Constants import PRINTS_DEBUG

def findCue(imgDil, min_line_length, img_with_lines):
    lines = cv2.HoughLinesP(imgDil, 1, np.pi/180, 100, minLineLength=min_line_length)
    # Find the dominant line representing the cue
    if lines is not None:
        # Initialize variables to store the most dominant line
        max_length = 0
        dominant_line = None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Check if the line length is longer than previous ones
            if length > max_length:
                max_length = length
                dominant_line = line

        if dominant_line is not None:
            # Access the first element of the dominant_line array
            dominant_line = dominant_line[0]

            # Print dominant line for investigation
            print("Dominant line:", dominant_line) if PRINTS_DEBUG else None

            # Check if the dominant line is in correct format
            if len(dominant_line) == 4:
                # Draw the dominant line
                x1, y1, x2, y2 = dominant_line
                cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 3)
                return (x1, y1, x2, y2)
            else:
                print("Dominant line format is not as expected.") if PRINTS_DEBUG else None
                return None, None, None, None
        else:
            print("No dominant line found.") if PRINTS_DEBUG else None
            return None, None, None, None
    else:
        print("No lines found.") if PRINTS_DEBUG else None
        return None, None, None, None



def draw_trajectory(window, position, polygon, angle, max_bounces):
    # Define the initial direction vector based on the given angle
    direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

    # Define the initial position of the red line
    prev_position = position.astype(np.int32)

    # Initialize bounce count
    bounce_count = 0

    while bounce_count < max_bounces:
        # Calculate the next position by adding the direction vector to the current position
        next_position = position + direction

        # Check if the next position is inside the polygon
        if cv2.pointPolygonTest(polygon, (next_position[0], next_position[1]), False) >= 0:
            # Update the position to the next position
            position = next_position

            # Draw the red line from the previous position to the current position
            cv2.line(window, tuple(prev_position), tuple(position.astype(np.int32)), (0, 0, 255), 2)

            # Update the previous position
            prev_position = position.astype(np.int32)
        else:
            # Find the edge of the polygon that the position is closest to
            _, edge = find_closest_edge(polygon, position)

            # Reflect the direction vector off the edge normal
            direction = reflect_direction(direction, edge)

            # Increment bounce count
            bounce_count += 1

def reflect_direction(direction, edge):
    # Get the normal vector of the edge
    p1, p2 = edge
    edge_vector = p2 - p1
    edge_normal = np.array([-edge_vector[1], edge_vector[0]], dtype=np.float64)
    edge_normal /= np.linalg.norm(edge_normal)

    # Reflect the direction vector off the edge normal
    reflected_direction = direction - 2 * np.dot(direction, edge_normal) * edge_normal

    return reflected_direction

def find_closest_edge(polygon, position):
    min_distance = float('inf')
    closest_edge = None

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        edge_vector = p2 - p1
        position_vector = position - p1
        edge_length_squared = np.dot(edge_vector, edge_vector)
        t = np.dot(position_vector, edge_vector) / edge_length_squared
        t = np.clip(t, 0, 1)
        closest_point_on_edge = p1 + t * edge_vector
        distance_squared = np.dot(position - closest_point_on_edge, position - closest_point_on_edge)

        if distance_squared < min_distance:
            min_distance = distance_squared
            closest_edge = (p1, p2)

    return np.sqrt(min_distance), closest_edge



def find_tipCue_tape(cueTip_mask, x1_orig, y1_orig, x2_orig, y2_orig, img_with_polygon):
    # Mss fel roze --> bijna niemand draagt roze kleding, valt op op de pooltafel -----------------------------------------------------------------
    # Zwart lukt niet, teveel donkere kleuren op de pooltafel

    # Red
    # lower_tape = np.array([0, 100, 100])
    # upper_tape = np.array([10, 255, 255])

    # Green
    lower_tape = np.array([40, 40, 40])
    upper_tape = np.array([80, 255, 255])
    tape_mask = cv2.inRange(cueTip_mask, lower_tape, upper_tape)

    contours, _ = cv2.findContours(tape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to compute weighted average
    total_area = 0
    weighted_sum_x = 0
    weighted_sum_y = 0

    for contour in contours:
        # Compute moments of the contour
        M = cv2.moments(contour)
        
        # Calculate centroid coordinates
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # Set centroid to arbitrary point if contour has no area
            cX, cY = 0, 0
        
        # Calculate area of contour
        area = cv2.contourArea(contour)
        
        # Accumulate values for weighted average
        total_area += area
        weighted_sum_x += cX * area
        weighted_sum_y += cY * area

    # Calculate weighted average to get the center of mass
    if total_area != 0:
        center_of_mass_x = int(weighted_sum_x / total_area)
        center_of_mass_y = int(weighted_sum_y / total_area)
    else:
        # Set center of mass to arbitrary point if total area is zero
        center_of_mass_x, center_of_mass_y = 0, 0

    cv2.circle(img_with_polygon, (center_of_mass_x, center_of_mass_y), 5, (20, 20, 20), cv2.FILLED)

    distance_point1 = np.sqrt((x1_orig - center_of_mass_x)**2 + (y1_orig - center_of_mass_y)**2)
    distance_point2 = np.sqrt((x2_orig - center_of_mass_x)**2 + (y2_orig - center_of_mass_y)**2)

    if distance_point1 < distance_point2:
        x_temp, y_temp = x1_orig, y1_orig
        x1_orig, y1_orig = x2_orig, y2_orig
        x2_orig, y2_orig = x_temp, y_temp
    
    return x1_orig, y1_orig, x2_orig, y2_orig, tape_mask