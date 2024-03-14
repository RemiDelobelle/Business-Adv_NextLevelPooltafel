import cv2
import numpy as np
from Mod_Constants import PRINTS, THRESHOLD_CONTOUR

def processImg(img, interpreter, input_details, output_details):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(image_rgb, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5  # Normalize

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_heatmap = cv2.resize(output_data[0, :, :, 0], (img.shape[1], img.shape[0]))
    outp_hm_np = np.round(1 - output_heatmap, 2)

    threshold_contour = THRESHOLD_CONTOUR
    binary_map = np.where(outp_hm_np >= threshold_contour, 1, 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, output_heatmap