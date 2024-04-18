import cv2
import numpy as np


def detect_light_color(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, yellow, green
    red_lower = np.array([0, 50, 50])
    red_upper = np.array([20, 255, 255])
    yellow_lower = np.array([20, 50, 50])
    yellow_upper = np.array([40, 255, 255])
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])

    # Threshold the HSV image to get only specific colors
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Check which color is the most prevalent
    red_pixels = np.sum(red_mask)
    yellow_pixels = np.sum(yellow_mask)
    green_pixels = np.sum(green_mask)

    # Return the color
    if red_pixels > yellow_pixels and red_pixels > green_pixels:
        return {'name': 'Red', 'value': (0, 0, 255)}
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
        return {'name': 'Yellow', 'value': (0, 128, 255)}
    elif green_pixels > red_pixels and green_pixels > yellow_pixels:
        return {'name': 'Green', 'value': (0, 255, 0)}
    else:
        return {'name': 'Unknown', 'value': 0}


# Load YOLO, pre-trained model and weights
net = cv2.dnn.readNet('./assets/yolov4.weights', './assets/yolov4.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open video file
cap = cv2.VideoCapture('./assets/la_cruise.mp4')

ctr = 0
while True:
    # Video Frame Analysis (Every 3rd frame)
    ret, frame = cap.read()
    if not ret:
        break

    ctr += 1
    if ctr % 3 != 0:
        continue

    # Preprocessing the image for YOLO and running forward propagation
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Variables for storing data gathered from current img frame
    class_ids = []
    confidences = []
    boxes = []

    Height, Width = frame.shape[:2]

    # Processing detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter only 'traffic light' detections
            if class_id == 9:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.5)

    for i in indices:
        if type(i) == np.ndarray or type(i) == list:
            print(i)
            i = i[0]

        box = boxes[i]
        x, y, w, h = box[:4]

        # Color detection
        cropped_img = frame[y:y+h, x:x+w]

        if cropped_img.size != 0:
            light_color = detect_light_color(cropped_img)

        if light_color['value'] == 0:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), light_color['value'], 2)
        cv2.putText(frame, f"{light_color['name']}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_color['value'], 2)

    cv2.imshow('Traffic Light Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit the program
        break

cap.release()
cv2.destroyAllWindows()
