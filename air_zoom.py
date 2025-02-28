import cv2
import mediapipe as mp
import numpy as np
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load the image
image_path = r"C:\Users\Harshavardhan Reddy\Desktop\testimage.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found! Check the file path.")
    exit()

# Get image dimensions
height, width, _ = image.shape
zoom_factor = 1.0
center_x, center_y = width // 2, height // 2

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for natural hand tracking
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Use first detected hand

        # Get fingertip landmarks for index and thumb
        index_finger_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]

        # Convert to pixel coordinates
        index_x, index_y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)
        thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)

        # Calculate distance between thumb and index finger
        distance = np.linalg.norm([index_x - thumb_x, index_y - thumb_y])

        # Adjust zoom factor based on pinch gesture
        zoom_factor = np.clip(zoom_factor + (distance - 50) * 0.001, 1.0, 3.0)

        # Draw landmarks
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Compute zoomed region
    zoom_w, zoom_h = int(width / zoom_factor), int(height / zoom_factor)
    x1, y1 = max(0, center_x - zoom_w // 2), max(0, center_y - zoom_h // 2)
    x2, y2 = min(width, center_x + zoom_w // 2), min(height, center_y + zoom_h // 2)

    # Crop and resize the image
    zoomed_image = image[y1:y2, x1:x2]
    zoomed_image = cv2.resize(zoomed_image, (width, height))

    # Show results
    cv2.imshow("Air Zoom", zoomed_image)
    cv2.imshow("Webcam Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

