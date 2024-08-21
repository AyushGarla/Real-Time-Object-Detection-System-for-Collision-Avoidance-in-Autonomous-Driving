import cv2
from ultralytics import YOLO
import cvzone
import math

cap = cv2.VideoCapture(0)  # Using webcam
# cap = cv2.VideoCapture('../videos/MUNICH_dashcam.mp4') # For videos
cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height

# Model
model = YOLO('../yolo_weights/yolov8n.pt')

# Detecting the class from coco dataset
classNames = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing for vechiles over 3.5 metric tons', 'Road Block',
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vechiles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Double curve', 'Bumpy Road', 'Slippery road',
    'Road narrows on the right', 'Road Work', 'Traffic Signals', 'Pedestrians',
    'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left',
    'Roundabout mandatory', 'End of no passing', 'End of no passing by vechiles over 3.5 metric tons'
]

while True:
    success, img = cap.read()
    results = model(img, stream=True)  # stream = True will use generators and will be efficient
    # Creating bounding boxes
    for r in results:
        boxes = r.boxes  # Bounding box for each result
        for box in boxes:  # Loop through boxes
            # Making the bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]  # Finding the (x,y) for each box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert the coordinates to int to use them

            # For cvzone
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            print(x1, y1, w, h)

            # Confidence values
            conf = math.ceil((box.conf[0] * 100)) / 100  # Ceil is to round the conf values
            print(conf)

            # Display the confidence and class name; displaying it on a rectangle (text format)
            # Class name
            cls = int(box.cls[0])
            print(f'Detected class: {cls}')

            # Check if cls is within the range of classNames
            if 0 <= cls < len(classNames):
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=3, thickness=3)
            else:
                print(f'Class index {cls} is out of range')

    cv2.imshow('video', img)
    # If 'q' is pressed, the pop-up is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
