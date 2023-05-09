# getting started with YOLO
from ultralytics import YOLO
import cv2
import cvzone

# # initializing model
# model = YOLO('Yolo-Weights/yolov8l.pt')
# results = model('images/students.jpg', show=True)
# cv2.waitKey(0)


cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)


model = YOLO('Yolo-Weights/yolov8l.pt')


while True:
    source, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, x2, x3, x4 = boxes.xyxy[0]
            x1, x2, x3, x4 = int(x1), int(x2), int(x3), int(x4)
            print(x1, x2, x3, x4)

    cv2.imshow('image', img)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
