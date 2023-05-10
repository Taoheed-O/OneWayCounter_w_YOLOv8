# getting started with YOLO
from ultralytics import YOLO
import cv2
import cvzone
from math import ceil

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
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, x2, x3, x4)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1,w, h))

            conf = ceil(box.conf[0]*100)/100
            cvzone.putTextRect(img, f"{conf}", (max(0, x1), max(35, y1)))
            print(conf)

    cv2.imshow('image', img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


