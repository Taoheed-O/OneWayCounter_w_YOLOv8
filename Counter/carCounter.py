# getting started with YOLO
from ultralytics import YOLO
import cv2
import cvzone
from math import ceil
from sort import *

# # initializing model
# model = YOLO('Yolo-Weights/yolov8l.pt')
# results = model('images/students.jpg', show=True)
# cv2.waitKey(0)


cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 150)

model = YOLO('Yolo-Weights/yolov8n.pt')


classnames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird','cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kites', 'baseball bat',
              'baseball glove', 'skateboard', 'surf board', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwitch', 'orange', 'broccoli',
              'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
              'dinningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']

mask = cv2.imread("###")

# Tracker
tracker =  Sort(max_age=20, min_hits=3, iou_threshold=0.3)


while True:
    source, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, x2, x3, x4)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1

            # confidence level 
            conf = ceil(box.conf[0]*100)/100

            # class Name
            cls =  int(box.cls[0])
            current_class = classnames[cls]
            
            # if statement to filter some classes
            if current_class == 'car' and conf > 0.5:
                cvzone.putTextRect(img, f"{classnames[cls]} {conf}", (max(0, x1), max(35, y1)), thickness=1, scale=0.6 , offset=5)
                cvzone.cornerRect(img, (x1, y1,w, h), l=8, rt=5)
                currentArray = np.array([x1,y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    
    # looping through the results
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1,w, h), l=8, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f" {Id}", (max(0, x1), max(35, y1)), thickness=1, scale=0.6 , offset=5)


    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
