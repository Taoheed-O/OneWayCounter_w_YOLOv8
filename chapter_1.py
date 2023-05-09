# getting started with YOLO
from ultralytics import YOLO
import cv2

# initializing model
model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model('images/students.jpg', show=True)
cv2.waitKey(0)
