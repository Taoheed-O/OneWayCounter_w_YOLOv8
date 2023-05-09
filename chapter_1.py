# getting started with YOLO
from ultralytics import YOLO
import cv2

# initializing model
model = YOLO('yolov8n.pt')
results = model('images/students.jpg', show=True)
cv2.waitKey(0)
