import cv2
import numpy as np
from configs.config import YOLO_SIZE, DEEPLAB_SIZE
from .letterbox import letterbox

def preprocess_yolo(frame):
    """
    Tiền xử lý cho YOLOv8.
    - Letterbox về YOLO_SIZE x YOLO_SIZE
    - Chuyển BGR->RGB
    - Chuyển sang float32, chuẩn hóa [0,1]
    """
    img = letterbox(frame, YOLO_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_deeplab(frame):
    """
    Tiền xử lý cho DeepLabV3+.
    - Resize về DEEPLAB_SIZE x DEEPLAB_SIZE
    - Chuyển BGR->RGB
    - Chuyển sang float32, chuẩn hóa [0,1]
    """
    img = cv2.resize(frame, (DEEPLAB_SIZE, DEEPLAB_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img
