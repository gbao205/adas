import cv2
import numpy as np

def letterbox(img, new_size):
    """
    Resize ảnh về new_size x new_size bằng letterbox (padding giữ tỉ lệ).
    Màu nền padding là 114 (giống quy ước YOLO).
    """
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(w * scale), int(h * scale)
    # Resize trong khung nhỏ nhất
    resized = cv2.resize(img, (nw, nh))
    # Tạo khung vuông nền xám (114)
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas
