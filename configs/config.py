# configs/config.py
VIDEO_PATH = "data/samples/test1.mp4"  # Đường dẫn video thử
QUEUE_SIZE = 64                      # Kích thước queue buffer
NUM_WORKERS = 2                      # Số worker song song

YOLO_SIZE = 640     # Kích thước đầu vào cho YOLO (ví dụ 640x640)
DEEPLAB_SIZE = 512  # Kích thước đầu vào cho DeepLab (ví dụ 512x512)
