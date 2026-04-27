# configs/config.py
VIDEO_PATH = "data/samples/test1.mp4"  # Đường dẫn video thử
QUEUE_SIZE = 64                      # Kích thước queue buffer
NUM_WORKERS = 2                      # Số worker song song

YOLO_SIZE = 640     # Kích thước đầu vào cho YOLO (ví dụ 640x640)
DEEPLAB_SIZE = 512  # Kích thước đầu vào cho DeepLab (ví dụ 512x512)

# Cấu hình Visualizer
VISUALIZER_MAX_SIZE = (1280, 720)    # Kích thước hiển thị tối đa (width, height)
VISUALIZER_SKIP_FRAMES = 1           # Vẽ mỗi N frame (1 = vẽ tất cả, 2 = vẽ 1/2 frame)
SHOW_WINDOW = True                   # Hiển thị cửa sổ OpenCV
