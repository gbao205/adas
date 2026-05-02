class ADASConfig:
    # Cấu hình Video Input
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS_TARGET = 30

    # Cấu hình Model YOLOv8 (Object Detection)
    YOLO_WEIGHTS = "weights/yolov8x.pt"
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45

    # Cấu hình Model DeepLabV3+ (Lane Segmentation)
    DEEPLAB_WEIGHTS = "weights/deeplabv3.pth"
    LANE_MASK_THRESHOLD = 0.5

    # Cấu hình Cảnh báo (Warning Logic)
    SAFE_DISTANCE_PIXELS = 150 # Cảnh báo va chạm nếu xe khác cách dưới ngưỡng này
    LANE_DEPARTURE_RATIO = 0.1 # Cảnh báo nếu xe lấn qua 10% vạch kẻ đường


# Cấu hình Pipeline (Compatibility với pipeline.py cũ)
QUEUE_SIZE = 64
NUM_WORKERS = 2

# Cấu hình Preprocessing
YOLO_SIZE = 640
DEEPLAB_SIZE = 512

# Cấu hình Visualizer
VISUALIZER_MAX_SIZE = (1280, 720)
VISUALIZER_SKIP_FRAMES = 1
SHOW_WINDOW = False  # Tắt mặc định để test nhanh