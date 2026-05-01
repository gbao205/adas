import torch
import cv2
import numpy as np
from torchvision import models, transforms

class DeeplabAdapter:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Đang khởi chạy DeepLabV3+ (Real AI) trên thiết bị: {self.device}")
        
        self.model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
        self.model.to(self.device)
        self.model.half()  # Dùng FP16
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Biến cờ để tạo roi_mask (chỉ tạo 1 lần duy nhất)
        self.roi_mask_tensor = None

    def infer(self, frame):
        original_height, original_width = frame.shape[:2]
        
        # 1. Resize nhỏ để chạy model (Scale 50%)
        process_width, process_height = int(original_width * 0.5), int(original_height * 0.5)
        small_frame = cv2.resize(frame, (process_width, process_height))
        
        # 2. Tiền xử lý
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(rgb_frame).unsqueeze(0).to(self.device).half()

        # 3. Chạy Inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # 4. TÍNH TOÁN TRỰC TIẾP TRÊN GPU (Không dùng .cpu() ở đây)
        # Lấy nhãn class (0 là background/đường)
        predicted_classes = output.argmax(0)
        
        # Tạo mask nhị phân (chỉ lấy class 0)
        drivable_area_tensor = (predicted_classes == 0).byte() * 255
        
        # 5. Tạo ROI Mask (Chỉ tính 1 lần, đẩy lên GPU)
        if self.roi_mask_tensor is None:
            mask_roi = np.zeros((process_height, process_width), dtype=np.uint8)
            polygon = np.array([[
                (int(process_width * 0.20), process_height),
                (int(process_width * 0.80), process_height),
                (int(process_width * 0.55), int(process_height * 0.65)),
                (int(process_width * 0.45), int(process_height * 0.65))
            ]], np.int32)
            cv2.fillPoly(mask_roi, polygon, 255)
            # Chuyển ROI mask thành tensor và đẩy lên GPU
            self.roi_mask_tensor = torch.from_numpy(mask_roi).to(self.device).byte()

        # 6. Bitwise AND trực tiếp trên GPU
        final_tensor = torch.bitwise_and(drivable_area_tensor, self.roi_mask_tensor)

        # 7. CHỈ đưa kết quả cuối cùng về CPU
        final_mask_small = final_tensor.cpu().numpy()

        # 8. Phóng to lại bằng kích thước gốc
        final_lane_mask = cv2.resize(final_mask_small, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        return final_lane_mask