# scripts/run_pipeline.py
import os
import sys
import argparse
import customtkinter as ctk
from tkinter import filedialog

# 🔥 Fix lỗi import
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.pipeline.pipeline import Pipeline

# Thiết lập giao diện Web-like (Dark Mode, Màu chủ đạo Xanh biển)
ctk.set_appearance_mode("Dark")  # Có thể đổi thành "Light"
ctk.set_default_color_theme("blue")

class ADASLauncher(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Cấu hình cửa sổ
        self.title("ADAS Vision Engine")
        self.geometry("650x450")
        self.eval('tk::PlaceWindow . center') # Canh giữa màn hình

        # 1. Tiêu đề chính (Header)
        self.header_label = ctk.CTkLabel(
            self, 
            text="ADAS VISION ENGINE", 
            font=ctk.CTkFont(family="Roboto", size=28, weight="bold"),
            text_color="#3b82f6" # Màu xanh Web Tailwind
        )
        self.header_label.pack(pady=(40, 5))

        self.sub_label = ctk.CTkLabel(
            self, 
            text="Hệ thống nhận diện xe cộ & làn đường thời gian thực", 
            font=ctk.CTkFont(family="Roboto", size=14),
            text_color="gray"
        )
        self.sub_label.pack(pady=(0, 30))

        # 2. Khu vực chọn file (Upload Box)
        self.frame = ctk.CTkFrame(self, width=500, height=120, corner_radius=15)
        self.frame.pack(pady=10, padx=40, fill="both", expand=True)

        self.file_label = ctk.CTkLabel(
            self.frame, 
            text="📁 Chưa có video nào được chọn", 
            font=ctk.CTkFont(size=14)
        )
        self.file_label.pack(pady=(30, 15))

        self.select_btn = ctk.CTkButton(
            self.frame, 
            text="Tải Video Lên (Browse)", 
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=8,
            command=self.select_file
        )
        self.select_btn.pack()

        # 3. Nút Khởi động (Action Button)
        self.run_btn = ctk.CTkButton(
            self, 
            text="🚀 BẮT ĐẦU CHẠY PIPELINE", 
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#3f3f46",             # Nền xám đen khi vô hiệu hóa
            text_color_disabled="#d4d4d8",  # Chữ xám sáng để nổi bật trên nền tối
            height=50,
            corner_radius=10,
            state="disabled",               # Khóa nút khi chưa chọn file
            command=self.run_pipeline
        )
        self.run_btn.pack(pady=(20, 40))

        self.video_path = None

    def select_file(self):
        """Mở hộp thoại chọn file"""
        path = filedialog.askopenfilename(
            title="Chọn video test",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
        )
        if path:
            self.video_path = path
            # Hiển thị tên file thay vì đường dẫn dài ngoằng
            file_name = os.path.basename(path)
            self.file_label.configure(text=f"🎥 Đã chọn: {file_name}", text_color="#10b981")
            
            # Mở khóa nút Bắt đầu
            self.run_btn.configure(state="normal")

    def run_pipeline(self):
        """Chạy ADAS Pipeline"""
        print(f"[INFO] Bắt đầu xử lý video: {self.video_path}")
        self.withdraw()  # Ẩn giao diện Launcher đi
        
        # Chạy pipeline hệ thống chính
        pipeline = Pipeline(self.video_path)
        pipeline.run()
        
        self.destroy()  # Đóng hẳn chương trình sau khi video chạy xong


def main():
    # Vẫn giữ lại Argparse cho dân kỹ thuật chạy bằng dòng lệnh nếu muốn
    parser = argparse.ArgumentParser(description="Run Video Pipeline")
    parser.add_argument("--video", type=str, default=None, help="Đường dẫn tới file video (.mp4)")
    args = parser.parse_args()

    if args.video:
        # Nếu truyền bằng CLI, chạy thẳng không mở GUI
        if not os.path.exists(args.video):
            print(f"[ERROR] Không tìm thấy video: {args.video}")
            return
        pipeline = Pipeline(args.video)
        pipeline.run()
    else:
        # Nếu không truyền gì, mở giao diện App hiện đại
        app = ADASLauncher()
        app.mainloop()

if __name__ == "__main__":
    main()