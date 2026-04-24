import argparse
import os
from src.pipeline.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Video Pipeline")

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Đường dẫn tới file video (.mp4)"
    )

    args = parser.parse_args()

    video_path = args.video

    # Check file tồn tại
    if not os.path.exists(video_path):
        print(f"[ERROR] Không tìm thấy video: {video_path}")
        return

    print(f"[INFO] Đang chạy với video: {video_path}")

    pipeline = Pipeline(video_path)
    pipeline.run()


if __name__ == "__main__":
    main()