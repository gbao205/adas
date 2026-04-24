import time

class FPS:
    """Đơn giản đo số frame trên giây."""
    def __init__(self):
        self.start_time = time.time()
        self.count = 0

    def increment(self, n=1):
        self.count += n

    def get(self):
        elapsed = time.time() - self.start_time
        return self.count / elapsed if elapsed > 0 else 0.0
