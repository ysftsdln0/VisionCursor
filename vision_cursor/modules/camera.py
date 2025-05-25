import cv2
import numpy as np
import threading

class Camera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.frame = None
        self.thread = None

    def start(self, callback=None):
        if self.is_running:
            return True
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            print("Kamera açılamadı!")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, args=(callback,))
        self.thread.daemon = True
        self.thread.start()
        return True

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        self.cap = None
        self.frame = None

    def _capture_loop(self, callback):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            self.frame = frame.copy()
            if callback:
                callback(frame)

    def get_frame(self):
        return self.frame.copy() if self.frame is not None else None

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened() 