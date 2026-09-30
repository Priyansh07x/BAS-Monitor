try:
    import cv2
except ImportError:
    cv2 = None


class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.capture = None

    def open(self):
        if cv2 is None:
            return False
        if self.capture is not None:
            return True

        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            return False

        return True

    def read(self):
        if self.capture is None or cv2 is None:
            return None

        success, frame = self.capture.read()

        if not success:
            return None

        return frame

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def is_open(self):
        return self.capture is not None and getattr(self.capture, "isOpened", lambda: True)()