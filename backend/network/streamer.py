import threading
import time
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
import numpy as np
from socketserver import ThreadingMixIn

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    allow_reuse_address = True
    pass

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler that serves a continuous MJPEG stream.
    Reads frames from the global streamer instance.
    """
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            
            # Access the global stream manager
            streamer = get_ip_streamer()
            
            try:
                while True:
                    frame = streamer.get_latest_frame()
                    if frame is None:
                        time.sleep(0.05)
                        continue
                        
                    # Encode frame to JPEG
                    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not ret:
                        continue
                        
                    frame_bytes = jpeg.tobytes()
                    
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame_bytes))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
                    
                    # Control frame rate of the stream (approx 15-20 FPS)
                    time.sleep(0.05)
            except Exception as e:
                # Client disconnected
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default HTTP logging to avoid console spam
        pass


class IPStreamer:
    """
    Manages the background HTTP server for MJPEG streaming.
    Provides thread-safe access to the latest camera frame.
    """
    def __init__(self, port=8554):
        self.port = port
        self.server = None
        self.server_thread = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self._is_running = False

    @property
    def is_running(self):
        return self._is_running

    def get_local_ip(self):
        """Attempts to get the local network IP (e.g., LAN IP)."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Doesn't have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    def update_frame(self, frame: np.ndarray):
        """Called by the main app loop to update the stream with the latest frame."""
        if not self._is_running:
            return
            
        with self._lock:
            # Keep a small reference, resize slightly if needed to save LAN bandwidth
            self._latest_frame = cv2.resize(frame, (854, 480))

    def get_latest_frame(self):
        """Called by the HTTP request handler to get the frame to send."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def start(self):
        """Starts the background HTTP server."""
        if self._is_running:
            return f"http://{self.get_local_ip()}:{self.port}/stream"

        self.server = ThreadedHTTPServer(('0.0.0.0', self.port), MJPEGStreamHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        
        self._is_running = True
        return f"http://{self.get_local_ip()}:{self.port}/stream"

    def stop(self):
        """Stops the background HTTP server."""
        if not self._is_running:
            return
            
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
            
        self._is_running = False
        self.server = None
        self.server_thread = None

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_streamer_instance = None
_streamer_lock = threading.Lock()

def get_ip_streamer(**kwargs) -> IPStreamer:
    """Return or create a module-level IPStreamer singleton."""
    global _streamer_instance
    with _streamer_lock:
        if _streamer_instance is None:
            _streamer_instance = IPStreamer(**kwargs)
        return _streamer_instance
