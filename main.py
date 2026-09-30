"""
BAS Experiment Monitor — PySide6 Desktop Shell
ISRO SIH26174 • On-board Edge HAR System

Loads the frontend HTML/CSS/JS dashboard inside a Qt WebEngine window
and connects the Python backend via QWebChannel.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication, QMainWindow
from backend.bridge import Bridge
from backend.app_state import AppState
from backend.network.gui_server import GUIServer

INDEX_FILE = PROJECT_ROOT / "frontend" / "index.html"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

def main() -> int:
    app = QApplication(sys.argv)

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Frontend file not found: {INDEX_FILE}")

    # Main window just shows instructions
    window = QMainWindow()
    window.setWindowTitle("BAS Experiment Monitor (Edge Node)")
    window.resize(600, 200)
    
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
    from PySide6.QtCore import Qt
    
    label = QLabel("Edge Server Running.\nConnect via LAN: http://<EDGE_LAN_IP>:8000\nor locally at http://localhost:8000")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    font = label.font()
    font.setPointSize(16)
    label.setFont(font)
    
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    layout.addWidget(label)
    window.setCentralWidget(central_widget)

    app_state = AppState()
    bridge = Bridge(app_state)
    
    # Start the LAN HTTP + WebSocket server
    gui_server = GUIServer(bridge, FRONTEND_DIR, http_port=8000, ws_port=8001)

    # Start the WebRTC signaling server (Phase A8-10 Resilient Streaming)
    from backend.network.webrtc_server import WebRTCServer
    webrtc_server = WebRTCServer(port=8555)
    webrtc_server.start()

    window.show()

    # Clean up on exit
    def cleanup():
        bridge.shutdown()
        gui_server.shutdown()
        
    app.aboutToQuit.connect(cleanup)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
