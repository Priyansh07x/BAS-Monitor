"""
BAS Experiment Monitor — PySide6 Desktop Shell
ISRO SIH26174 • On-board Edge HAR System

Loads the frontend HTML/CSS/JS dashboard inside a Qt WebEngine window
and connects the Python backend via QWebChannel.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QUrl
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage

from backend.bridge import Bridge
from backend.app_state import AppState


INDEX_FILE = PROJECT_ROOT / "frontend" / "index.html"


class CustomWebEnginePage(QWebEnginePage):
    """Auto-grants camera/microphone permissions for the local frontend."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.featurePermissionRequested.connect(self._on_permission)

    def _on_permission(self, url, feature):
        if feature in (
            QWebEnginePage.Feature.MediaAudioCapture,
            QWebEnginePage.Feature.MediaVideoCapture,
            QWebEnginePage.Feature.MediaAudioVideoCapture,
        ):
            self.setFeaturePermission(
                url, feature, QWebEnginePage.PermissionPolicy.PermissionGrantedByUser
            )
        else:
            self.setFeaturePermission(
                url, feature, QWebEnginePage.PermissionPolicy.PermissionDeniedByUser
            )


def main() -> int:
    app = QApplication(sys.argv)

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Frontend file not found: {INDEX_FILE}")

    window = QMainWindow()
    window.setWindowTitle("BAS Experiment Monitor")
    window.resize(1600, 900)

    # --- Browser + Custom Page (grants camera permissions) ---
    browser = QWebEngineView()
    page = CustomWebEnginePage(browser)
    browser.setPage(page)

    # --- QWebChannel: expose Python backend to JS ---
    app_state = AppState()

    channel = QWebChannel(page)
    bridge = Bridge(app_state)

    channel.registerObject("backend", bridge)
    page.setWebChannel(channel)

    browser.load(QUrl.fromLocalFile(str(INDEX_FILE)))

    window.setCentralWidget(browser)
    window.show()

    # Clean up on exit
    app.aboutToQuit.connect(bridge.shutdown)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())