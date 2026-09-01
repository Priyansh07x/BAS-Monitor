import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QUrl
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView

from backend.bridge import Bridge
from backend.app_state import AppState


INDEX_FILE = PROJECT_ROOT / "frontend" / "index.html"


def main() -> int:
    app = QApplication(sys.argv)

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Frontend file not found: {INDEX_FILE}"
        )

    window = QMainWindow()
    window.setWindowTitle("BAS Experiment Monitor")
    window.resize(1600, 900)

    browser = QWebEngineView()

    app_state = AppState()

    channel = QWebChannel()
    bridge = Bridge(app_state)

    channel.registerObject("backend", bridge)
    browser.page().setWebChannel(channel)

    browser.load(QUrl.fromLocalFile(str(INDEX_FILE)))

    window.setCentralWidget(browser)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())