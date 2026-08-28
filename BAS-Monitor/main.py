import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView


BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "frontend" / "index.html"


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
    browser.load(QUrl.fromLocalFile(str(INDEX_FILE)))

    window.setCentralWidget(browser)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
