import sys
from PySide6.QtWidgets import QApplication, QFileDialog

app = QApplication(sys.argv)
path, _ = QFileDialog.getOpenFileName(None, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv)")
print("Selected:", path)
