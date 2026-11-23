"""
VOID · FOD Detection / AI Vision

Install:  pip install torch torchvision opencv-python PySide6 numpy pygrabber
          (pygrabber is Windows-only: real camera names such as "OBS Virtual Camera")
Run:      python main.py

Files:    main.py    starts the app
          ui.py      stylesheet, widgets, main window
          engine.py  settings, cameras, model, FOD detection, worker thread

Use: pick a camera -> keep the ply clean for a few seconds (it learns) -> detection runs live.
     Drag on the live feed to mark an object (numbered blue box + tile on the left).
"""
import sys
from PySide6.QtWidgets import QApplication
from ui import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")                 # same look on every OS; the stylesheet does the rest
    win = MainWindow(); win.showFullScreen()
    sys.exit(app.exec())