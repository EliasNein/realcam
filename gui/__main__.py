"""Start the GUI: python -m gui"""
import sys

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.cleanup_previews()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
