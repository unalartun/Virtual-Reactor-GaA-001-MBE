"""Launch the Virtual Reactor desktop application."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from models import KMCParameters


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow(KMCParameters())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

