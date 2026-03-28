import sys


def ensure_qapplication(argv=None):
    from PySide6 import QtWidgets

    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication(list(argv or sys.argv))
    return application


def show_window(window, *, width: int = 1700, height: int = 860) -> int:
    application = ensure_qapplication()
    window.resize(width, height)
    window.show()
    return application.exec()
