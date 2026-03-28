import sys
from unittest.mock import patch

from lcip_app.runtime import ensure_qapplication


class FakeQApplication:
    _instance = None
    created_with = None

    def __init__(self, argv):
        type(self)._instance = self
        type(self).created_with = list(argv)

    @classmethod
    def instance(cls):
        return cls._instance


def test_ensure_qapplication_creates_application_when_missing():
    FakeQApplication._instance = None
    FakeQApplication.created_with = None
    fake_pyside6 = type("FakePySide6", (), {"QtWidgets": type("FakeQtWidgets", (), {"QApplication": FakeQApplication})})

    with patch.dict(sys.modules, {"PySide6": fake_pyside6}):
        application = ensure_qapplication(["prog", "--flag"])

    assert application is FakeQApplication._instance
    assert FakeQApplication.created_with == ["prog", "--flag"]


def test_ensure_qapplication_reuses_existing_application():
    FakeQApplication._instance = None
    FakeQApplication.created_with = None
    fake_pyside6 = type("FakePySide6", (), {"QtWidgets": type("FakeQtWidgets", (), {"QApplication": FakeQApplication})})
    existing_application = FakeQApplication(["existing"])

    with patch.dict(sys.modules, {"PySide6": fake_pyside6}):
        application = ensure_qapplication(["prog", "--flag"])

    assert application is existing_application
    assert FakeQApplication.created_with == ["existing"]
