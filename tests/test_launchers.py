import sys
import types
from unittest.mock import patch

from lcip_app.launchers import train_new_model_basic


def test_basic_launcher_initializes_qapplication_before_window():
    events = []

    class FakeWindow:
        def __init__(self, **kwargs):
            events.append("window")

    fake_datasets = types.ModuleType("lcip_app.datasets")
    fake_factories = types.ModuleType("lcip_app.factories")

    fake_datasets.load_basic_dataset = lambda dataset_name: type(
        "Dataset",
        (),
        {
            "features": "features",
            "labels": "labels",
            "data_shape": (28, 28, 1),
            "show3d": False,
        },
    )()
    fake_factories.fit_projection_model = lambda *args, **kwargs: type(
        "Projection",
        (),
        {
            "inverse_projection": "pinv",
            "embedding": "embedding",
        },
    )()
    fake_factories.train_classifier = lambda *args, **kwargs: "classifier"

    with patch("lcip_app.launchers.import_basic_gui", return_value=FakeWindow):
        with patch.dict(sys.modules, {"lcip_app.datasets": fake_datasets, "lcip_app.factories": fake_factories}):
            with patch("lcip_app.runtime.ensure_qapplication", side_effect=lambda: events.append("app")):
                with patch("lcip_app.runtime.show_window", return_value=0):
                    result = train_new_model_basic(clf=None)

    assert result == 0
    assert events == ["app", "window"]
