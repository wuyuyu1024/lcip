from unittest.mock import patch

from lcip_app.cli import build_parser, main
from lcip_app.settings import (
    DEFAULT_DATASET,
    DEFAULT_GRID,
    DEFAULT_INVERSE_PROJECTION,
    DEFAULT_PROJECTION,
    DEFAULT_SAVED_MODEL_DIR,
)


def test_parser_defaults_match_runtime_defaults():
    args = build_parser().parse_args([])
    assert args.dataset == DEFAULT_DATASET
    assert args.projection == DEFAULT_PROJECTION
    assert args.pinv == DEFAULT_INVERSE_PROJECTION
    assert args.grid == DEFAULT_GRID
    assert args.load_paper is False
    assert args.clf is False


def test_main_routes_basic_datasets_to_basic_launcher():
    with patch("lcip_app.cli.train_new_model_basic") as basic_launcher, patch(
        "lcip_app.cli.train_new_model_gan"
    ) as gan_launcher:
        basic_launcher.return_value = 0

        exit_code = main(["-d", "mnist", "-c", "-g", "120"])

    assert exit_code == 0
    gan_launcher.assert_not_called()
    basic_launcher.assert_called_once_with(
        dataset_name="mnist",
        P_name="tsne",
        Pinv_name="lcip",
        clf=True,
        GRID=120,
    )


def test_main_routes_wafhq_to_gan_launcher():
    with patch("lcip_app.cli.train_new_model_basic") as basic_launcher, patch(
        "lcip_app.cli.train_new_model_gan"
    ) as gan_launcher:
        gan_launcher.return_value = 0

        exit_code = main(["-d", "w_afhq", "-p", "umap", "-i", "rbf"])

    assert exit_code == 0
    basic_launcher.assert_not_called()
    gan_launcher.assert_called_once_with(
        P_name="umap",
        Pinv_name="rbf",
        clf=None,
        GRID=100,
    )


def test_main_routes_load_paper_flag():
    with patch("lcip_app.cli.load_saved_paper") as load_saved_paper_mock:
        load_saved_paper_mock.return_value = 0

        exit_code = main(["-l"])

    assert exit_code == 0
    load_saved_paper_mock.assert_called_once_with(
        DEFAULT_SAVED_MODEL_DIR,
        clf=None,
        GRID=100,
    )
