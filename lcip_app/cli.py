import argparse

from lcip_app.launchers import load_saved_paper, train_new_model_basic, train_new_model_gan
from lcip_app.settings import (
    DATASET_CHOICES,
    DEFAULT_DATASET,
    DEFAULT_GRID,
    DEFAULT_INVERSE_PROJECTION,
    DEFAULT_PROJECTION,
    DEFAULT_SAVED_MODEL_DIR,
    INVERSE_PROJECTION_CHOICES,
    PROJECTION_CHOICES,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--load_paper",
        action="store_true",
        help="Load the saved model from the paper. This ignores the projection and inverse projection selections.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=DATASET_CHOICES,
        default=DEFAULT_DATASET,
        help="Choose the dataset. The default is wAFHQv2.",
    )
    parser.add_argument(
        "-p",
        "--projection",
        type=str,
        choices=PROJECTION_CHOICES,
        default=DEFAULT_PROJECTION,
        help="Choose the projection method.",
    )
    parser.add_argument(
        "-i",
        "--pinv",
        type=str,
        choices=INVERSE_PROJECTION_CHOICES,
        default=DEFAULT_INVERSE_PROJECTION,
        help="Choose the inverse projection method.",
    )
    parser.add_argument(
        "-c",
        "--clf",
        action="store_true",
        help="Train a classifier for decision maps.",
    )
    parser.add_argument(
        "-g",
        "--grid",
        type=int,
        default=DEFAULT_GRID,
        help="Grid size for decision maps. Default: 100.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    classifier = True if args.clf else None

    if args.load_paper:
        print("Loading the saved model from the paper")
        return load_saved_paper(DEFAULT_SAVED_MODEL_DIR, clf=classifier, GRID=args.grid)

    print(f"Training a new model on {args.dataset} dataset")
    if args.dataset == "w_afhq":
        return train_new_model_gan(
            P_name=args.projection,
            Pinv_name=args.pinv,
            clf=classifier,
            GRID=args.grid,
        )
    return train_new_model_basic(
        dataset_name=args.dataset,
        P_name=args.projection,
        Pinv_name=args.pinv,
        clf=classifier,
        GRID=args.grid,
    )
