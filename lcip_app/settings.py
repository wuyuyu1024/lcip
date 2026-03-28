from dataclasses import dataclass
from pathlib import Path


PROJECTION_CHOICES = ("umap", "tsne", "mds")
INVERSE_PROJECTION_CHOICES = ("ilamp", "nninv", "rbf", "lcip", "imds")
DATASET_CHOICES = ("mnist", "fashionmnist", "w_afhq", "blob")

DEFAULT_DATASET = "w_afhq"
DEFAULT_PROJECTION = "tsne"
DEFAULT_INVERSE_PROJECTION = "lcip"
DEFAULT_GRID = 100

DEFAULT_SAVED_MODEL_DIR = Path("./models/wAFHQv2_paper")
STYLEGAN_CHECKPOINT_PATH = Path("./models/stylegan2-afhqv2-512x512.pkl")
WAFHQ_LATENTS_PATH = Path("./datasets/w_afhqv2/w_afhqv2.npy")
WAFHQ_LABELS_PATH = Path("./datasets/w_afhqv2/labels.npy")


@dataclass(frozen=True)
class BasicDatasetSpec:
    source_name: str | None
    data_shape: tuple[int, ...] | None
    show3d: bool
    train_size: int
    test_size: int | None
    random_state: int


BASIC_DATASET_SPECS = {
    "mnist": BasicDatasetSpec(
        source_name="mnist_784",
        data_shape=(28, 28, 1),
        show3d=False,
        train_size=3000,
        test_size=2000,
        random_state=420,
    ),
    "fashionmnist": BasicDatasetSpec(
        source_name="fashion-mnist",
        data_shape=(28, 28, 1),
        show3d=False,
        train_size=3000,
        test_size=2000,
        random_state=420,
    ),
    "blob": BasicDatasetSpec(
        source_name=None,
        data_shape=None,
        show3d=True,
        train_size=500,
        test_size=None,
        random_state=2,
    ),
}
