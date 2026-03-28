from pathlib import Path

from lcip_app.settings import DEFAULT_SAVED_MODEL_DIR, STYLEGAN_CHECKPOINT_PATH


def train_new_model_basic(
    dataset_name: str = "mnist",
    P_name: str = "tsne",
    Pinv_name: str = "lcip",
    clf=None,
    GRID: int = 100,
) -> int:
    from gui import LCIP_GUI_Basic
    from lcip_app.datasets import load_basic_dataset
    from lcip_app.factories import fit_projection_model, train_classifier
    from lcip_app.runtime import show_window

    dataset = load_basic_dataset(dataset_name)
    classifier = train_classifier(dataset.features, dataset.labels) if clf else None
    trained_projection = fit_projection_model(
        dataset.features,
        projection_name=P_name,
        inverse_name=Pinv_name,
        lcip_beta=0.1,
    )

    window = LCIP_GUI_Basic(
        clf=classifier,
        Pinv=trained_projection.inverse_projection,
        X=dataset.features,
        X2d=trained_projection.embedding,
        y=dataset.labels,
        GRID=GRID,
        show3d=dataset.show3d,
        padding=0.1,
        data_shape=dataset.data_shape,
        cmap="tab10",
    )
    return show_window(window)


def train_new_model_gan(
    P_name: str = "tsne",
    Pinv_name: str = "lcip",
    clf=None,
    GRID: int = 100,
) -> int:
    from gui import LCIP_GUI_GAN
    from lcip_app.datasets import load_gan_dataset
    from lcip_app.factories import fit_projection_model, train_classifier
    from lcip_app.runtime import show_window

    dataset = load_gan_dataset()
    classifier = train_classifier(dataset.features, dataset.labels) if clf else None
    trained_projection = fit_projection_model(
        dataset.features,
        projection_name=P_name,
        inverse_name=Pinv_name,
        lcip_beta=0.01,
    )

    window = LCIP_GUI_GAN(
        clf=classifier,
        Pinv=trained_projection.inverse_projection,
        X=dataset.features,
        X2d=trained_projection.embedding,
        y=dataset.labels,
        GRID=GRID,
        show3d=True,
        padding=0.1,
        data_shape=(512, 512, 3),
        G_path=str(STYLEGAN_CHECKPOINT_PATH),
        w_scaler=dataset.scaler,
        cmap="tab10",
    )
    return show_window(window)


def load_saved_paper(folder: str | Path = DEFAULT_SAVED_MODEL_DIR, clf=None, GRID: int = 100) -> int:
    from gui import LCIP_GUI_GAN
    import numpy as np
    import torch

    from lcip import LCIP
    from lcip_app.datasets import load_gan_dataset
    from lcip_app.factories import train_classifier
    from lcip_app.runtime import show_window

    folder_path = Path(folder)
    dataset = load_gan_dataset()
    data_dict = np.load(folder_path / "data_dict.npz")

    train_features = torch.from_numpy(data_dict["X_train"]).float()
    embedding = data_dict["X2d_unscaled"]
    train_labels = data_dict["y_train"]

    inverse_projection = LCIP()
    inverse_projection.load_model(str(folder_path), input_dim=train_features.shape[1])
    classifier = train_classifier(train_features, train_labels) if clf else None

    window = LCIP_GUI_GAN(
        clf=classifier,
        Pinv=inverse_projection,
        X=train_features,
        X2d=embedding,
        y=train_labels,
        GRID=GRID,
        show3d=True,
        padding=0.1,
        data_shape=(512, 512, 3),
        G_path=str(STYLEGAN_CHECKPOINT_PATH),
        w_scaler=dataset.scaler,
        cmap="tab10",
    )
    return show_window(window)
