from dataclasses import dataclass

from lcip_app.scalers import MinMaxScaler_T
from lcip_app.settings import BASIC_DATASET_SPECS, WAFHQ_LABELS_PATH, WAFHQ_LATENTS_PATH


@dataclass
class BasicDatasetBundle:
    features: object
    labels: object
    data_shape: tuple[int, ...] | None
    show3d: bool


@dataclass
class GanDatasetBundle:
    features: object
    labels: object
    scaler: MinMaxScaler_T


def load_basic_dataset(dataset_name: str) -> BasicDatasetBundle:
    import numpy as np
    from sklearn.datasets import fetch_openml, make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import minmax_scale

    if dataset_name not in BASIC_DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    spec = BASIC_DATASET_SPECS[dataset_name]

    if dataset_name == "blob":
        features, labels = make_blobs(
            n_samples=spec.train_size,
            centers=6,
            n_features=3,
            random_state=spec.random_state,
            cluster_std=1.8,
        )
    else:
        features, labels = fetch_openml(
            spec.source_name,
            version=1,
            return_X_y=True,
            as_frame=False,
        )
        features = np.asarray(features, dtype="float32") / 255.0
        labels = np.asarray(labels).astype("int")
        features, _, labels, _ = train_test_split(
            features,
            labels,
            train_size=spec.train_size,
            test_size=spec.test_size,
            random_state=spec.random_state,
        )

    features = minmax_scale(features).astype("float32")
    return BasicDatasetBundle(
        features=features,
        labels=labels,
        data_shape=spec.data_shape,
        show3d=spec.show3d,
    )


def load_gan_dataset() -> GanDatasetBundle:
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split

    labels = np.load(WAFHQ_LABELS_PATH)
    latents = torch.from_numpy(np.load(WAFHQ_LATENTS_PATH)).float()

    scaler = MinMaxScaler_T()
    scaled_latents = scaler.fit_transform(latents)
    train_features, _, train_labels, _ = train_test_split(
        scaled_latents,
        labels,
        train_size=5000,
        random_state=42,
    )
    return GanDatasetBundle(features=train_features, labels=train_labels, scaler=scaler)
