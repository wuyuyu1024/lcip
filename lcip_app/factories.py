from dataclasses import dataclass

from lcip.utils import Simple_P_wrapper


@dataclass
class ProjectionTrainingResult:
    inverse_projection: object
    embedding: object


def create_projection(name: str) -> object:
    name = name.lower()
    if name == "umap":
        from umap import UMAP

        return UMAP(n_components=2, random_state=420)

    from sklearn.manifold import MDS, TSNE

    if name == "tsne":
        return TSNE(n_components=2, random_state=420, n_jobs=8)
    if name == "mds":
        return MDS(n_components=2, random_state=420, n_jobs=8)
    raise ValueError(f"Unsupported projection: {name}")


def create_inverse_projection(name: str, *, lcip_beta: float) -> object:
    name = name.lower()
    if name == "ilamp":
        from invprojection import Pinv_ilamp

        return Pinv_ilamp(k=6)
    if name == "nninv":
        from invprojection import NNinv_torch

        return NNinv_torch()
    if name == "rbf":
        from invprojection import RBFinv

        return RBFinv()
    if name == "lcip":
        from lcip import LCIP

        return LCIP(beta=lcip_beta)
    if name == "imds":
        from invprojection import MDSinv

        return MDSinv()
    raise ValueError(f"Unsupported inverse projection: {name}")


def train_classifier(features: object, labels: object, *, epochs: int = 100) -> object:
    import numpy as np
    import torch

    from classifiers import NNClassifier

    classifier = NNClassifier(
        input_dim=features.shape[1],
        n_classes=np.unique(labels).shape[0],
        layer_sizes=(512, 256, 128),
    )
    label_tensor = torch.from_numpy(np.asarray(labels)).long().to(classifier.device)
    if isinstance(features, torch.Tensor):
        feature_tensor = features.float().to(classifier.device)
    else:
        feature_tensor = torch.from_numpy(features).float().to(classifier.device)
    dataset = torch.utils.data.TensorDataset(feature_tensor, label_tensor)
    classifier.fit(dataset, epochs=epochs)
    return classifier


def fit_projection_model(
    features: object,
    *,
    projection_name: str,
    inverse_name: str,
    lcip_beta: float,
    epochs: int = 120,
) -> ProjectionTrainingResult:
    wrapper = Simple_P_wrapper(
        create_projection(projection_name),
        create_inverse_projection(inverse_name, lcip_beta=lcip_beta),
    )
    wrapper.fit(features, epochs=epochs, early_stop=False)
    return ProjectionTrainingResult(
        inverse_projection=wrapper.Pinv,
        embedding=wrapper.X2d,
    )
