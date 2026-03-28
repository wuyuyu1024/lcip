import torch


class MinMaxTensorScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min = None
        self.max = None

    def fit(self, tensor: torch.Tensor) -> "MinMaxTensorScaler":
        self.min = tensor.min(dim=0).values.detach().cpu()
        self.max = tensor.max(dim=0).values.detach().cpu()
        return self

    def fit_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        self.fit(tensor)
        return self.transform(tensor)

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler must be fitted before calling transform().")
        min_value = self.min.to(tensor.device)
        max_value = self.max.to(tensor.device)
        return (tensor - min_value) / (max_value - min_value)

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler must be fitted before calling inverse_transform().")
        max_value = self.max.to(tensor.device)
        min_value = self.min.to(tensor.device)
        return tensor * (max_value - min_value) + min_value


MinMaxScaler_T = MinMaxTensorScaler

__all__ = ["MinMaxScaler_T", "MinMaxTensorScaler"]
