import pytest

torch = pytest.importorskip("torch")

from lcip_app.scalers import MinMaxScaler_T


def test_round_trip_preserves_original_values():
    tensor = torch.tensor([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    scaler = MinMaxScaler_T()

    scaled = scaler.fit_transform(tensor)
    restored = scaler.inverse_transform(scaled)

    assert torch.allclose(scaled[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(scaled[-1], torch.tensor([1.0, 1.0]))
    assert torch.allclose(restored, tensor)
