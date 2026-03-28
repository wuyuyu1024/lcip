import unittest

try:
    import torch
    from lcip_app.scalers import MinMaxScaler_T
except ModuleNotFoundError:
    torch = None
    MinMaxScaler_T = None


@unittest.skipUnless(torch is not None, "torch is not installed")
class MinMaxScalerTests(unittest.TestCase):
    def test_round_trip_preserves_original_values(self):
        tensor = torch.tensor([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        scaler = MinMaxScaler_T()

        scaled = scaler.fit_transform(tensor)
        restored = scaler.inverse_transform(scaled)

        self.assertTrue(torch.allclose(scaled[0], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.allclose(scaled[-1], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(restored, tensor))


if __name__ == "__main__":
    unittest.main()
