import unittest

from lcip_app.settings import BASIC_DATASET_SPECS


class DatasetSettingsTests(unittest.TestCase):
    def test_image_datasets_have_expected_shape(self):
        self.assertEqual(BASIC_DATASET_SPECS["mnist"].data_shape, (28, 28, 1))
        self.assertEqual(BASIC_DATASET_SPECS["fashionmnist"].data_shape, (28, 28, 1))

    def test_blob_dataset_is_marked_as_3d(self):
        self.assertTrue(BASIC_DATASET_SPECS["blob"].show3d)
        self.assertIsNone(BASIC_DATASET_SPECS["blob"].data_shape)


if __name__ == "__main__":
    unittest.main()
