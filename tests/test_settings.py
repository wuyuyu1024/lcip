from lcip_app.settings import BASIC_DATASET_SPECS


def test_image_datasets_have_expected_shape():
    assert BASIC_DATASET_SPECS["mnist"].data_shape == (28, 28, 1)
    assert BASIC_DATASET_SPECS["fashionmnist"].data_shape == (28, 28, 1)


def test_blob_dataset_is_marked_as_3d():
    assert BASIC_DATASET_SPECS["blob"].show3d is True
    assert BASIC_DATASET_SPECS["blob"].data_shape is None
