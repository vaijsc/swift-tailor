import pytest

from src.data.garmentcode_dataset import GarmentCodeData, DATASET_SPLIT


class TestGarmentCodeData:
    @pytest.fixture
    def garmentcode_test_ds(self):
        return GarmentCodeData(
            use_official_split=False,
            split=DATASET_SPLIT.TEST_MODE,
        )

    def test_garmentcode_test_ds(self, garmentcode_test_ds):
        # Check if the dataset is loaded correctly
        assert len(garmentcode_test_ds) == 100
        assert garmentcode_test_ds[0].shape == (128, 128, 3)

    def test_existing_split(self):
        with pytest.raises(ValueError):
            GarmentCodeData(
                use_official_split=False,
                split="non_existing_split",  # type: ignore
            )

    def test_all_files_exist(self, garmentcode_test_ds):
        for i in range(len(garmentcode_test_ds)):
            garmentcode_test_ds[i]
