from app.models import DatasetConfig
from app.services.dataset_validator import validate_dataset


def test_missing_root():
    cfg = DatasetConfig(root="nonexistent_path_xyz", data_yaml=None, task_type="detect")
    result = validate_dataset(cfg)
    assert result.ok is False
    assert any("Root not found" in msg.text for msg in result.messages)


def test_missing_data_yaml(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    cfg = DatasetConfig(root=str(root), data_yaml=str(root / "data.yaml"), task_type="detect")
    result = validate_dataset(cfg)
    assert result.ok is False
