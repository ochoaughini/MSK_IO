from pathlib import Path
from msk_io.config import PipelineSettings


def test_settings_validation(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings = PipelineSettings(data_path=data_dir)
    assert settings.data_path == data_dir
