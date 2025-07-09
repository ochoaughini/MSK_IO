import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from msk_io.config import PipelineSettings


def test_invalid_threshold(tmp_path: Path):
    cfg = tmp_path / 'settings.json'
    cfg.write_text(json.dumps({'segmentor': {'threshold': -0.1}, 'data_path': str(tmp_path)}))
    with pytest.raises(ValidationError):
        PipelineSettings.model_validate_json(cfg.read_text())


def test_env_override(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    rules = tmp_path / 'rules.json'
    rules.write_text(json.dumps({}))
    monkeypatch.setenv('MSK_RULES_PATH', str(rules))
    settings = PipelineSettings(data_path=data_dir)
    assert settings.rules_path == rules
