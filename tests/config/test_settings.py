from pathlib import Path
import json
from msk_io.config import PipelineSettings


def test_env_loading(tmp_path, monkeypatch):
    cfg = tmp_path / "rules.json"
    cfg.write_text(json.dumps({}))
    monkeypatch.setenv("MSK_RULES_PATH", str(cfg))
    monkeypatch.setenv("MSK_THRESHOLD", "0.7")
    settings = PipelineSettings()
    assert settings.rules_path == cfg
    assert settings.threshold == 0.7
