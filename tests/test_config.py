import pytest
import os
import logging
from unittest.mock import patch
from pydantic import ValidationError, SecretStr

from msk_io.config import AppConfig, load_config, AppSettings, LLMSettings
from msk_io.errors import ConfigurationError
from msk_io.utils.log_config import setup_logging


@pytest.fixture(autouse=True)
def clean_env_vars():
    original_env = os.environ.copy()
    keys_to_clear = [
        "MSKIO_APP_WATCH_DIRECTORY",
        "MSKIO_APP_OUTPUT_DIRECTORY",
        "MSKIO_RETRIEVAL_DATA_DOWNLOAD_DIR",
        "MSKIO_IMG_IMAGE_CACHE_DIR",
        "MSKIO_INDEXER_VECTOR_DB_PATH",
        "MSKIO_LLM_OPENAI_API_KEY",
        "MSKIO_LLM_GOOGLE_API_KEY",
        "MSKIO_IMG_DEFAULT_SEGMENTATION_MODEL_PATH",
        "MSKIO_APP_LOG_LEVEL",
        "MSKIO_APP_ENV_FILE",
        "MSKIO_APP_LOG_FILE_PATH",
    ]
    for key in keys_to_clear:
        if key in os.environ:
            del os.environ[key]
    yield
    for key in keys_to_clear:
        if key in os.environ:
            del os.environ[key]
    for key, value in original_env.items():
        if key not in os.environ:
            os.environ[key] = value


def test_app_config_loads_defaults():
    config = load_config()
    assert config.app.log_level == "INFO"
    assert config.llm.default_llm_model == "gpt-4o"
    assert config.llm.openai_api_key.get_secret_value() == ""
    assert "msk_io.log" in config.app.log_file_path


def test_app_config_loads_from_env_vars():
    os.environ["MSKIO_APP_WATCH_DIRECTORY"] = "/custom/watch"
    os.environ["MSKIO_LLM_DEFAULT_LLM_MODEL"] = "llama-2"
    os.environ["MSKIO_LLM_OPENAI_API_KEY"] = "sk-testkey123"

    config = load_config()
    assert config.app.watch_directory == "/custom/watch"
    assert config.llm.default_llm_model == "llama-2"
    assert config.llm.openai_api_key.get_secret_value() == "sk-testkey123"


def test_app_config_loads_from_dot_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
MSKIO_APP_LOG_LEVEL=DEBUG
MSKIO_RETRIEVAL_DICOM_SNIFF_URL=http://test-dicom.local
MSKIO_LLM_GOOGLE_API_KEY=test-google-secret
"""
    )
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        config = load_config()
        assert config.app.log_level == "DEBUG"
        assert config.retrieval.dicom_sniff_url == "http://test-dicom.local"
        assert config.llm.google_api_key.get_secret_value() == "test-google-secret"
    finally:
        os.chdir(original_cwd)


def test_app_config_env_vars_override_dot_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MSKIO_LLM_DEFAULT_LLM_MODEL=llama-2")
    os.environ["MSKIO_LLM_DEFAULT_LLM_MODEL"] = "gpt-3.5-turbo"

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        config = load_config()
        assert config.llm.default_llm_model == "gpt-3.5-turbo"
    finally:
        os.chdir(original_cwd)


def test_app_config_validation_error():
    os.environ["MSKIO_APP_MAX_CONCURRENT_TASKS"] = "not-an-int"
    with pytest.raises(ConfigurationError) as excinfo:
        load_config()
    assert "Invalid application configuration" in str(excinfo.value)
    assert "value is not a valid integer" in str(excinfo.value)


def test_app_config_directory_creation(tmp_path):
    watch_dir = tmp_path / "new_watch_dir"
    output_dir = tmp_path / "new_output_dir"
    download_dir = tmp_path / "new_download_dir"
    cache_dir = tmp_path / "new_cache_dir"

    os.environ["MSKIO_APP_WATCH_DIRECTORY"] = str(watch_dir)
    os.environ["MSKIO_APP_OUTPUT_DIRECTORY"] = str(output_dir)
    os.environ["MSKIO_RETRIEVAL_DATA_DOWNLOAD_DIR"] = str(download_dir)
    os.environ["MSKIO_IMG_IMAGE_CACHE_DIR"] = str(cache_dir)

    config = load_config()

    assert watch_dir.is_dir()
    assert output_dir.is_dir()
    assert download_dir.is_dir()
    assert cache_dir.is_dir()


def test_app_config_logging_setup_on_load():
    with patch('msk_io.config.setup_logging') as mock_setup_logging:
        config = load_config()
        mock_setup_logging.assert_called_once()
        args, kwargs = mock_setup_logging.call_args
        assert kwargs['level'] == logging.INFO
        assert kwargs['log_file'] == config.app.log_file_path


def test_app_config_log_level_changes_logging(tmp_path):
    log_file = tmp_path / "test.log"
    os.environ["MSKIO_APP_LOG_LEVEL"] = "DEBUG"
    os.environ["MSKIO_APP_LOG_FILE_PATH"] = str(log_file)

    with patch('msk_io.config.setup_logging') as mock_setup_logging:
        config = load_config()
        mock_setup_logging.assert_called_once_with(level=logging.DEBUG, log_file=str(log_file))


def test_secret_str_redaction():
    os.environ["MSKIO_LLM_OPENAI_API_KEY"] = "my_super_secret_key"
    config = load_config()
    config_dict = config.model_dump(mode='json', exclude_sensitive=True)
    assert config_dict['llm']['openai_api_key'] == "********"
    assert config.llm.openai_api_key.get_secret_value() == "my_super_secret_key"

