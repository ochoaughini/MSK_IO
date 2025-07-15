import os
import logging
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from msk_io.utils.log_config import get_logger
from msk_io.errors import ConfigurationError

logger = get_logger(__name__)


class LLMSettings(BaseSettings):
    openai_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("OPENAI_API_KEY", "")),
        description="OpenAI API key for LLM access.",
    )
    google_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("GOOGLE_API_KEY", "")),
        description="Google Cloud API key for LLM access.",
    )
    default_llm_model: str = Field("gpt-4o", description="Default LLM model to use.")
    llm_timeout_seconds: int = Field(
        60, description="Timeout for LLM API calls in seconds."
    )

    model_config = SettingsConfigDict(env_prefix="MSKIO_LLM_", case_sensitive=False)


class ImageProcessingSettings(BaseSettings):
    default_segmentation_model_path: str = Field(
        "/models/nnunet/model_weights.pth",
        description="Default path for deep learning segmentation model weights.",
    )
    segmentation_confidence_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum confidence for segmentation masks."
    )
    image_cache_dir: str = Field(
        "/tmp/msk_io_cache/images",
        description="Directory for caching processed images.",
    )

    model_config = SettingsConfigDict(env_prefix="MSKIO_IMG_", case_sensitive=False)


class RetrievalSettings(BaseSettings):
    dicom_sniff_url: str = Field(
        "http://localhost:8042/dicom", description="Base URL for DICOM stream sniffing."
    )
    ohif_extractor_headless: bool = Field(
        True, description="Run OHIF extractor in headless mode."
    )
    data_download_dir: str = Field(
        "/data/msk_io_downloads", description="Directory for downloaded medical data."
    )
    dicom_sniff_url_timeout_seconds: int = Field(
        30, description="Timeout for remote DICOM URL downloads in seconds."
    )

    model_config = SettingsConfigDict(
        env_prefix="MSKIO_RETRIEVAL_", case_sensitive=False
    )


class IndexerSettings(BaseSettings):
    embedding_model_name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model for text embeddings.",
    )
    vector_db_path: str = Field(
        "/data/msk_io_vectordb",
        description="Path to the vector database for semantic indexing.",
    )

    model_config = SettingsConfigDict(env_prefix="MSKIO_INDEXER_", case_sensitive=False)


class AppSettings(BaseSettings):
    log_level: str = Field(
        "INFO",
        description="Global logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    log_file_path: str = Field(
        default_factory=lambda: os.path.join(os.getcwd(), "msk_io.log"),
        description="Path to the application log file.",
    )
    watch_directory: str = Field(
        "/data/msk_io_input", description="Directory to monitor for new medical data."
    )
    output_directory: str = Field(
        "/data/msk_io_output",
        description="Directory for saving diagnostic reports and results.",
    )
    max_concurrent_tasks: int = Field(
        5, description="Maximum number of concurrent pipeline tasks."
    )

    model_config = SettingsConfigDict(env_prefix="MSKIO_APP_", case_sensitive=False)


class AppConfig(BaseSettings):
    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    image_processing: ImageProcessingSettings = Field(
        default_factory=ImageProcessingSettings
    )
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    indexer: IndexerSettings = Field(default_factory=IndexerSettings)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._setup_logging_from_config()
        self._ensure_directories_exist()
        logger.info("Application configuration loaded successfully.")

    def _setup_logging_from_config(self) -> None:
        from msk_io.utils.log_config import setup_logging

        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = log_level_map.get(self.app.log_level.upper(), logging.INFO)
        setup_logging(level=level, log_file=self.app.log_file_path)

    def _ensure_directories_exist(self) -> None:
        directories = [
            self.app.output_directory,
            self.app.watch_directory,
            self.retrieval.data_download_dir,
            self.image_processing.image_cache_dir,
        ]
        for path in directories:
            try:
                os.makedirs(path, exist_ok=True)
                logger.debug(f"Ensured directory exists: {path}")
            except OSError as e:
                logger.error(f"Failed to create directory {path}: {e}")
                raise ConfigurationError(
                    f"Required directory creation failed: {path}"
                ) from e


def load_config(env_file: str | None = None) -> AppConfig:
    """Load and return the application configuration.

    Parameters
    ----------
    env_file : str | None, optional
        Path to a ``.env`` file. If ``None``, a file named ``.env`` in the
        current working directory is used. The ``MSKIO_APP_ENV_FILE`` environment
        variable takes precedence if set.
    """

    if env_file is None:
        env_file = os.environ.get("MSKIO_APP_ENV_FILE", os.path.join(os.getcwd(), ".env"))

    if os.path.exists(env_file):
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value)
        except OSError as e:
            logger.warning(f"Failed to read env file {env_file}: {e}")

    try:
        config = AppConfig(_env_file=None)
        return config
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        raise ConfigurationError(f"Invalid application configuration: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading configuration: {e}")
        raise ConfigurationError(
            f"Failed to load application configuration: {e}"
        ) from e


if __name__ == "__main__":
    with open(".env", "w") as f:
        f.write("MSKIO_APP_LOG_LEVEL=DEBUG\n")
        f.write("MSKIO_LLM_DEFAULT_LLM_MODEL=llama3\n")
        f.write("MSKIO_RETRIEVAL_DICOM_SNIFF_URL=http://test-dicom.org\n")

    try:
        config = load_config()
        print(f"Log Level: {config.app.log_level}")
        print(f"Default LLM Model: {config.llm.default_llm_model}")
        print(f"DICOM Sniff URL: {config.retrieval.dicom_sniff_url}")
        print(
            f"OpenAI API Key (first 5 chars): {config.llm.openai_api_key.get_secret_value()[:5]}..."
        )
        logger.info("Configuration demo complete.")
    except ConfigurationError as e:
        logger.critical(f"Fatal: {e}")
    finally:
        if os.path.exists(".env"):
            os.remove(".env")
