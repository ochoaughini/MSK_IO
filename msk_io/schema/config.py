from typing import Optional
from pydantic import Field, SecretStr
from msk_io.schema._pydantic_base import MSKIOBaseModel

class LLMSettingsSchema(MSKIOBaseModel):
    openai_api_key: Optional[SecretStr] = Field(None, description="OpenAI API key.")
    google_api_key: Optional[SecretStr] = Field(None, description="Google Cloud API key.")
    default_llm_model: str = Field("gpt-4o", description="Default LLM model to use.")
    llm_timeout_seconds: int = Field(60, description="Timeout for LLM API calls.")

class ImageProcessingSettingsSchema(MSKIOBaseModel):
    default_segmentation_model_path: str = Field("/models/nnunet/model_weights.pth", description="Default path for deep learning segmentation model weights.")
    segmentation_confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence for segmentation masks.")
    image_cache_dir: str = Field("/tmp/msk_io_cache/images", description="Directory for caching processed images.")

class RetrievalSettingsSchema(MSKIOBaseModel):
    dicom_sniff_url: str = Field("http://localhost:8042/dicom", description="Base URL for DICOM stream sniffing.")
    ohif_extractor_headless: bool = Field(True, description="Run OHIF extractor in headless mode.")
    data_download_dir: str = Field("/data/msk_io_downloads", description="Directory for downloaded medical data.")

class IndexerSettingsSchema(MSKIOBaseModel):
    embedding_model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="HuggingFace model for text embeddings.")
    vector_db_path: str = Field("/data/msk_io_vectordb", description="Path to the vector database for semantic indexing.")

class AppSettingsSchema(MSKIOBaseModel):
    log_level: str = Field("INFO", description="Global logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    log_file_path: Optional[str] = Field(None, description="Path to the application log file.")
    watch_directory: str = Field("/data/msk_io_input", description="Directory to monitor for new medical data.")
    output_directory: str = Field("/data/msk_io_output", description="Directory for saving diagnostic reports and results.")
    max_concurrent_tasks: int = Field(5, description="Maximum number of concurrent pipeline tasks.")

class AppConfigSchema(MSKIOBaseModel):
    app: AppSettingsSchema = Field(default_factory=AppSettingsSchema)
    llm: LLMSettingsSchema = Field(default_factory=LLMSettingsSchema)
    image_processing: ImageProcessingSettingsSchema = Field(default_factory=ImageProcessingSettingsSchema)
    retrieval: RetrievalSettingsSchema = Field(default_factory=RetrievalSettingsSchema)
    indexer: IndexerSettingsSchema = Field(default_factory=IndexerSettingsSchema)
