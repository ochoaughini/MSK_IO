import pytest
import os
import shutil
from unittest.mock import MagicMock
from datetime import date, time
import logging
import pydicom
from pydicom import Dataset, uid
from pydicom.sequence import Sequence
from pydicom.dataset import FileDataset, FileMetaDataset

from msk_io.config import AppConfig, load_config
from msk_io.utils.log_config import setup_logging
from msk_io.schema.dicom_data import DICOMVolume, DICOMPatientInfo, DICOMStudyInfo, DICOMSeriesInfo


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Sets up logging for the test session."""
    setup_logging(level=getattr(logging, os.getenv("TEST_LOG_LEVEL", "INFO").upper()))


@pytest.fixture(scope="session", autouse=True)
def manage_test_base_dir():
    temp_dir = "./pytest_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    pytest.test_base_dir = temp_dir
    yield
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_config(manage_test_base_dir):
    test_base_dir = pytest.test_base_dir

    os.environ["MSKIO_APP_WATCH_DIRECTORY"] = os.path.join(test_base_dir, "watch")
    os.environ["MSKIO_APP_OUTPUT_DIRECTORY"] = os.path.join(test_base_dir, "output")
    os.environ["MSKIO_RETRIEVAL_DATA_DOWNLOAD_DIR"] = os.path.join(test_base_dir, "downloads")
    os.environ["MSKIO_IMG_IMAGE_CACHE_DIR"] = os.path.join(test_base_dir, "img_cache")
    os.environ["MSKIO_INDEXER_VECTOR_DB_PATH"] = os.path.join(test_base_dir, "vectordb_mock.db")
    os.environ["MSKIO_LLM_OPENAI_API_KEY"] = "sk-test-openai-key"
    os.environ["MSKIO_LLM_GOOGLE_API_KEY"] = "test-google-key"
    os.environ["MSKIO_IMG_DEFAULT_SEGMENTATION_MODEL_PATH"] = os.path.join(test_base_dir, "dummy_model.pth")
    os.environ["MSKIO_APP_LOG_FILE_PATH"] = os.path.join(test_base_dir, "test_msk_io.log")

    with open(os.environ["MSKIO_IMG_DEFAULT_SEGMENTATION_MODEL_PATH"], "w") as f:
        f.write("DUMMY MODEL CONTENT")

    config = load_config()
    yield config

    keys_to_clear = [
        "MSKIO_APP_WATCH_DIRECTORY",
        "MSKIO_APP_OUTPUT_DIRECTORY",
        "MSKIO_RETRIEVAL_DATA_DOWNLOAD_DIR",
        "MSKIO_IMG_IMAGE_CACHE_DIR",
        "MSKIO_INDEXER_VECTOR_DB_PATH",
        "MSKIO_LLM_OPENAI_API_KEY",
        "MSKIO_LLM_GOOGLE_API_KEY",
        "MSKIO_IMG_DEFAULT_SEGMENTATION_MODEL_PATH",
        "MSKIO_APP_LOG_FILE_PATH",
    ]
    for key in keys_to_clear:
        if key in os.environ:
            del os.environ[key]


@pytest.fixture
def mock_dicom_volume(test_config) -> DICOMVolume:
    patient_info = DICOMPatientInfo(
        patient_id="P123",
        patient_name="Doe^John",
        patient_sex="M",
        patient_birth_date=date(1980, 1, 1),
    )
    study_info = DICOMStudyInfo(
        study_instance_uid="1.2.3.4.5.6.7",
        study_id="STUDY001",
        study_date=date(2023, 1, 15),
    )
    series_info = DICOMSeriesInfo(
        series_instance_uid="1.2.3.4.5.6.7.8",
        series_number=1,
        modality="CT",
        series_description="Brain CT",
    )
    temp_nifti_path = os.path.join(pytest.test_base_dir, "dummy_volume.nii.gz")
    with open(temp_nifti_path, "w") as f:
        f.write("DUMMY NIFTI CONTENT")

    return DICOMVolume(
        series_instance_uid="1.2.3.4.5.6.7.8",
        dicom_files=["/path/to/dicom1.dcm", "/path/to/dicom2.dcm"],
        volume_path=temp_nifti_path,
        original_modality="CT",
        patient_info=patient_info,
        study_info=study_info,
        series_info=series_info,
        volume_shape=[128, 256, 256],
        voxel_spacing=[1.0, 0.5, 0.5],
    )


@pytest.fixture(scope="session", autouse=True)
def create_sample_data_dir():
    sample_data_dir = "./examples/sample_data"
    os.makedirs(sample_data_dir, exist_ok=True)
    try:
        filename = os.path.join(sample_data_dir, "sample_dicom.dcm")
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = uid.CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = uid.generate_uid()
        file_meta.TransferSyntaxUID = uid.ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = uid.generate_uid()
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientID = "TestPatient"
        ds.PatientName = "Test^Patient"
        ds.StudyInstanceUID = uid.generate_uid()
        ds.SeriesInstanceUID = uid.generate_uid()
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.StudyDate = "20230101"
        ds.Rows = 128
        ds.Columns = 128
        ds.PixelSpacing = [0.5, 0.5]
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.StudyTime = "120000"
        ds.PatientSex = "O"
        ds.PatientBirthDate = "19900101"
        ds.save_as(filename)
        print(f"Created dummy DICOM file for testing at: {filename}")
    except ImportError:
        print("Pydicom not installed. Cannot create dummy DICOM file. Some tests may fail.")
        with open(os.path.join(sample_data_dir, "sample_dicom.dcm"), "w") as f:
            f.write("DUMMY DICOM CONTENT (No Pydicom)")
    yield
    shutil.rmtree(sample_data_dir, ignore_errors=True)
