import os
import shutil
from typing import List, Dict, Any, Optional
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from msk_io.schema.dicom_data import (
    DICOMVolume,
    DICOMPatientInfo,
    DICOMStudyInfo,
    DICOMSeriesInfo,
)
from msk_io.schema.retrieval_info import RetrievedDataInfo, DataSource
from msk_io.errors import RetrievalError, DataValidationError, ExternalServiceError
from msk_io.utils.log_config import get_logger
from msk_io.utils.decorators import handle_errors, log_method_entry_exit
from datetime import datetime
import httpx

logger = get_logger(__name__)


class DICOMStreamSniffer:
    """
    Simulates sniffing and retrieving DICOM studies from a source.

    In a real-world scenario, this would involve complex network sniffing,
    DICOMweb queries, or PACS integrations. For this implementation, it
    simulates retrieval from a local directory or a predefined URL structure
    and converts found DICOM series into standardized volumes.
    """

    def __init__(self, config):
        """Initializes the DICOMStreamSniffer with application configuration."""
        self.config = config
        self.base_url = config.retrieval.dicom_sniff_url
        self.download_dir = config.retrieval.data_download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        logger.info(
            f"DICOM Sniffer initialized. Base URL: {self.base_url}, Download Dir: {self.download_dir}"
        )

    @handle_errors
    @log_method_entry_exit
    async def discover_and_retrieve_studies(
        self,
        patient_id: Optional[str] = None,
        study_uid: Optional[str] = None,
        local_dicom_path: Optional[str] = None,
        remote_dicom_url: Optional[str] = None,
    ) -> RetrievedDataInfo:
        """Discovers and retrieves DICOM studies."""
        logger.warning(
            "DICOMStreamSniffer.discover_and_retrieve_studies is a conceptual stub. Prioritizing remote_dicom_url, then local_dicom_path, then dummy."
        )
        start_time = datetime.now()

        source_dicom_path = None
        retrieved_file_path = None
        source_type_str = "Local_Filesystem"

        if remote_dicom_url:
            logger.info(f"Attempting to download DICOM from URL: {remote_dicom_url}")
            try:
                async with httpx.AsyncClient(
                    timeout=self.config.retrieval.dicom_sniff_url_timeout_seconds
                ) as client:
                    response = await client.get(remote_dicom_url)
                    response.raise_for_status()

                    filename = os.path.basename(remote_dicom_url.split("?")[0])
                    if not filename:
                        filename = f"downloaded_dicom_{datetime.now().strftime('%Y%m%d%H%M%S')}.dcm"
                    elif not any(
                        filename.lower().endswith(ext) for ext in [".dcm", ".dicom"]
                    ):
                        filename += ".dcm"

                    retrieved_file_path = os.path.join(self.download_dir, filename)
                    with open(retrieved_file_path, "wb") as f:
                        f.write(response.content)
                    logger.info(
                        f"Downloaded DICOM from {remote_dicom_url} to {retrieved_file_path}"
                    )
                    source_dicom_path = remote_dicom_url
                    source_type_str = "Remote_URL"
            except httpx.RequestError as e:
                raise ExternalServiceError(
                    f"Failed to download DICOM from {remote_dicom_url}: Network or request error: {e}"
                ) from e
            except httpx.HTTPStatusError as e:
                raise ExternalServiceError(
                    f"Failed to download DICOM from {remote_dicom_url}: HTTP error {e.response.status_code} - {e.response.text}"
                ) from e
            except Exception as e:
                raise RetrievalError(
                    f"An unexpected error occurred during remote DICOM download from {remote_dicom_url}: {e}"
                ) from e
        elif local_dicom_path:
            source_dicom_path = local_dicom_path
            if not os.path.exists(source_dicom_path):
                raise RetrievalError(
                    f"Provided local DICOM file not found: {source_dicom_path}"
                )
            logger.info(f"Using provided local DICOM file: {source_dicom_path}")

            retrieved_file_path = os.path.join(
                self.download_dir, os.path.basename(source_dicom_path)
            )
            try:
                shutil.copy(source_dicom_path, retrieved_file_path)
                logger.info(
                    f"Copied DICOM file to download directory: {retrieved_file_path}"
                )
            except Exception as e:
                raise RetrievalError(
                    f"Failed to copy DICOM file from {source_dicom_path} to {retrieved_file_path}: {e}"
                ) from e
        else:
            source_dicom_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "examples",
                "sample_data",
                "sample_dicom.dcm",
            )
            if not os.path.exists(source_dicom_path):
                raise RetrievalError(
                    f"Dummy DICOM file not found at {source_dicom_path}. Cannot simulate retrieval."
                )
            logger.info(f"Using dummy DICOM file: {source_dicom_path}")

            retrieved_file_path = os.path.join(
                self.download_dir, os.path.basename(source_dicom_path)
            )
            try:
                shutil.copy(source_dicom_path, retrieved_file_path)
                logger.info(
                    f"Copied dummy DICOM file to download directory: {retrieved_file_path}"
                )
            except Exception as e:
                raise RetrievalError(
                    f"Failed to copy dummy DICOM file from {source_dicom_path} to {retrieved_file_path}: {e}"
                ) from e

        if retrieved_file_path is None or not os.path.exists(retrieved_file_path):
            raise RetrievalError(
                "No DICOM file could be retrieved or found for processing."
            )

        dicom_volume = None
        try:
            ds = dcmread(retrieved_file_path, stop_before_pixels=True)
            patient_id_val = getattr(ds, "PatientID", "UNKNOWN_PATIENT_ID")
            patient_name_val = str(getattr(ds, "PatientName", "UNKNOWN^PATIENT"))
            patient_sex_val = getattr(ds, "PatientSex", None)
            patient_birth_date_val = (
                datetime.strptime(ds.PatientBirthDate, "%Y%m%d").date()
                if "PatientBirthDate" in ds and ds.PatientBirthDate
                else None
            )

            study_instance_uid_val = getattr(
                ds, "StudyInstanceUID", "UNKNOWN_STUDY_UID"
            )
            study_id_val = getattr(ds, "StudyID", "UNKNOWN")
            study_date_val = (
                datetime.strptime(ds.StudyDate, "%Y%m%d").date()
                if "StudyDate" in ds and ds.StudyDate
                else None
            )
            study_time_val = (
                datetime.strptime(ds.StudyTime, "%H%M%S").time()
                if "StudyTime" in ds and ds.StudyTime
                else None
            )

            series_instance_uid_val = getattr(
                ds, "SeriesInstanceUID", "UNKNOWN_SERIES_UID"
            )
            series_number_val = getattr(ds, "SeriesNumber", None)
            modality_val = getattr(ds, "Modality", "UNKNOWN")

            patient_info = DICOMPatientInfo(
                patient_id=patient_id_val,
                patient_name=patient_name_val,
                patient_sex=patient_sex_val,
                patient_birth_date=patient_birth_date_val,
            )
            study_info = DICOMStudyInfo(
                study_instance_uid=study_instance_uid_val,
                study_id=study_id_val,
                study_description=getattr(ds, "StudyDescription", None),
                study_date=study_date_val,
                study_time=study_time_val,
                accession_number=getattr(ds, "AccessionNumber", None),
                referring_physician_name=getattr(ds, "ReferringPhysicianName", None),
            )
            series_info = DICOMSeriesInfo(
                series_instance_uid=series_instance_uid_val,
                series_number=series_number_val,
                series_description=getattr(ds, "SeriesDescription", None),
                modality=modality_val,
                body_part_examined=getattr(ds, "BodyPartExamined", None),
                protocol_name=getattr(ds, "ProtocolName", None),
            )

            dummy_volume_path = os.path.join(
                self.download_dir, f"{series_info.series_instance_uid}.nii.gz"
            )
            with open(dummy_volume_path, "w") as f:
                f.write(f"DUMMY NIFTI CONTENT FOR {series_info.series_instance_uid}")
            logger.debug(f"Simulated NIfTI volume creation at {dummy_volume_path}")

            rows = getattr(ds, "Rows", 1)
            columns = getattr(ds, "Columns", 1)
            pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])

            dicom_volume = DICOMVolume(
                series_instance_uid=series_info.series_instance_uid,
                dicom_files=[retrieved_file_path],
                volume_path=dummy_volume_path,
                original_modality=series_info.modality,
                patient_info=patient_info,
                study_info=study_info,
                series_info=series_info,
                volume_shape=[1, rows, columns],
                voxel_spacing=[1.0, pixel_spacing[0], pixel_spacing[1]],
            )

        except InvalidDicomError as e:
            raise DataValidationError(
                f"Retrieved file is not a valid DICOM: {e}"
            ) from e
        except Exception as e:
            raise RetrievalError(
                f"Failed to process retrieved DICOM metadata from {retrieved_file_path}: {e}"
            ) from e

        end_time = datetime.now()
        data_source = DataSource(
            source_id="retrieved-data",
            source_type=source_type_str,
            endpoint_url=source_dicom_path,
            access_method="Direct Download" if remote_dicom_url else "Local File Copy",
            last_accessed=end_time,
        )

        retrieved_info = RetrievedDataInfo(
            data_source=data_source,
            original_query=patient_id
            or study_uid
            or (remote_dicom_url or os.path.basename(source_dicom_path)),
            retrieved_file_paths=[retrieved_file_path],
            total_files_retrieved=1,
            total_size_bytes=os.path.getsize(retrieved_file_path),
            retrieval_start_time=start_time,
            retrieval_end_time=end_time,
        )
        if dicom_volume:
            retrieved_info.series_volumes = [dicom_volume]
            retrieved_info.studies = [study_info]

        return retrieved_info
