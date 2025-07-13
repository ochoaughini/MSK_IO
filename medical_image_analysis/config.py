# Configuration file for Medical Image Analysis Pipeline
# Requirements: Pillow, numpy, scikit-image, matplotlib

# --- Configuration Settings ---
TARGET_IMAGE_SIZE = (256, 256)  # Default size for resizing
OUTPUT_DIRECTORY = "analysis_results"
MODALITY = "CT"  # Can be changed to "MRI" to simulate MRI

# Modality-specific parameters
MODALITY_PARAMS = {
    "CT": {
        "normalization_range": (0, 255),
        "canny_sigma": 3,
        "roi_radius_factor": 4,  # divisor for image dimension
        "glcm_levels": 256,
        "classification_area_threshold": 5000,
        "classification_intensity_threshold": 100,
        "benign_area_threshold": 1000,
        "benign_intensity_threshold": 150
    },
    "MRI": {
        "normalization_range": (55, 255),
        "canny_sigma": 2,
        "roi_radius_factor": 3,
        "glcm_levels": 128,
        "classification_area_threshold": 4000,
        "classification_intensity_threshold": 120,
        "benign_area_threshold": 1200,
        "benign_intensity_threshold": 180
    }
}

# Default file paths
DEFAULT_INPUT_IMAGE = "example_input_image.png"
DEFAULT_OUTPUT_DIR = "analysis_results"
