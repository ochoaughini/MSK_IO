# Medical Image Analysis Pipeline
A comprehensive Python pipeline for automated medical image analysis, supporting both CT and MRI modalities with segmentation, feature extraction, classification, and automated reporting capabilities.

## Features
- **Multi-modal Support:** Optimized processing for both CT and MRI images
- **Automated Segmentation:** Canny edge detection with morphological refinement
- **Comprehensive Feature Extraction:** Geometric, intensity, and texture features
- **AI-Assisted Classification:** Rule-based classification with confidence scoring
- **Automated Reporting:** Detailed reports with visualizations and clinical recommendations
- **Batch Processing:** Support for analyzing multiple images

## Installation
Clone or download the pipeline files and install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start
### Basic Usage
```bash
python main.py input_image.png
```
### Specify Modality
```bash
python main.py input_image.png --modality CT
python main.py input_image.png --modality MRI
```
### Custom Output Directory
```bash
python main.py input_image.png --output-dir /path/to/results
```
### Verbose Output
```bash
python main.py input_image.png --verbose
```

## Pipeline Architecture
The pipeline consists of five main stages:
1. **Image Preprocessing (`preprocessing.py`)**
   - Grayscale conversion
   - Modality-specific intensity normalization
   - Image resizing and quality validation
2. **Segmentation (`segmentation.py`)**
   - Canny edge detection
   - Morphological operations
   - Connected component analysis
3. **Feature Extraction (`feature_extraction.py`)**
   - Geometric features
   - Intensity features
   - Texture features using GLCM
4. **Classification (`classification.py`)**
   - Rule-based classification with confidence scoring
   - Risk level assessment and clinical recommendations
5. **Reporting (`reporting.py`)**
   - Comprehensive text reports
   - Annotated visualizations
   - Feature analysis plots

## Configuration
Edit `config.py` to customize default settings such as `MODALITY`, `TARGET_IMAGE_SIZE`, `OUTPUT_DIRECTORY`, and modality-specific parameters.

## Usage Examples
See `examples.py` for programmatic usage and demonstration scripts.

## License
This project is provided for educational and research purposes. Results should be validated by qualified medical professionals.
