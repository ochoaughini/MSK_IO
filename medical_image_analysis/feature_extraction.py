"""
Medical Image Analysis Pipeline - Feature Extraction Module
This module extracts quantitative features from segmented regions.
"""

import numpy as np
import skimage.feature
import skimage.measure
from .config import MODALITY, MODALITY_PARAMS


def extract_features(image_np: np.ndarray, mask_np: np.ndarray) -> dict:
    """Extracts quantitative features from the segmented region."""
    print("3.1 Measuring Area and Perimeter...")

    features = {
        "total_area_pixels": 0,
        "total_perimeter_pixels": 0,
        "mean_intensity": 0.0,
        "intensity_std_dev": 0.0,
        "texture_contrast": 0.0,
        "texture_dissimilarity": 0.0,
        "num_segmented_regions": 0,
        "compactness": 0.0,
        "eccentricity": 0.0,
        "solidity": 0.0
    }

    mask_bool = mask_np > 0
    labeled_mask = skimage.measure.label(mask_bool)
    regions = skimage.measure.regionprops(labeled_mask)

    features["num_segmented_regions"] = len(regions)

    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        features["total_area_pixels"] = largest_region.area
        features["total_perimeter_pixels"] = largest_region.perimeter
        features["compactness"] = _calculate_compactness(largest_region)
        features["eccentricity"] = largest_region.eccentricity
        features["solidity"] = largest_region.solidity
        features.update(_extract_intensity_features(image_np, mask_bool))
        features.update(_extract_texture_features(image_np, largest_region))

    return features


def _calculate_compactness(region) -> float:
    """Calculates the compactness of a region (4π * area / perimeter²)."""
    if region.perimeter == 0:
        return 0.0
    return (4 * np.pi * region.area) / (region.perimeter ** 2)


def _extract_intensity_features(image_np: np.ndarray, mask_bool: np.ndarray) -> dict:
    """Extracts intensity-based features from the segmented region."""
    print("3.2 Calculating Mean and Standard Deviation of Intensity...")

    features = {
        "mean_intensity": 0.0,
        "intensity_std_dev": 0.0,
        "min_intensity": 0.0,
        "max_intensity": 0.0,
        "intensity_range": 0.0
    }

    intensities_in_mask = image_np[mask_bool]

    if intensities_in_mask.size > 0:
        features["mean_intensity"] = np.mean(intensities_in_mask)
        features["intensity_std_dev"] = np.std(intensities_in_mask)
        features["min_intensity"] = np.min(intensities_in_mask)
        features["max_intensity"] = np.max(intensities_in_mask)
        features["intensity_range"] = features["max_intensity"] - features["min_intensity"]

    return features


def _extract_texture_features(image_np: np.ndarray, region) -> dict:
    """Extracts texture features using Gray-Level Co-occurrence Matrix (GLCM)."""
    print("3.3 Analyzing Texture (GLCM)...")

    features = {
        "texture_contrast": 0.0,
        "texture_dissimilarity": 0.0,
        "texture_homogeneity": 0.0,
        "texture_energy": 0.0,
        "texture_correlation": 0.0
    }

    params = MODALITY_PARAMS[MODALITY]
    levels = params["glcm_levels"]

    try:
        min_row, min_col, max_row, max_col = region.bbox
        sub_image = image_np[min_row:max_row, min_col:max_col]

        if image_np.dtype != np.uint8:
            image_glcm = (sub_image / sub_image.max() * 255).astype(np.uint8)
        else:
            image_glcm = sub_image

        glcm = skimage.feature.greycomatrix(
            image_glcm,
            distances=[1],
            angles=[0],
            levels=levels,
            symmetric=True,
            normed=True
        )

        features["texture_contrast"] = skimage.feature.greycoprops(glcm, 'contrast')[0, 0]
        features["texture_dissimilarity"] = skimage.feature.greycoprops(glcm, 'dissimilarity')[0, 0]
        features["texture_homogeneity"] = skimage.feature.greycoprops(glcm, 'homogeneity')[0, 0]
        features["texture_energy"] = skimage.feature.greycoprops(glcm, 'energy')[0, 0]
        features["texture_correlation"] = skimage.feature.greycoprops(glcm, 'correlation')[0, 0]

    except (ValueError, IndexError) as e:
        print(f"Warning: Unable to compute GLCM. {e}. Setting texture features to 0.")

    return features


def validate_features(features: dict) -> bool:
    """Validates extracted features for reasonable values."""
    non_negative_features = [
        "total_area_pixels", "total_perimeter_pixels", "intensity_std_dev",
        "texture_contrast", "texture_dissimilarity", "texture_homogeneity",
        "texture_energy"
    ]

    for feature_name in non_negative_features:
        if feature_name in features and features[feature_name] < 0:
            print(f"Warning: {feature_name} has negative value: {features[feature_name]}")
            return False

    if features.get("compactness", 0) > 1.0:
        print(f"Warning: Compactness > 1.0: {features['compactness']}")
        return False

    if features.get("eccentricity", 0) > 1.0:
        print(f"Warning: Eccentricity > 1.0: {features['eccentricity']}")
        return False

    return True


def get_feature_summary(features: dict) -> str:
    """Creates a summary string of the extracted features."""
    summary = f"""
Feature Summary:
- Area: {features['total_area_pixels']:.0f} pixels
- Perimeter: {features['total_perimeter_pixels']:.2f} pixels
- Mean Intensity: {features['mean_intensity']:.2f}
- Intensity Std Dev: {features['intensity_std_dev']:.2f}
- Compactness: {features['compactness']:.3f}
- Eccentricity: {features['eccentricity']:.3f}
- Texture Contrast: {features['texture_contrast']:.4f}
- Texture Homogeneity: {features['texture_homogeneity']:.4f}
"""
    return summary
