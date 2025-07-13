"""
Medical Image Analysis Pipeline - Segmentation Module
This module handles image segmentation using edge detection and morphological operations.
"""

import numpy as np
import skimage.feature
import skimage.morphology
from .config import MODALITY, MODALITY_PARAMS


def segment_structures(image_np: np.ndarray) -> np.ndarray:
    """Simulates segmentation using edge detection and morphological operations."""
    print("2.1 Detecting edges (Canny)...")
    params = MODALITY_PARAMS[MODALITY]
    sigma = params["canny_sigma"]
    edges = skimage.feature.canny(image_np, sigma=sigma)

    print("2.2 Simulating segmentation with AI model (U-Net placeholder)...")
    mask_ai = create_simulated_roi(image_np)

    segmented_mask = mask_ai

    print("2.3 Refining segmentation (morphological operations)...")
    segmented_mask = apply_morphological_operations(segmented_mask)

    return segmented_mask


def create_simulated_roi(image_np: np.ndarray) -> np.ndarray:
    """Creates a simulated region of interest (ROI) for testing."""
    params = MODALITY_PARAMS[MODALITY]
    roi_radius_factor = params["roi_radius_factor"]

    mask_ai = np.zeros_like(image_np, dtype=bool)
    center_x, center_y = image_np.shape[1] // 2, image_np.shape[0] // 2
    radius = min(image_np.shape) // roi_radius_factor

    Y, X = np.ogrid[:image_np.shape[0], :image_np.shape[1]]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask_ai[dist_from_center <= radius] = True

    return mask_ai


def apply_morphological_operations(mask: np.ndarray) -> np.ndarray:
    """Applies morphological operations to refine the segmentation mask."""
    mask = skimage.morphology.binary_closing(mask, skimage.morphology.disk(3))
    mask = skimage.morphology.remove_small_objects(mask, min_size=50)
    mask = skimage.morphology.binary_erosion(mask, skimage.morphology.disk(1))
    return mask


def validate_segmentation(mask: np.ndarray) -> bool:
    """Validates the segmentation mask for basic quality checks."""
    if np.sum(mask) == 0:
        print("Warning: Segmentation mask is empty")
        return False

    total_pixels = mask.size
    segmented_pixels = np.sum(mask)

    if segmented_pixels / total_pixels > 0.9:
        print("Warning: Segmentation mask covers >90% of image")
        return False
    return True


def get_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Extracts the largest connected component from the segmentation mask."""
    import skimage.measure
    labeled_mask = skimage.measure.label(mask)
    if labeled_mask.max() == 0:
        return mask
    component_sizes = np.bincount(labeled_mask.flat)
    component_sizes[0] = 0
    largest_component_label = component_sizes.argmax()
    largest_component_mask = (labeled_mask == largest_component_label)
    return largest_component_mask
