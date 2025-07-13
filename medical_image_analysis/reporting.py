"""
Medical Image Analysis Pipeline - Reporting Module
This module generates comprehensive analysis reports with visualizations.
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .config import MODALITY, OUTPUT_DIRECTORY


def generate_report(original_image: Image.Image, processed_image_np: np.ndarray,
                   segmented_mask_np: np.ndarray, features: dict,
                   classification_result: dict, output_directory: str = None) -> str:
    """Generates an automated report with extracted features and visualizations."""
    if output_directory is None:
        output_directory = OUTPUT_DIRECTORY

    os.makedirs(output_directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    report_filename = _generate_text_report(
        original_image, processed_image_np, segmented_mask_np,
        features, classification_result, output_directory, timestamp
    )

    _generate_annotated_image(
        processed_image_np, segmented_mask_np, output_directory, timestamp
    )

    _generate_feature_plots(features, output_directory, timestamp)

    print(f"Report generated and saved at: {report_filename}")
    return report_filename


def _generate_text_report(original_image: Image.Image, processed_image_np: np.ndarray,
                         segmented_mask_np: np.ndarray, features: dict,
                         classification_result: dict, output_directory: str,
                         timestamp: str) -> str:
    """Generates the main text report file."""
    report_filename = os.path.join(output_directory, f"analysis_report_{timestamp}.txt")

    with open(report_filename, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write("MEDICAL IMAGE ANALYSIS REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"Generated: {timestamp} ({MODALITY} Analysis)\n")
        f.write(f"{'='*60}\n\n")

        f.write("1. IMAGE PROCESSING SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"   Original Image Size: {original_image.size}\n")
        f.write(f"   Processed Image Size: {processed_image_np.shape[1]}x{processed_image_np.shape[0]}\n")
        f.write(f"   Imaging Modality: {MODALITY}\n")
        f.write("   Processing Steps: Grayscale conversion, intensity normalization, resizing\n")
        f.write(f"   Segmentation Quality: {_assess_segmentation_quality(segmented_mask_np)}\n\n")

        f.write("2. SEGMENTATION RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"   Number of Detected Regions: {features.get('num_segmented_regions', 0)}\n")
        f.write("   Segmentation Method: Canny edge detection + morphological operations\n")
        f.write(f"   Primary ROI Area: {features.get('total_area_pixels', 0):.0f} pixels\n")
        f.write(f"   ROI Coverage: {_calculate_roi_coverage(segmented_mask_np):.1%} of image\n")
        f.write(f"   Annotated image: annotated_image_{timestamp}.png\n\n")

        f.write("3. QUANTITATIVE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write("   Geometric Features:\n")
        f.write(f"     - Area: {features.get('total_area_pixels', 0):.0f} pixels\n")
        f.write(f"     - Perimeter: {features.get('total_perimeter_pixels', 0):.2f} pixels\n")
        f.write(f"     - Compactness: {features.get('compactness', 0):.3f}\n")
        f.write(f"     - Eccentricity: {features.get('eccentricity', 0):.3f}\n")
        f.write(f"     - Solidity: {features.get('solidity', 0):.3f}\n\n")

        f.write("   Intensity Features:\n")
        f.write(f"     - Mean Intensity: {features.get('mean_intensity', 0):.2f}\n")
        f.write(f"     - Intensity Std Dev: {features.get('intensity_std_dev', 0):.2f}\n")
        f.write(f"     - Intensity Range: {features.get('intensity_range', 0):.2f}\n\n")

        f.write("   Texture Features:\n")
        f.write(f"     - Contrast: {features.get('texture_contrast', 0):.4f}\n")
        f.write(f"     - Dissimilarity: {features.get('texture_dissimilarity', 0):.4f}\n")
        f.write(f"     - Homogeneity: {features.get('texture_homogeneity', 0):.4f}\n")
        f.write(f"     - Energy: {features.get('texture_energy', 0):.4f}\n\n")

        f.write("4. CLASSIFICATION AND DIAGNOSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"   Primary Classification: {classification_result.get('primary_classification', 'Unknown')}\n")
        f.write(f"   Confidence Score: {classification_result.get('confidence_score', 0):.1%}\n")
        f.write(f"   Risk Level: {classification_result.get('risk_level', 'Unknown')}\n\n")

        f.write("5. CLINICAL RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        recommendations = classification_result.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                f.write(f"   {i}. {rec}\n")
        else:
            f.write("   No specific recommendations generated.\n")
        f.write("\n")

        f.write("6. AI ASSISTANT INSIGHTS\n")
        f.write("-" * 30 + "\n")
        from .classification import integrate_ai_assistance
        ai_query = f"Explain the clinical implications of {classification_result.get('primary_classification', 'Unknown')}"
        ai_response = integrate_ai_assistance(ai_query, classification_result)
        f.write(f"   {ai_response}\n\n")

        f.write("7. TECHNICAL NOTES\n")
        f.write("-" * 30 + "\n")
        f.write("   - This analysis is for research/educational purposes only\n")
        f.write("   - Results should be validated by qualified medical professionals\n")
        f.write("   - Algorithm performance may vary with different image qualities\n")
        f.write(f"   - Modality-specific parameters used for {MODALITY} analysis\n\n")

        f.write("=" * 60 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")

    return report_filename


def _generate_annotated_image(processed_image_np: np.ndarray, segmented_mask_np: np.ndarray,
                             output_directory: str, timestamp: str) -> None:
    """Generates an annotated image with segmentation overlay."""
    annotated_image_filename = os.path.join(output_directory, f"annotated_image_{timestamp}.png")

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(processed_image_np, cmap='gray')
    plt.title(f"Original {MODALITY} Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed_image_np, cmap='gray')
    mask_overlay = np.zeros((*segmented_mask_np.shape, 4), dtype=np.uint8)
    mask_overlay[segmented_mask_np > 0] = [255, 0, 0, 120]
    plt.imshow(mask_overlay)
    plt.title(f"Segmented Regions ({MODALITY})")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(annotated_image_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Annotated image saved: {os.path.basename(annotated_image_filename)}")


def _generate_feature_plots(features: dict, output_directory: str, timestamp: str) -> None:
    """Generates visualization plots for extracted features."""
    plot_filename = os.path.join(output_directory, f"feature_plots_{timestamp}.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    geometric_features = ['total_area_pixels', 'total_perimeter_pixels', 'compactness', 'eccentricity']
    geometric_values = [features.get(f, 0) for f in geometric_features]
    geometric_labels = ['Area', 'Perimeter', 'Compactness', 'Eccentricity']
    axes[0, 0].bar(geometric_labels, geometric_values)
    axes[0, 0].set_title('Geometric Features')
    axes[0, 0].tick_params(axis='x', rotation=45)

    intensity_features = ['mean_intensity', 'intensity_std_dev', 'min_intensity', 'max_intensity']
    intensity_values = [features.get(f, 0) for f in intensity_features]
    intensity_labels = ['Mean', 'Std Dev', 'Min', 'Max']
    axes[0, 1].bar(intensity_labels, intensity_values)
    axes[0, 1].set_title('Intensity Features')

    texture_features = ['texture_contrast', 'texture_dissimilarity', 'texture_homogeneity', 'texture_energy']
    texture_values = [features.get(f, 0) for f in texture_features]
    texture_labels = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy']
    axes[1, 0].bar(texture_labels, texture_values)
    axes[1, 0].set_title('Texture Features')
    axes[1, 0].tick_params(axis='x', rotation=45)

    axes[1, 1].text(0.5, 0.5, f"Feature Summary\n\n"
                                f"Total Features: {len(features)}\n"
                                f"Area: {features.get('total_area_pixels', 0):.0f}\n"
                                f"Mean Intensity: {features.get('mean_intensity', 0):.1f}\n"
                                f"Texture Contrast: {features.get('texture_contrast', 0):.3f}\n"
                                f"Compactness: {features.get('compactness', 0):.3f}",
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary Statistics')

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Feature plots saved: {os.path.basename(plot_filename)}")


def _assess_segmentation_quality(mask: np.ndarray) -> str:
    """Assesses the quality of segmentation based on basic metrics."""
    total_pixels = mask.size
    segmented_pixels = np.sum(mask > 0)
    coverage = segmented_pixels / total_pixels

    if coverage < 0.01:
        return "Poor (very small ROI)"
    elif coverage < 0.1:
        return "Fair (small ROI)"
    elif coverage < 0.5:
        return "Good (moderate ROI)"
    elif coverage < 0.9:
        return "Excellent (large ROI)"
    else:
        return "Questionable (too large ROI)"


def _calculate_roi_coverage(mask: np.ndarray) -> float:
    """Calculates the percentage of image covered by the ROI."""
    return np.sum(mask > 0) / mask.size


def create_summary_report(features: dict, classification_result: dict) -> str:
    """Creates a brief summary report for quick review."""
    summary = f"""
QUICK ANALYSIS SUMMARY ({MODALITY})
{'='*40}

Classification: {classification_result.get('primary_classification', 'Unknown')}
Confidence: {classification_result.get('confidence_score', 0):.1%}
Risk Level: {classification_result.get('risk_level', 'Unknown')}

Key Measurements:
- Area: {features.get('total_area_pixels', 0):.0f} pixels
- Mean Intensity: {features.get('mean_intensity', 0):.1f}
- Texture Contrast: {features.get('texture_contrast', 0):.3f}

Primary Recommendation: {classification_result.get('recommendations', ['None available'])[0]}

{'='*40}
"""
    return summary
