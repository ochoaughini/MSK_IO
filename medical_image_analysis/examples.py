#!/usr/bin/env python3
"""
Medical Image Analysis Pipeline - Example Usage
This script demonstrates various ways to use the medical image analysis pipeline.
"""

import os
import sys
from .main import execute_image_analysis_pipeline, run_batch_analysis
from .reporting import create_summary_report
from . import config


def example_basic_usage():
    """Example 1: Basic pipeline usage with default settings"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Pipeline Usage")
    print("=" * 60)
    image_path = "example_input_image.png"
    try:
        results = execute_image_analysis_pipeline(image_path)
        print("\nKey Results:")
        print(f"Classification: {results['classification']['primary_classification']}")
        print(f"Confidence: {results['classification']['confidence_score']:.1%}")
        print(f"Risk Level: {results['classification']['risk_level']}")
        print(f"Report Path: {results['report_path']}")
        return results
    except Exception as e:
        print(f"Error in basic usage example: {e}")
        return None


def example_modality_comparison():
    """Example 2: Compare CT vs MRI analysis on the same image"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: CT vs MRI Modality Comparison")
    print("=" * 60)
    image_path = "example_input_image.png"
    results = {}
    print("\nAnalyzing with CT settings...")
    config.MODALITY = "CT"
    try:
        ct_results = execute_image_analysis_pipeline(image_path, "ct_results")
        results['CT'] = ct_results
        print(f"CT Classification: {ct_results['classification']['primary_classification']}")
        print(f"CT Confidence: {ct_results['classification']['confidence_score']:.1%}")
    except Exception as e:
        print(f"CT analysis failed: {e}")
    print("\nAnalyzing with MRI settings...")
    config.MODALITY = "MRI"
    try:
        mri_results = execute_image_analysis_pipeline(image_path, "mri_results")
        results['MRI'] = mri_results
        print(f"MRI Classification: {mri_results['classification']['primary_classification']}")
        print(f"MRI Confidence: {mri_results['classification']['confidence_score']:.1%}")
    except Exception as e:
        print(f"MRI analysis failed: {e}")
    if len(results) == 2:
        print("\nComparison Summary:")
        print(f"CT vs MRI Classification: {results['CT']['classification']['primary_classification']} vs {results['MRI']['classification']['primary_classification']}")
        print(f"CT vs MRI Confidence: {results['CT']['classification']['confidence_score']:.1%} vs {results['MRI']['classification']['confidence_score']:.1%}")
    return results


def example_batch_processing():
    """Example 3: Batch processing multiple images"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 60)
    test_images = []
    for i in range(3):
        image_path = f"test_image_{i+1}.png"
        test_images.append(image_path)
        if not os.path.exists(image_path):
            from .preprocessing import create_fictional_image
            create_fictional_image(image_path)
    print(f"Processing {len(test_images)} images...")
    try:
        batch_results = run_batch_analysis(test_images, "batch_results")
        print("\nBatch Processing Summary:")
        for image_path, results in batch_results.items():
            if "error" not in results:
                classification = results['classification']['primary_classification']
                confidence = results['classification']['confidence_score']
                print(f"  {image_path}: {classification} ({confidence:.1%})")
            else:
                print(f"  {image_path}: ERROR - {results['error']}")
        return batch_results
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return None


def example_custom_configuration():
    """Example 4: Custom configuration and advanced usage"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Configuration")
    print("=" * 60)
    original_modality = config.MODALITY
    original_image_size = config.TARGET_IMAGE_SIZE
    try:
        config.MODALITY = "CT"
        config.TARGET_IMAGE_SIZE = (512, 512)
        config.MODALITY_PARAMS["CT"]["classification_area_threshold"] = 3000
        config.MODALITY_PARAMS["CT"]["canny_sigma"] = 2
        print("Running with custom configuration:")
        print(f"  Modality: {config.MODALITY}")
        print(f"  Image Size: {config.TARGET_IMAGE_SIZE}")
        print(f"  Area Threshold: {config.MODALITY_PARAMS['CT']['classification_area_threshold']}")
        results = execute_image_analysis_pipeline("example_input_image.png", "custom_results")
        print("\nCustom Configuration Results:")
        print(f"Classification: {results['classification']['primary_classification']}")
        print(f"Confidence: {results['classification']['confidence_score']:.1%}")
        return results
    except Exception as e:
        print(f"Custom configuration example failed: {e}")
        return None
    finally:
        config.MODALITY = original_modality
        config.TARGET_IMAGE_SIZE = original_image_size


def example_programmatic_usage():
    """Example 5: Programmatic usage with custom analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Programmatic Usage")
    print("=" * 60)
    try:
        from .preprocessing import load_and_preprocess_image
        from .segmentation import segment_structures
        from .feature_extraction import extract_features
        from .classification import classify_structures
        image_path = "example_input_image.png"
        print("Step 1: Preprocessing...")
        processed_img = load_and_preprocess_image(image_path)
        processed_array = np.array(processed_img)
        print("Step 2: Segmentation...")
        mask = segment_structures(processed_array)
        print("Step 3: Feature extraction...")
        features = extract_features(processed_array, mask)
        print("Step 4: Classification...")
        classification = classify_structures(features)
        print("\nCustom Analysis:")
        area_size = features.get('total_area_pixels', 0)
        if area_size > 5000:
            print(f"  Large lesion detected: {area_size} pixels")
        else:
            print(f"  Small lesion detected: {area_size} pixels")
        texture_contrast = features.get('texture_contrast', 0)
        if texture_contrast > 0.5:
            print(f"  High texture heterogeneity: {texture_contrast:.3f}")
        else:
            print(f"  Low texture heterogeneity: {texture_contrast:.3f}")
        summary = create_summary_report(features, classification)
        print(f"\nCustom Summary:\n{summary}")
        return {
            'processed_image': processed_array,
            'mask': mask,
            'features': features,
            'classification': classification
        }
    except Exception as e:
        print(f"Programmatic usage example failed: {e}")
        return None


def run_all_examples():
    """Run all examples in sequence"""
    print("Medical Image Analysis Pipeline - Example Usage")
    print("=" * 60)
    import numpy as np
    example_basic_usage()
    example_modality_comparison()
    example_batch_processing()
    example_custom_configuration()
    example_programmatic_usage()
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)
    print("\nGenerated files and directories:")
    for item in os.listdir('.'):
        if os.path.isdir(item) and ('results' in item or 'analysis' in item):
            print(f"  📁 {item}/")
        elif item.endswith('.png') and 'test' in item:
            print(f"  🖼️  {item}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_basic_usage()
        elif example_num == "2":
            example_modality_comparison()
        elif example_num == "3":
            example_batch_processing()
        elif example_num == "4":
            example_custom_configuration()
        elif example_num == "5":
            example_programmatic_usage()
        else:
            print("Usage: python examples.py [1|2|3|4|5]")
            print("  1: Basic usage")
            print("  2: Modality comparison")
            print("  3: Batch processing")
            print("  4: Custom configuration")
            print("  5: Programmatic usage")
    else:
        run_all_examples()
