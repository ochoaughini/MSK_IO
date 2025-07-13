#!/usr/bin/env python3
"""
Medical Image Analysis Pipeline - Main Script
This script orchestrates the complete medical image analysis workflow.
"""

import sys
import os
import argparse
import numpy as np
from PIL import Image

from .config import MODALITY, OUTPUT_DIRECTORY, DEFAULT_INPUT_IMAGE
from .preprocessing import load_and_preprocess_image, validate_image, create_fictional_image
from .segmentation import segment_structures, validate_segmentation, get_largest_connected_component
from .feature_extraction import extract_features, validate_features, get_feature_summary
from .classification import classify_structures, validate_classification
from .reporting import generate_report, create_summary_report


def execute_image_analysis_pipeline(image_path: str, output_dir: str = None) -> dict:
    """Executes the complete image analysis pipeline."""
    print(f"\n{'='*60}")
    print("MEDICAL IMAGE ANALYSIS PIPELINE")
    print(f"{'='*60}")
    print(f"Input Image: {image_path}")
    print(f"Modality: {MODALITY}")
    print(f"Output Directory: {output_dir or OUTPUT_DIRECTORY}")
    print(f"{'='*60}\n")

    try:
        print("STAGE 1: IMAGE PREPROCESSING")
        print("-" * 30)
        if not validate_image(image_path):
            print("Creating fictional image for testing...")
            create_fictional_image(image_path)
        original_img = Image.open(image_path)
        processed_img_pil = load_and_preprocess_image(image_path)
        processed_img_np = np.array(processed_img_pil)
        print("✓ Image preprocessing completed successfully\n")

        print("STAGE 2: IMAGE SEGMENTATION")
        print("-" * 30)
        segmented_mask = segment_structures(processed_img_np)
        if not validate_segmentation(segmented_mask):
            print("Warning: Segmentation quality may be poor")
        segmented_mask = get_largest_connected_component(segmented_mask)
        print("✓ Segmentation completed successfully\n")

        print("STAGE 3: FEATURE EXTRACTION")
        print("-" * 30)
        features = extract_features(processed_img_np, segmented_mask)
        if not validate_features(features):
            print("Warning: Some extracted features may be invalid")
        print(get_feature_summary(features))
        print("✓ Feature extraction completed successfully\n")

        print("STAGE 4: CLASSIFICATION AND DIAGNOSIS")
        print("-" * 30)
        classification_result = classify_structures(features)
        if not validate_classification(classification_result):
            print("Warning: Classification result may be invalid")
        print("✓ Classification completed successfully\n")

        print("STAGE 5: REPORT GENERATION")
        print("-" * 30)
        report_path = generate_report(
            original_img, processed_img_np, segmented_mask,
            features, classification_result, output_dir
        )
        summary = create_summary_report(features, classification_result)
        print(summary)
        print("✓ Report generation completed successfully\n")

        results = {
            'original_image': original_img,
            'processed_image': processed_img_np,
            'segmentation_mask': segmented_mask,
            'features': features,
            'classification': classification_result,
            'report_path': report_path,
            'summary': summary
        }

        print(f"{'='*60}")
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")

        return results

    except Exception as e:
        print(f"\n❌ ERROR: Pipeline execution failed: {str(e)}")
        print("Please check your input image and try again.")
        raise


def main():
    """Main function to run the medical image analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Medical Image Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py image.png
    python main.py image.png --modality MRI
    python main.py image.png --output-dir /path/to/output
    python main.py image.png --modality CT --output-dir results
        """
    )

    parser.add_argument('image_path', nargs='?', default=DEFAULT_INPUT_IMAGE,
                        help=f'Path to input image (default: {DEFAULT_INPUT_IMAGE})')
    parser.add_argument('--modality', choices=['CT', 'MRI'], default=MODALITY,
                        help=f'Imaging modality (default: {MODALITY})')
    parser.add_argument('--output-dir', default=OUTPUT_DIRECTORY,
                        help=f'Output directory for results (default: {OUTPUT_DIRECTORY})')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    import importlib
    config = importlib.import_module('.config', __package__)
    config.MODALITY = args.modality
    config.OUTPUT_DIRECTORY = args.output_dir

    if not args.verbose:
        import warnings
        warnings.filterwarnings('ignore')

    if not os.path.exists(args.image_path):
        print(f"Warning: Image file '{args.image_path}' not found.")
        print("A fictional image will be created for testing purposes.")

    try:
        results = execute_image_analysis_pipeline(args.image_path, args.output_dir)
        print("\nFINAL RESULTS:")
        print(f"Classification: {results['classification']['primary_classification']}")
        print(f"Confidence: {results['classification']['confidence_score']:.1%}")
        print(f"Risk Level: {results['classification']['risk_level']}")
        print(f"Report saved to: {results['report_path']}")
        print("\nFiles generated:")
        for filename in os.listdir(args.output_dir):
            print(f"  - {filename}")
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        sys.exit(1)


def run_batch_analysis(image_list: list, output_base_dir: str = "batch_results") -> dict:
    """Runs the pipeline on multiple images."""
    print(f"\n{'='*60}")
    print("BATCH IMAGE ANALYSIS")
    print(f"{'='*60}")
    print(f"Processing {len(image_list)} images...")
    print(f"Base output directory: {output_base_dir}")
    print(f"{'='*60}\n")

    batch_results = {}
    for i, image_path in enumerate(image_list, 1):
        print(f"\nProcessing image {i}/{len(image_list)}: {image_path}")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(output_base_dir, f"{image_name}_analysis")
        try:
            results = execute_image_analysis_pipeline(image_path, output_dir)
            batch_results[image_path] = results
            print(f"✓ Successfully processed: {image_path}")
        except Exception as e:
            print(f"❌ Failed to process {image_path}: {str(e)}")
            batch_results[image_path] = {"error": str(e)}

    _generate_batch_summary(batch_results, output_base_dir)
    return batch_results


def _generate_batch_summary(batch_results: dict, output_dir: str):
    """Generates a summary report for batch analysis."""
    summary_path = os.path.join(output_dir, "batch_summary.txt")
    os.makedirs(output_dir, exist_ok=True)

    with open(summary_path, 'w') as f:
        f.write("BATCH ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        successful = 0
        failed = 0
        for image_path, results in batch_results.items():
            if "error" in results:
                f.write(f"❌ FAILED: {image_path}\n")
                f.write(f"   Error: {results['error']}\n\n")
                failed += 1
            else:
                f.write(f"✓ SUCCESS: {image_path}\n")
                classification = results['classification']
                f.write(f"   Classification: {classification['primary_classification']}\n")
                f.write(f"   Confidence: {classification['confidence_score']:.1%}\n")
                f.write(f"   Risk Level: {classification['risk_level']}\n\n")
                successful += 1
        f.write("\nSUMMARY STATISTICS:\n")
        f.write(f"Total images processed: {len(batch_results)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success rate: {successful/len(batch_results)*100:.1f}%\n")
    print(f"Batch summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
