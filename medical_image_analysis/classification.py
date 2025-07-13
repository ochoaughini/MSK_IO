"""
Medical Image Analysis Pipeline - Classification Module
This module handles structure classification and diagnostic suggestions.
"""

import numpy as np
from .config import MODALITY, MODALITY_PARAMS


def classify_structures(features: dict) -> dict:
    """Simulates classification of structures based on extracted features."""
    print("4.1 Classifying structures (AI model placeholder)...")
    params = MODALITY_PARAMS[MODALITY]
    classification_result = {
        "primary_classification": "Indeterminate",
        "confidence_score": 0.0,
        "risk_level": "Unknown",
        "recommendations": []
    }
    area = features.get("total_area_pixels", 0)
    mean_intensity = features.get("mean_intensity", 0)
    texture_contrast = features.get("texture_contrast", 0)
    compactness = features.get("compactness", 0)

    if MODALITY == "CT":
        classification_result = _classify_ct_structures(
            area, mean_intensity, texture_contrast, compactness, params
        )
    elif MODALITY == "MRI":
        classification_result = _classify_mri_structures(
            area, mean_intensity, texture_contrast, compactness, params
        )

    classification_result["recommendations"] = _generate_recommendations(
        classification_result, features
    )

    print(f"4.2 Classification result: {classification_result['primary_classification']}")
    print(f"4.3 Confidence score: {classification_result['confidence_score']:.3f}")

    return classification_result


def _classify_ct_structures(area: float, mean_intensity: float,
                          texture_contrast: float, compactness: float,
                          params: dict) -> dict:
    """CT-specific classification logic."""
    result = {
        "primary_classification": "Indeterminate",
        "confidence_score": 0.5,
        "risk_level": "Medium",
        "recommendations": []
    }

    large_area_threshold = params["classification_area_threshold"]
    low_intensity_threshold = params["classification_intensity_threshold"]
    benign_area_threshold = params["benign_area_threshold"]
    benign_intensity_threshold = params["benign_intensity_threshold"]

    if area > large_area_threshold and mean_intensity < low_intensity_threshold:
        result["primary_classification"] = "Potentially Malignant"
        result["confidence_score"] = 0.8
        result["risk_level"] = "High"
        if texture_contrast > 0.5 and compactness < 0.7:
            result["confidence_score"] = 0.9
    elif area < benign_area_threshold and mean_intensity > benign_intensity_threshold:
        result["primary_classification"] = "Benign"
        result["confidence_score"] = 0.7
        result["risk_level"] = "Low"
        if compactness > 0.8 and texture_contrast < 0.3:
            result["confidence_score"] = 0.85
    else:
        result["primary_classification"] = "Requires Additional Review"
        result["confidence_score"] = 0.6
        result["risk_level"] = "Medium"

    return result


def _classify_mri_structures(area: float, mean_intensity: float,
                           texture_contrast: float, compactness: float,
                           params: dict) -> dict:
    """MRI-specific classification logic."""
    result = {
        "primary_classification": "Indeterminate",
        "confidence_score": 0.5,
        "risk_level": "Medium",
        "recommendations": []
    }

    large_area_threshold = params["classification_area_threshold"]
    low_intensity_threshold = params["classification_intensity_threshold"]
    benign_area_threshold = params["benign_area_threshold"]
    benign_intensity_threshold = params["benign_intensity_threshold"]

    if area > large_area_threshold and mean_intensity < low_intensity_threshold:
        result["primary_classification"] = "Potentially Malignant"
        result["confidence_score"] = 0.75
        result["risk_level"] = "High"
        if texture_contrast > 0.4 and compactness < 0.6:
            result["confidence_score"] = 0.85
    elif area < benign_area_threshold and mean_intensity > benign_intensity_threshold:
        result["primary_classification"] = "Benign"
        result["confidence_score"] = 0.75
        result["risk_level"] = "Low"
        if compactness > 0.7 and texture_contrast < 0.25:
            result["confidence_score"] = 0.9
    else:
        result["primary_classification"] = "Requires Additional Review"
        result["confidence_score"] = 0.55
        result["risk_level"] = "Medium"

    return result


def _generate_recommendations(classification_result: dict, features: dict) -> list:
    """Generates clinical recommendations based on classification results."""
    recommendations = []
    classification = classification_result["primary_classification"]
    confidence = classification_result["confidence_score"]

    if classification == "Potentially Malignant":
        recommendations.extend([
            "Urgent referral to oncology specialist recommended",
            "Consider tissue biopsy for definitive diagnosis",
            "Additional imaging with contrast may be beneficial",
            "Multidisciplinary team review suggested"
        ])
        if confidence < 0.8:
            recommendations.append("Consider second opinion due to moderate confidence")
    elif classification == "Benign":
        recommendations.extend([
            "Routine follow-up imaging in 6-12 months",
            "Continue standard monitoring protocols",
            "No immediate intervention required"
        ])
        if confidence < 0.8:
            recommendations.append("Consider additional imaging modalities for confirmation")
    else:
        recommendations.extend([
            "Radiologist review recommended",
            "Consider additional imaging sequences",
            "Correlate with clinical history and symptoms",
            "Short-term follow-up imaging (3-6 months) may be appropriate"
        ])

    if features.get("total_area_pixels", 0) > 10000:
        recommendations.append("Large lesion size warrants expedited evaluation")
    if features.get("texture_contrast", 0) > 0.6:
        recommendations.append("High texture heterogeneity noted - consider dynamic imaging")

    return recommendations


def integrate_ai_assistance(query: str, classification_result: dict) -> str:
    """Simulates interaction with a model like ChatGPT for additional insights."""
    print("5.2 Integrating AI assistance...")
    classification = classification_result["primary_classification"]
    confidence = classification_result["confidence_score"]

    if "malignant" in classification.lower():
        response = f"""Based on the '{classification}' classification with {confidence:.1%} confidence, \
        additional clinical correlation and possibly a biopsy is recommended. Consider additional \
        imaging modalities like {('MRI' if MODALITY == 'CT' else 'CT')} for detailed evaluation. \
        The elevated risk level suggests urgent specialist referral."""
    elif "benign" in classification.lower():
        response = f"""The '{classification}' classification with {confidence:.1%} confidence suggests \
        a low-risk finding. Routine follow-up may be sufficient, but always consult a medical \
        professional for definitive diagnosis. Consider standard surveillance protocols."""
    else:
        response = f"""The analysis indicates '{classification}' with {confidence:.1%} confidence. \
        This may mean the features are ambiguous or fall into a gray area. Additional patient \
        history, clinical correlation, or specialized imaging may be beneficial. Consider \
        multidisciplinary team review."""

    return response


def validate_classification(classification_result: dict) -> bool:
    """Validates the classification result for consistency."""
    required_keys = ["primary_classification", "confidence_score", "risk_level", "recommendations"]

    for key in required_keys:
        if key not in classification_result:
            print(f"Missing required key: {key}")
            return False

    confidence = classification_result["confidence_score"]
    if not (0.0 <= confidence <= 1.0):
        print(f"Invalid confidence score: {confidence}")
        return False

    valid_risk_levels = ["Low", "Medium", "High", "Unknown"]
    if classification_result["risk_level"] not in valid_risk_levels:
        print(f"Invalid risk level: {classification_result['risk_level']}")
        return False

    return True
