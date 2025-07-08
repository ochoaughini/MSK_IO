"""Placeholder evaluation utilities for segmentation outputs."""
from __future__ import annotations


def evaluate(pred: str, truth: str) -> float:
    """Return dummy Dice score between prediction and ground truth."""
    # Real implementation would load volumes and compute Dice.
    return 1.0 if pred == truth else 0.0
