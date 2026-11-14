"""Score terrain rows with the saved Mars-similarity model (no fitting happens here).

The model is the one stored in model/model_bundle.joblib, created by build_model.py:
  preprocessor : clip-and-rescale each feature, then standardise (sklearn Pipeline)
  model        : a 1-point NearestNeighbors that holds the "Mars profile";
                 its distance to a row = how far the row is from the average Mars terrain.
  similarity   = 1 / (1 + distance)   (same formula as the project's stored ranking)
Needs scikit-learn, pandas and joblib (same version as the one that built the bundle).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model_loader import load_model_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = PROJECT_ROOT / "model" / "model_bundle.joblib"
METADATA_PATH = PROJECT_ROOT / "model" / "model_metadata.json"
FEATURES = ["elevation", "slope_deg", "local_relief", "roughness", "elev_std"]


def load_metadata(path: str | Path | None = None) -> dict:
    with Path(path or METADATA_PATH).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_bundle(path: str | Path | None = None):
    """Load the saved model + preprocessor (trusted local file only)."""
    return load_model_artifacts(bundle_path=path or BUNDLE_PATH)


def score_rows(rows, bundle=None) -> list[dict]:
    """rows: list of dicts (or a DataFrame) with elevation, slope_deg, local_relief, roughness.

    elev_std is optional: in this project's feature tables it is identical to roughness,
    so it is filled from roughness when missing.
    """
    table = pd.DataFrame(rows).copy()
    missing = [c for c in FEATURES[:4] if c not in table.columns]
    if missing:
        raise ValueError(f"missing feature(s): {missing}")
    if "elev_std" not in table.columns:
        table["elev_std"] = table["roughness"]
    table = table[FEATURES].astype(float)
    if table.isna().any().any():
        raise ValueError("feature values must be numbers (no empty values)")
    bundle = bundle or load_bundle()
    distance = bundle.model.kneighbors(bundle.preprocessor.transform(table))[0][:, 0]
    similarity = 1.0 / (1.0 + distance)
    return [{"distance_to_mars_profile": float(d), "similarity_score": float(s)}
            for d, s in zip(distance, similarity)]
