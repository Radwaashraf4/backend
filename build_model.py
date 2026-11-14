"""
build_model.py - rebuild and SAVE the project's Mars-similarity model.

Why this exists: the notebooks computed the ranking in memory and never saved the fitted
scaler/profile. By comparing the saved ranking with the saved feature tables, the method was
recovered exactly (see MODEL_REPRODUCTION.md) and is rebuilt here as a normal scikit-learn model.

Method (reproduces outputs/discovery_results/novel_candidates_ranked.csv):
  1. For each feature, rescale to 0-1 between the lowest 1st-percentile and the highest
     99th-percentile of the two bodies (Earth, Mars), clipping outside values.
  2. Standardise (mean 0, std 1) using Earth + Mars rows together.
  3. The "Mars profile" = the average Mars row after steps 1-2.
  4. distance = Euclidean distance from a row to that profile; similarity = 1 / (1 + distance).

Run from the project root:   python build_model.py
Writes: model/model_bundle.joblib, model/model_metadata.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from integration_layer.scoring import FEATURES, load_bundle, score_rows  # noqa: E402

STORED_RANKING = ROOT / "outputs" / "discovery_results" / "novel_candidates_ranked.csv"


def fit_bundle(earth: pd.DataFrame, mars: pd.DataFrame):
    low = pd.concat([earth[FEATURES].quantile(0.01), mars[FEATURES].quantile(0.01)], axis=1).min(axis=1)
    high = pd.concat([earth[FEATURES].quantile(0.99), mars[FEATURES].quantile(0.99)], axis=1).max(axis=1)
    minmax = MinMaxScaler(clip=True).fit(pd.DataFrame([low, high]))        # bounds = percentiles
    standard = StandardScaler().fit(minmax.transform(pd.concat([earth[FEATURES], mars[FEATURES]])))
    preprocessor = Pipeline([("minmax", minmax), ("standard", standard)])
    profile = preprocessor.transform(mars[FEATURES]).mean(axis=0, keepdims=True)
    model = NearestNeighbors(n_neighbors=1).fit(profile)                    # holds the Mars profile
    return {"model": model, "preprocessor": preprocessor}, low, high


def describe(table: pd.DataFrame) -> dict:
    out = {}
    for name in FEATURES[:4]:
        col = table[name]
        out[name] = {"mean": col.mean(), "median": col.median(), "p25": col.quantile(.25), "p75": col.quantile(.75)}
    return {k: {a: round(float(b), 4) for a, b in v.items()} for k, v in out.items()}


def main():
    earth = pd.read_csv(ROOT / "data/features/earth_features.csv")
    mars = pd.read_csv(ROOT / "data/features/mars_features.csv")
    bundle, low, high = fit_bundle(earth, mars)
    (ROOT / "model").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "model/model_bundle.joblib")

    loaded = load_bundle()
    scores = pd.DataFrame(score_rows(earth, loaded))
    ranked = earth.assign(**scores).sort_values("similarity_score", ascending=False).reset_index(drop=True)

    verification = {"stored_ranking_found": STORED_RANKING.is_file()}
    if STORED_RANKING.is_file():
        stored = pd.read_csv(STORED_RANKING)
        top = ranked.head(len(stored))
        diff = float(np.abs(top.distance_to_mars_profile.values - stored.distance_to_mars_profile.values).max())
        same_places = bool(np.allclose(top.lat, stored.lat) and np.allclose(top.lon, stored.lon))
        verification.update(stored_rows_compared=len(stored), max_abs_distance_difference=diff,
                            same_locations_same_order=same_places,
                            reproduces_stored_ranking=bool(diff < 1e-5 and same_places))
    print("Verification against stored ranking:", verification)

    scaled_profile = (loaded.preprocessor.named_steps["standard"].mean_  # combined-data mean (scaled space)
                      + loaded.model._fit_X[0] * loaded.preprocessor.named_steps["standard"].scale_)
    metadata = {
        "method": "Distance to the average Mars terrain profile, after percentile-clip rescaling and standardisation",
        "similarity_formula": "similarity = 1 / (1 + distance)",
        "features": FEATURES,
        "rescale_low": {k: round(float(v), 6) for k, v in low.items()},
        "rescale_high": {k: round(float(v), 6) for k, v in high.items()},
        "standardise_scale": {k: round(float(v), 6) for k, v in zip(FEATURES, loaded.preprocessor.named_steps["standard"].scale_)},
        "mars_profile_rescaled": {k: round(float(v), 6) for k, v in zip(FEATURES, scaled_profile)},
        "mars_reference_raw": describe(mars),
        "earth_sample_raw": describe(earth),
        "n_earth_points": int(len(earth)),
        "n_mars_points": int(len(mars)),
        "earth_similarity_quantiles": {q: round(float(ranked.similarity_score.quantile(float(q))), 4)
                                       for q in ("0.0", "0.5", "0.9", "0.99", "1.0")},
        "verification": verification,
        "sklearn_version": sklearn.__version__,
        "note": "Rebuilt from the saved feature tables; scikit-learn objects are version-sensitive, "
                "so re-run build_model.py if loading fails on a different version.",
    }
    (ROOT / "model/model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Saved model/model_bundle.joblib and model/model_metadata.json")


if __name__ == "__main__":
    main()
