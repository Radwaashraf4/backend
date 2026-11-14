"""Reads the ranked candidates from outputs/discovery_results/novel_candidates_ranked.csv. Use ONLY if your project has no integration.py."""
import csv, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "discovery_results" / "novel_candidates_ranked.csv"

def get_discoveries(csv_path=None):
    path = Path(csv_path) if csv_path else CSV
    if not path.is_file():
        raise FileNotFoundError(path)
    out = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append({"discovery_id": r["candidate_id"], "latitude": float(r["lat"]), "longitude": float(r["lon"]),
                        "similarity_score": float(r["similarity_score"]), "distance_to_mars_profile": float(r["distance_to_mars_profile"]),
                        "terrain_features": {"elevation_m": float(r["elevation"]), "slope_deg": float(r["slope_deg"]),
                                             "local_relief_m": float(r["local_relief"]), "roughness_m": float(r["roughness"])},
                        "source_file": r["source"], "candidate_status": r["scientific_status"],
                        "verification_status": r["confidence_status"]})
    return out


def get_discovery(discovery_id, csv_path=None):
    for r in get_discoveries(csv_path):
        if r["discovery_id"].lower() == str(discovery_id).lower():
            return r
    raise KeyError(discovery_id)


def compare_discoveries(id_a, id_b, csv_path=None):
    a, b = get_discovery(id_a, csv_path), get_discovery(id_b, csv_path)
    fa, fb = a["terrain_features"], b["terrain_features"]
    shared = [k.replace("_m", "") if k != "slope_deg" else k for k in fa if k in fb]
    return {"discovery_a": a["discovery_id"], "discovery_b": b["discovery_id"], "shared_terrain_features": shared,
            "differences": {k: round(fb[k] - fa[k], 4) for k in fa if k in fb},
            "composite_similarity_score": None,
            "composite_score_note": "Composite score not calculated: the two candidates are compared feature by feature only."}
