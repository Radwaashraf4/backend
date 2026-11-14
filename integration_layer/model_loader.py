"""Loads model/model_bundle.joblib. Use ONLY if your project has no model_loader.py."""
from pathlib import Path
from types import SimpleNamespace
import joblib
ROOT = Path(__file__).resolve().parents[1]
class ModelArtifactsNotFound(FileNotFoundError): pass
def load_model_artifacts(project_root=None, bundle_path=None):
    p = Path(bundle_path) if bundle_path else Path(project_root or ROOT) / "model" / "model_bundle.joblib"
    if not p.is_file(): raise ModelArtifactsNotFound(str(p))
    d = joblib.load(p); return SimpleNamespace(**d)
def model_artifact_status(project_root=None):
    ok = (Path(project_root or ROOT) / "model" / "model_bundle.joblib").is_file()
    return {"model_available": ok, "preprocessor_available": ok}
