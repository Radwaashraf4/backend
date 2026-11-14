from __future__ import annotations

import tempfile
import unittest

from integration_layer.integration import (
    compare_discoveries,
    get_discoveries,
    get_discovery,
)
from integration_layer.model_loader import ModelArtifactsNotFound, load_model_artifacts, model_artifact_status


class IntegrationTests(unittest.TestCase):
    def test_ranked_dataset_loads_without_retraining(self) -> None:
        records = get_discoveries()
        self.assertEqual(len(records), 20)
        self.assertEqual(records[0]["discovery_id"], "MARS-EARTH-01")
        self.assertIn("terrain_features", records[0])
        self.assertEqual(records[0]["candidate_status"], "Potentially Novel Terrain Analog Candidate")
        self.assertIn("pending", records[0]["verification_status"].lower())
        self.assertNotIn("rank", records[0])  # Source CSV does not contain an explicit rank field.

    def test_lookup_by_id(self) -> None:
        record = get_discovery("MARS-EARTH-02")
        self.assertEqual(record["discovery_id"], "MARS-EARTH-02")
        self.assertIsInstance(record["latitude"], float)
        self.assertIsInstance(record["longitude"], float)

    def test_pairwise_comparison_is_feature_specific(self) -> None:
        result = compare_discoveries("MARS-EARTH-01", "MARS-EARTH-02")
        self.assertIn("slope_deg", result["shared_terrain_features"])
        self.assertIsNone(result["composite_similarity_score"])
        self.assertIn("not calculated", result["composite_score_note"].lower())

    def test_missing_model_is_reported_not_retrained(self) -> None:
        # Original intent kept: in a folder with no saved artifacts the loader reports that, and never trains.
        with tempfile.TemporaryDirectory() as empty_project:
            status = model_artifact_status(empty_project)
            self.assertFalse(status["model_available"])
            self.assertFalse(status["preprocessor_available"])
            with self.assertRaises(ModelArtifactsNotFound):
                load_model_artifacts(empty_project)


if __name__ == "__main__":
    unittest.main()
