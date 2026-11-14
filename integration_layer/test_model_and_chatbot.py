"""Tests for the saved model, chatbot, API and website export. Run from the project root:
    python -m unittest discover -s integration_layer -p 'test_*.py' -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from integration_layer import api, chatbot, scoring
from integration_layer.integration import get_discoveries
from integration_layer.model_loader import model_artifact_status

ROOT = Path(__file__).resolve().parents[1]
RECORDS = get_discoveries()
KEY = {"elevation": "elevation_m", "slope_deg": "slope_deg", "local_relief": "local_relief_m", "roughness": "roughness_m"}


def row_of(record):
    return {name: record["terrain_features"][key] for name, key in KEY.items()}


class ModelTests(unittest.TestCase):
    def test_saved_model_is_found_and_loads(self):
        status = model_artifact_status()
        self.assertTrue(status["model_available"] and status["preprocessor_available"])
        bundle = scoring.load_bundle()
        self.assertTrue(hasattr(bundle.model, "kneighbors") and hasattr(bundle.preprocessor, "transform"))

    def test_saved_model_reproduces_stored_ranking(self):
        scores = scoring.score_rows([row_of(r) for r in RECORDS])
        for record, score in zip(RECORDS, scores):
            self.assertAlmostEqual(score["distance_to_mars_profile"], record["distance_to_mars_profile"], places=5)
            self.assertAlmostEqual(score["similarity_score"], record["similarity_score"], places=5)

    def test_metadata_says_reproduction_verified(self):
        self.assertTrue(scoring.load_metadata()["verification"]["reproduces_stored_ranking"])

    def test_mars_like_row_beats_extreme_row(self):
        mars = scoring.load_metadata()["mars_reference_raw"]
        typical = {k: mars[k]["median"] for k in ("elevation", "slope_deg", "local_relief", "roughness")}
        extreme = {"elevation": 5000, "slope_deg": 30, "local_relief": 3000, "roughness": 900}
        a, b = scoring.score_rows([typical, extreme])
        self.assertGreater(a["similarity_score"], b["similarity_score"])

    def test_score_rows_validation(self):
        with self.assertRaises(ValueError):
            scoring.score_rows([{"elevation": 1, "slope_deg": 1}])
        with self.assertRaises(ValueError):
            scoring.score_rows([{"elevation": None, "slope_deg": 1, "local_relief": 1, "roughness": 1}])


class ChatbotTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_id_parsing(self):
        self.assertEqual(chatbot.find_ids("tell me about mars-earth-3 and candidate 12"), ["MARS-EARTH-03", "MARS-EARTH-12"])
        self.assertEqual(chatbot.find_ids("hello there 2026"), [])

    def test_explanation_uses_real_numbers(self):
        r = RECORDS[2]
        reply = chatbot.answer("Why is MARS-EARTH-03 ranked high?")
        self.assertEqual(reply["mode"], "fallback")
        self.assertIn(str(round(r["similarity_score"], 4)), reply["answer"])
        self.assertIn(str(round(r["latitude"], 4)), reply["answer"])
        self.assertIn("geological review", reply["answer"])
        self.assertNotIn("confirmed discovery", reply["answer"].replace("not a discovery", ""))

    def test_shares_sum_to_100(self):
        meta = chatbot.load_metadata()
        for record in RECORDS:
            shares = [d["share_of_distance_percent"] for d in chatbot.descriptor_facts(record, meta)]
            self.assertAlmostEqual(sum(shares), 100.0, delta=0.3)

    def test_compare_limitations_glossary_unknown_and_top(self):
        self.assertIn("closer to the average Mars profile", chatbot.answer("compare candidate 1 and candidate 5")["answer"])
        self.assertIn("6,000 sampled points", chatbot.answer("Is this accurate?")["answer"])
        self.assertIn("1 divided by", chatbot.answer("what is the similarity score")["answer"])
        self.assertIn("can't find", chatbot.answer("explain MARS-EARTH-99")["answer"])
        self.assertIn("MARS-EARTH-01", chatbot.answer("which are the top candidates")["answer"])

    def test_open_candidate_context_does_not_hijack_general_questions(self):
        # The website sends the open candidate as context; general questions must still get general answers.
        ctx = ["MARS-EARTH-03"]
        self.assertIn("6,000 sampled points", chatbot.answer("What are the limitations?", discovery_ids=ctx)["answer"])
        self.assertIn("1 divided by", chatbot.answer("What does the similarity score mean?", discovery_ids=ctx)["answer"])
        own = chatbot.answer("Why is this candidate ranked here?", discovery_ids=ctx)["answer"]
        self.assertIn("MARS-EARTH-03 is at latitude", own)
        # naming the candidate yourself still gives its explanation
        self.assertIn("MARS-EARTH-03 is at latitude", chatbot.answer("What does MARS-EARTH-03 mean for Mars?", discovery_ids=ctx)["answer"])

    def test_api_chat_endpoint_with_context_and_history(self):
        client = api.app.test_client()
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        res = client.post("/api/chat", json={"message": "Why is this candidate ranked here?", "discovery_ids": ["MARS-EARTH-03"], "history": history})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["discovery_ids"], ["MARS-EARTH-03"])
        self.assertEqual(client.post("/api/chat", json={"message": "  "}).status_code, 400)

    def test_ai_reply_used_only_when_numbers_are_real(self):
        good = "MARS-EARTH-01 has a similarity score of 0.6759, so it is a candidate that needs review."
        with mock.patch.object(chatbot, "call_ai", return_value=good):
            self.assertEqual(chatbot.answer("explain MARS-EARTH-01")["mode"], "ai")
        with mock.patch.object(chatbot, "call_ai", return_value="Its similarity is 0.9912 and slope 77.31."):
            self.assertEqual(chatbot.answer("explain MARS-EARTH-01")["mode"], "fallback")

    def test_ai_request_format(self):
        seen = {}

        def fake_urlopen(request, timeout):
            seen["headers"] = {k.lower(): v for k, v in request.header_items()}
            seen["body"] = json.loads(request.data)
            raise OSError("offline test")

        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}), mock.patch("urllib.request.urlopen", fake_urlopen):
            self.assertIsNone(chatbot.call_ai("sys", [{"role": "user", "content": "hi"}]))
        self.assertEqual(seen["headers"]["x-api-key"], "k")
        self.assertEqual(seen["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(seen["body"]["system"], "sys")
        self.assertIn("max_tokens", seen["body"])

    def test_chat_works_from_bundled_json_when_csv_is_missing(self):
        with mock.patch.object(chatbot, "get_discoveries", side_effect=FileNotFoundError("no csv")):
            self.assertEqual(len(chatbot.load_records()), 20)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.client = api.app.test_client()

    def tearDown(self):
        self.env.stop()

    def test_read_endpoints(self):
        self.assertTrue(self.client.get("/api/health").get_json()["model_saved"])
        listing = self.client.get("/api/discoveries?limit=3").get_json()
        self.assertEqual(listing["count"], 3)
        self.assertEqual(listing["discoveries"][0]["discovery_id"], "MARS-EARTH-01")
        self.assertEqual(self.client.get("/api/discoveries/mars-earth-4").get_json()["discovery_id"], "MARS-EARTH-04")
        self.assertEqual(self.client.get("/api/discoveries/NOPE").status_code, 404)
        self.assertIn("limitations", self.client.get("/api/model").get_json())

    def test_chat_compare_score(self):
        chat = self.client.post("/api/chat", json={"message": "Why is this one ranked high?", "discovery_ids": ["MARS-EARTH-02"]}).get_json()
        self.assertIn("MARS-EARTH-02", chat["answer"])
        self.assertEqual(self.client.post("/api/chat", json={}).status_code, 400)
        cmp_ = self.client.post("/api/compare", json={"id_a": "MARS-EARTH-01", "id_b": "MARS-EARTH-02"}).get_json()
        self.assertEqual(len(cmp_["comparison"]), 4)
        self.assertEqual(self.client.post("/api/compare", json={"id_a": "MARS-EARTH-01", "id_b": "X"}).status_code, 404)
        score = self.client.post("/api/score", json={"rows": [row_of(RECORDS[0])]}).get_json()
        self.assertAlmostEqual(score["scores"][0]["similarity_score"], RECORDS[0]["similarity_score"], places=5)
        self.assertEqual(self.client.post("/api/score", json={"rows": [{"elevation": 1}]}).status_code, 400)
        self.assertEqual(self.client.options("/api/chat").headers["Access-Control-Allow-Origin"], "*")


class WebsiteExportTests(unittest.TestCase):
    def test_export_has_real_fields_only(self):
        subprocess.run([sys.executable, str(ROOT / "export_website_data.py")], cwd=ROOT, check=True, capture_output=True)
        text = (ROOT / "website_export" / "mlCandidates.js").read_text(encoding="utf-8")
        items = json.loads(text[text.index("["): text.rindex("]") + 1])
        self.assertEqual(len(items), 20)
        self.assertEqual(len({i["id"] for i in items}), 20)
        for item in items:
            self.assertEqual(item["status"], "candidate")
            self.assertEqual(item["climate"], "Not analysed")
            self.assertEqual(len(item["stats"]), 3)


if __name__ == "__main__":
    unittest.main()
