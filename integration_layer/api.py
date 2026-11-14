"""
api.py - small Flask API so the website can read the ML results and talk to the chatbot.

Run from the project root:   pip install flask scikit-learn pandas joblib
                             python integration_layer/api.py        (http://localhost:5000)
Optional: set ANTHROPIC_API_KEY for AI-written answers (otherwise the rule-based fallback answers).

GET  /api/health
GET  /api/model                       what the model is, and its limitations
GET  /api/discoveries?limit=5         ranked candidates (+ plain-language descriptor facts)
GET  /api/discoveries/<id>            one candidate, e.g. MARS-EARTH-03
POST /api/chat     {"message": "...", "discovery_ids": ["MARS-EARTH-03"], "history": []}
POST /api/compare  {"id_a": "MARS-EARTH-01", "id_b": "MARS-EARTH-02"}
POST /api/score    {"rows": [{"elevation":..,"slope_deg":..,"local_relief":..,"roughness":..}]}
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request  # noqa: E402

from integration_layer import chatbot, scoring  # noqa: E402

app = Flask(__name__)
DISCLAIMER = ("Ranked computational terrain-similarity candidates. Not confirmed discoveries; "
              "independent geological review is pending.")


@app.after_request
def allow_frontend(response):
    """Simple CORS so a website on another port/domain can call this API."""
    response.headers["Access-Control-Allow-Origin"] = os.environ.get("CORS_ORIGIN", "*")
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.errorhandler(FileNotFoundError)
def missing_file(error):
    return jsonify({"error": str(error)}), 503


def find(records, discovery_id):
    wanted = chatbot.normalise_id(str(discovery_id)) or str(discovery_id)
    return next((i for i, r in enumerate(records) if r["discovery_id"] == wanted), None)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "ai_enabled": bool(os.environ.get("ANTHROPIC_API_KEY")),
                    "model_saved": scoring.BUNDLE_PATH.is_file()})


@app.get("/api/model")
def model_info():
    meta = chatbot.load_metadata()
    return jsonify({"method": meta["method"], "similarity_formula": meta["similarity_formula"],
                    "n_earth_points": meta["n_earth_points"], "n_mars_points": meta["n_mars_points"],
                    "verification": meta["verification"], "limitations": chatbot.LIMITATIONS, "disclaimer": DISCLAIMER})


@app.get("/api/discoveries")
def discoveries():
    records, meta = chatbot.load_records(), chatbot.load_metadata()
    limit = request.args.get("limit", default=len(records), type=int)
    items = [{**r, "descriptor_facts": chatbot.descriptor_facts(r, meta), "position_in_ranking": i + 1}
             for i, r in enumerate(records[:max(limit, 0)])]
    return jsonify({"count": len(items), "disclaimer": DISCLAIMER, "discoveries": items})


@app.get("/api/discoveries/<discovery_id>")
def one_discovery(discovery_id):
    records, meta = chatbot.load_records(), chatbot.load_metadata()
    i = find(records, discovery_id)
    if i is None:
        return jsonify({"error": f"{discovery_id} not found"}), 404
    return jsonify({**records[i], "descriptor_facts": chatbot.descriptor_facts(records[i], meta),
                    "position_in_ranking": i + 1, "disclaimer": DISCLAIMER})


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    return jsonify(chatbot.answer(message, history=body.get("history"), discovery_ids=body.get("discovery_ids")))


@app.post("/api/compare")
def compare():
    body = request.get_json(silent=True) or {}
    records, meta = chatbot.load_records(), chatbot.load_metadata()
    ia, ib = find(records, body.get("id_a", "")), find(records, body.get("id_b", ""))
    if ia is None or ib is None:
        return jsonify({"error": "id_a and id_b must both be valid discovery ids"}), 404
    a, b = (chatbot.record_facts(records, i, meta) for i in (ia, ib))
    rows = chatbot.compare_facts(a, b)
    reply = chatbot.answer(f"Compare {a['id']} and {b['id']}.", records=records, meta=meta)
    return jsonify({**reply, "comparison": rows})


@app.post("/api/score")
def score():
    rows = (request.get_json(silent=True) or {}).get("rows")
    if not rows or not isinstance(rows, list):
        return jsonify({"error": "rows (a list of feature objects) is required"}), 400
    try:
        return jsonify({"scores": scoring.score_rows(rows),
                        "note": "Only comparable if features were computed exactly like the project's feature tables."})
    except (ValueError, KeyError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
