"""
chatbot.py - explains the ranked terrain-similarity candidates in plain language.

Data comes from the existing integration layer (get_discoveries) and model/model_metadata.json.
 * If ANTHROPIC_API_KEY is set: Claude writes the answer, but it only sees the real facts below.
   If its reply contains a decimal number that is not in those facts, the reply is discarded.
 * Otherwise (or if the call fails): a rule-based fallback answers.
Only the Python standard library is needed here.

    from integration_layer.chatbot import answer
    print(answer("Why is MARS-EARTH-03 ranked high?")["answer"])
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

from .integration import get_discoveries

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
PACKAGE_DIR = Path(__file__).resolve().parent
METADATA_PATH = PACKAGE_DIR.parent / "model" / "model_metadata.json"
CLEANED_JSON = PACKAGE_DIR / "data" / "cleaned_discoveries.json"

# record key -> (model feature name, plain name, unit, meaning)
DESCRIPTORS = {
    "elevation_m": ("elevation", "elevation", "m", "height above the reference level of the dataset"),
    "slope_deg": ("slope_deg", "slope", "degrees", "how steep the ground is on average"),
    "local_relief_m": ("local_relief", "local relief", "m", "highest minus lowest point in a small neighbourhood"),
    "roughness_m": ("roughness", "roughness", "m", "how much heights vary in a small neighbourhood"),
}

GLOSSARY = {
    "similarity": "The similarity score is 1 divided by (1 + distance). A higher score means the location's terrain numbers are closer "
                  "to the average Mars terrain profile. It is a measurement, not a probability, and not a confidence level.",
    "distance": "The distance is how far a location's terrain numbers are from the average Mars profile, after all numbers are put on a "
                "common scale. A smaller distance means a closer match.",
    "profile": "The Mars profile is the average of every Mars grid cell in the project's MOLA elevation data, using the same terrain numbers. "
               "It is one 'average Mars', not a specific place on Mars.",
    "candidate": "A candidate is one of the top-ranked Earth locations by similarity. It is a ranked computational terrain-similarity "
                 "candidate that still needs independent geological review. It is not a confirmed discovery.",
    "analog": "A terrain analog is an Earth place that resembles another world in some way. Here it only means similar numbers for "
              "elevation, slope, relief and roughness. Similar numbers do not prove a similar geology.",
    "nasadem": "NASADEM is a NASA elevation dataset for Earth (about 30 m resolution), reprocessed from Space Shuttle radar data.",
    "mola": "MOLA is the laser altimeter on the Mars Global Surveyor spacecraft. The project uses its global Mars topography map.",
    "slope": "Slope is the steepness of the ground in degrees: 0 is flat.",
    "relief": "Local relief is the height difference between the highest and lowest point in a small neighbourhood of the map.",
    "roughness": "Roughness is how much heights vary in a small neighbourhood (a standard deviation, in metres).",
    "elevation": "Elevation is height. Earth heights are measured from sea level and Mars heights from a different reference, "
                 "so raw elevations are not directly comparable between planets.",
    "machine learning": "The model here is simple: it rescales each terrain number, then measures the distance of each Earth location from the "
                        "average Mars profile (scikit-learn). Nothing is trained to recognise 'real' analogs, since no confirmed examples exist.",
}

LIMITATIONS = [
    "These are ranked computational terrain-similarity candidates, not confirmed discoveries or verified analogs. Independent geological review is pending.",
    "Only 6,000 sampled points from six NASADEM tiles were searched (1,000 per tile), so this is not a global search of Earth.",
    "Locations are compared with the average of all Mars, not with a particular Martian site, so a match says nothing about any specific place on Mars.",
    "Only four numbers are used (elevation, slope, relief, roughness). Rock type, minerals, climate and water are not considered.",
    "Roughness appears twice in the distance (it is duplicated as 'elevation spread'), so it counts double. Elevation is also included even though Earth and Mars heights use different references.",
    "Earth and Mars measurements were made at different map scales (about 27.8 km and 14.8 km per cell), which can change slope, relief and roughness.",
    "The score is not a probability, and no accuracy can be measured because there are no confirmed examples to test against.",
]

SYSTEM_PROMPT = (
    "You are ORBIT, a friendly science guide for a student project that ranks Earth terrain samples by how close their "
    "elevation, slope, relief and roughness are to the average Mars terrain (NASADEM vs MOLA).\n"
    "Rules:\n"
    "- Use ONLY the FACTS and GLOSSARY given. Never invent coordinates, place names, numbers, geology, causes or predictions.\n"
    "- Copy numbers exactly as given; do not do your own arithmetic. Use the provided differences and shares.\n"
    "- Call results 'ranked computational terrain-similarity candidates'. Never say discovery, confirmed, verified analog, anomaly, "
    "probability or accuracy. Say independent geological review is still needed.\n"
    "- Locations have coordinates only. If asked where or what a place is, say that the data has no place name.\n"
    "- If the facts do not answer the question, say so. Keep answers under 150 words, in simple words, "
    "in the same language the user writes in."
)


# ---------- loading ----------

def load_records() -> list[dict]:
    """Candidates via the integration layer; falls back to the bundled cleaned JSON."""
    try:
        return get_discoveries(csv_path=os.environ.get("TERRAIN_CSV"))
    except FileNotFoundError:
        with CLEANED_JSON.open(encoding="utf-8") as stream:
            return json.load(stream)


def load_metadata() -> dict:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"{METADATA_PATH} not found. Run build_model.py from the project root.")
    with METADATA_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def normalise_id(text: str) -> str | None:
    """'mars-earth-3', 'candidate 3', '#3', 'location 12' -> 'MARS-EARTH-03'."""
    m = re.search(r"mars-earth-(\d+)", text, re.I) or re.search(r"(?:candidate|location|discovery|number|id|#)\s*#?\s*(\d{1,2})\b", text, re.I)
    return f"MARS-EARTH-{int(m.group(1)):02d}" if m else None


def find_ids(message: str) -> list[str]:
    found = []
    for m in re.finditer(r"mars-earth-(\d+)|(?:candidate|location|discovery|number|id|#)\s*#?\s*(\d{1,2})\b", message, re.I):
        found.append(f"MARS-EARTH-{int(m.group(1) or m.group(2)):02d}")
    return list(dict.fromkeys(found))


# ---------- facts ----------

def descriptor_facts(record: dict, meta: dict) -> list[dict]:
    """Per-descriptor: value, Mars reference, and share of the distance (the 'why')."""
    features = record.get("terrain_features", {})
    rows, total = [], 0.0
    for key, (name, plain, unit, _meaning) in DESCRIPTORS.items():
        if key not in features:
            continue
        lo, hi = meta["rescale_low"][name], meta["rescale_high"][name]
        scaled = min(1.0, max(0.0, (features[key] - lo) / (hi - lo)))
        z = (scaled - meta["mars_profile_rescaled"][name]) / meta["standardise_scale"][name]
        weight = 2 if name == "roughness" else 1   # elev_std duplicates roughness, so it counts twice
        contribution = weight * z * z
        total += contribution
        ref = meta["mars_reference_raw"][name]
        rows.append({"descriptor": plain, "unit": unit, "value": round(features[key], 2), "mars_average": round(ref["mean"], 1),
                     "mars_median": round(ref["median"], 1), "mars_middle_half": [round(ref["p25"], 1), round(ref["p75"], 1)],
                     "direction_vs_mars_average": "higher" if features[key] > ref["mean"] else "lower",
                     "_c": contribution})
    for row in rows:
        row["share_of_distance_percent"] = round(100 * row.pop("_c") / total, 1) if total else None
    return rows


def record_facts(records: list[dict], index: int, meta: dict) -> dict:
    r = records[index]
    q = meta["earth_similarity_quantiles"]
    return {
        "id": r["discovery_id"], "position_in_ranking": index + 1, "candidates_listed": len(records),
        "latitude": round(r["latitude"], 4), "longitude": round(r["longitude"], 4),
        "similarity_score": round(r["similarity_score"], 4),
        "distance_to_mars_profile": round(r["distance_to_mars_profile"], 4),
        "descriptors": descriptor_facts(r, meta),
        "source_tile": r.get("source_file"),
        "status": r.get("candidate_status"), "verification": r.get("verification_status"),
        "context": {"earth_points_searched": meta["n_earth_points"], "best_similarity_among_all_searched": q["1.0"],
                    "median_similarity_among_all_searched": q["0.5"], "highest_possible_similarity": 1.0},
    }


def compare_facts(a: dict, b: dict) -> list[dict]:
    rows = []
    for da, db in zip(a["descriptors"], b["descriptors"]):
        rows.append({"descriptor": da["descriptor"], "unit": da["unit"], a["id"]: da["value"], b["id"]: db["value"],
                     "difference_second_minus_first": round(db["value"] - da["value"], 2)})
    return rows


def build_facts(records: list[dict], ids: list[str], meta: dict) -> dict:
    by_id = {r["discovery_id"]: i for i, r in enumerate(records)}
    chosen = [by_id[i] for i in ids if i in by_id]
    facts = {
        "method": meta["method"], "similarity_formula": meta["similarity_formula"],
        "candidates_listed": len(records),
        "model_check": ("The saved model reproduces the project's stored ranking."
                        if meta["verification"].get("reproduces_stored_ranking") else "Reproduction of the stored ranking was not verified."),
        "limitations": LIMITATIONS,
    }
    shown = chosen if chosen else list(range(min(5, len(records))))
    facts["candidates"] = [record_facts(records, i, meta) for i in shown]
    if not chosen:
        facts["note"] = "No candidate was named; these are the 5 highest-ranked ones."
    if len(chosen) == 2:
        facts["comparison"] = compare_facts(facts["candidates"][0], facts["candidates"][1])
    return facts


# ---------- fallback (no AI) ----------

def explain_candidate(f: dict) -> str:
    text = (f"{f['id']} is at latitude {f['latitude']}, longitude {f['longitude']} (the data has coordinates only, no place name). "
            f"It is number {f['position_in_ranking']} of {f['candidates_listed']} listed candidates, with a similarity score of "
            f"{f['similarity_score']} (1.0 would be a perfect match to the average Mars profile; the best of all "
            f"{f['context']['earth_points_searched']} points searched was {f['context']['best_similarity_among_all_searched']}).")
    ranked = sorted(f["descriptors"], key=lambda d: d["share_of_distance_percent"] or 0)
    closest, furthest = ranked[0], ranked[-1]
    text += (f" Its {closest['descriptor']} ({closest['value']} {closest['unit']}) is the number closest to the Mars average "
             f"({closest['mars_average']} {closest['unit']}), while its {furthest['descriptor']} ({furthest['value']} {furthest['unit']}) "
             f"is {furthest['direction_vs_mars_average']} than the Mars average ({furthest['mars_average']} {furthest['unit']}) and makes up "
             f"{furthest['share_of_distance_percent']}% of its distance.")
    return text + " This is a ranked computational candidate that still needs independent geological review, not a discovery."


def compare_text(a: dict, b: dict, rows: list[dict]) -> str:
    better = a if a["similarity_score"] > b["similarity_score"] else b
    text = (f"{a['id']} has similarity {a['similarity_score']} and {b['id']} has {b['similarity_score']}, so {better['id']} "
            f"is closer to the average Mars profile. Side by side:\n")
    for r in rows:
        text += f"- {r['descriptor']}: {r[a['id']]} vs {r[b['id']]} {r['unit']}\n"
    return text + "These differences are descriptive only; they do not say either place is geologically like Mars."


LIMITATION_WORDS = ("limit", "accura", "reliable", "wrong", "trust", "confirm", "discover", "real", "sure", "problem")


def limitations_text() -> str:
    return "Here is what to keep in mind:\n" + "\n".join(f"- {item}" for item in LIMITATIONS)


def fallback_answer(message: str, records: list[dict], ids: list[str], meta: dict) -> str:
    text = message.lower()
    facts = build_facts(records, ids, meta)
    named = [c for c in facts["candidates"]] if ids and any(i in {r["discovery_id"] for r in records} for i in ids) else []
    if len(named) >= 2:
        return compare_text(named[0], named[1], facts["comparison"])
    if len(named) == 1:
        # The website sends the open candidate as context even for general questions ("What are the limitations?").
        # If the user did not name this candidate themselves, answer a limitations or "what is / what does ... mean"
        # question as a general one instead of repeating the candidate explanation.
        if not find_ids(message):
            if any(w in text for w in LIMITATION_WORDS):
                return limitations_text()
            if re.match(r"\s*(what is|what's|what are|what does|define|meaning of)\b", text):
                for term, definition in GLOSSARY.items():
                    if term in text:
                        return definition
        return explain_candidate(named[0])
    if ids:
        return f"I can't find {ids[0]} in the results. The listed candidates are MARS-EARTH-01 to MARS-EARTH-{len(records):02d}."
    if any(w in text for w in LIMITATION_WORDS):
        return limitations_text()
    if re.search(r"\b(top|best|list|highest|which)\b", text) and "what is" not in text:
        lines = [f"- {c['id']} ({c['latitude']}, {c['longitude']}), similarity {c['similarity_score']}" for c in facts["candidates"]]
        return "The highest-ranked candidates:\n" + "\n".join(lines) + "\nAsk me about any of them, for example 'why is MARS-EARTH-01 ranked high?'"
    for term, definition in GLOSSARY.items():
        if term in text:
            return definition
    return (f"The project ranked Earth terrain samples by how close their elevation, slope, relief and roughness are to the average Mars "
            f"terrain. {len(records)} top-ranked locations are listed as candidates that need independent geological review. "
            f"Ask about one (e.g. 'explain MARS-EARTH-02'), compare two, or ask what a term means.")


# ---------- AI path ----------

def call_ai(system: str, messages: list[dict]) -> str | None:
    """Anthropic Messages API. Returns text, or None when there is no key or the call fails."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    body = json.dumps({"model": MODEL, "max_tokens": 600, "system": system, "messages": messages}).encode()
    request = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text") or None
    except Exception:
        return None


def numbers_are_grounded(reply: str, allowed_text: str) -> bool:
    """Every decimal number in the reply must appear in the allowed text (rounding is tolerated)."""
    allowed = [float(n) for n in re.findall(r"\d+\.\d+|\d+", allowed_text)]
    for token in re.findall(r"\d+\.\d+", reply):
        tolerance = 0.5 * 10 ** -len(token.split(".")[1]) + 1e-9
        if not any(abs(float(token) - a) <= tolerance for a in allowed):
            return False
    return True


def answer(message: str, history: list[dict] | None = None, discovery_ids: list[str] | None = None,
           records: list[dict] | None = None, meta: dict | None = None) -> dict:
    """Returns {"answer": str, "mode": "ai" | "fallback", "discovery_ids": [...]}."""
    records = records or load_records()
    meta = meta or load_metadata()
    ids = list(dict.fromkeys([normalise_id(i) or i for i in (discovery_ids or [])] + find_ids(message)))
    facts_text = json.dumps(build_facts(records, ids, meta))
    glossary_text = json.dumps(GLOSSARY)

    messages = [m for m in (history or [])[-6:] if m.get("role") in ("user", "assistant") and m.get("content")]
    messages.append({"role": "user", "content": f"FACTS:\n{facts_text}\n\nGLOSSARY:\n{glossary_text}\n\nQUESTION: {message}"})
    reply = call_ai(SYSTEM_PROMPT, messages)
    if reply and numbers_are_grounded(reply, facts_text + glossary_text):
        return {"answer": reply.strip(), "mode": "ai", "discovery_ids": ids}
    return {"answer": fallback_answer(message, records, ids, meta), "mode": "fallback", "discovery_ids": ids}


if __name__ == "__main__":
    print("ORBIT terrain chatbot (empty line to quit)")
    while (question := input("\nYou: ").strip()):
        reply = answer(question)
        print(f"\nORBIT [{reply['mode']}]: {reply['answer']}")
