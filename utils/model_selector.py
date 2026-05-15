"""
utils/model_selector.py
=======================
Contribution 2 — Model Selector Agent (Confidence-Aware Decision Tree)

3-question decision tree: crop -> task -> modality
Each answer includes a confidence score 1-10.
If confidence < 6, a clarifying question is returned instead of guessing.

This prevents confidently wrong answers — a system that asks for
clarification is more intelligent than one that guesses incorrectly.
"""

import json
import os
from pathlib import Path

MODEL_ZOO_PATH = Path("./model_zoo.json")
CONFIDENCE_THRESHOLD = 6


def _build_lookup(zoo_path: Path) -> dict:
    zoo = json.loads(zoo_path.read_text())
    lookup = {}
    for category, models in zoo.items():
        if not isinstance(models, list):
            continue
        for m in models:
            if not isinstance(m, dict):
                continue
            key = (m["crop"], m["task"], m["modality"])
            lookup[key] = m["id"]
    return lookup


LOOKUP = _build_lookup(MODEL_ZOO_PATH)


def _ask(client, model_name: str, system: str, user: str) -> tuple:
    """
    Ask one decision tree question.
    Returns (answer, confidence) where confidence is 1-10.
    LLM is prompted to reply as: answer:confidence e.g. 'rice:9'
    """
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
        max_tokens=15,
    )
    raw = resp.choices[0].message.content.strip().lower()

    # Parse "answer:confidence" format
    if ":" in raw:
        parts = raw.split(":")
        answer = parts[0].strip().split()[0].rstrip(".,")
        try:
            confidence = int(parts[1].strip())
        except ValueError:
            confidence = 5
    else:
        answer = raw.split()[0].rstrip(".,")
        confidence = 5

    return answer, confidence


def select_model(user_query: str, client=None, model_name: str = None):
    """
    Walk the 3-question decision tree with confidence checks.

    Returns
    -------
    str  : model ID if confident match found
    str  : clarifying question string (starts with 'CLARIFY:') if uncertain
    None : if no match found in lookup
    """
    from openai import AzureOpenAI

    if client is None:
        client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_API_URL"],
            api_version=os.environ["AZURE_API_VERSION"],
        )
    if model_name is None:
        model_name = os.environ["MODEL_NAME"]

    system = (
        "You are a plant phenotyping assistant. "
        "Answer with EXACTLY: answer:confidence where confidence is 1-10. "
        "Example: rice:9 or unknown:3. No other text."
    )

    # ── Q1: Crop ──────────────────────────────────────────────────────────────
    crop, crop_conf = _ask(
        client, model_name, system,
        f"Task: '{user_query}'\n\n"
        "What crop species? Options: rice, wheat, maize, banana, coffee, "
        "arabidopsis, potato, multi_crop, unknown\n"
        "Reply as: answer:confidence"
    )
    print(f"[ModelSelector] Q1 crop     -> {crop} (confidence: {crop_conf}/10)")

    if crop_conf < CONFIDENCE_THRESHOLD or crop == "unknown":
        q = (
            "CLARIFY: Could you specify the crop species? "
            "(e.g. rice, wheat, maize, banana, coffee, arabidopsis, potato)"
        )
        print(f"[ModelSelector] Low confidence on crop — asking user: {q}")
        return q

    # ── Q2: Task ──────────────────────────────────────────────────────────────
    task, task_conf = _ask(
        client, model_name, system,
        f"Task: '{user_query}'\n\n"
        "What is the primary computer vision task? "
        "Options: nutrient_deficiency, detection_counting, segmentation, "
        "temporal_phenotyping, unknown\n"
        "Reply as: answer:confidence"
    )
    print(f"[ModelSelector] Q2 task     -> {task} (confidence: {task_conf}/10)")

    # Task is usually clear from the verb — only ask if truly unknown
    if task == "unknown" or task_conf < 4:
        q = (
            "CLARIFY: What should I do with the images? "
            "(e.g. classify nutrient deficiency, count/detect organs, "
            "segment leaves, analyse time series)"
        )
        print(f"[ModelSelector] Low confidence on task — asking user: {q}")
        return q

    # ── Q3: Modality ──────────────────────────────────────────────────────────
    modality, mod_conf = _ask(
        client, model_name, system,
        f"Task: '{user_query}'\n\n"
        "What imaging modality? "
        "Options: close_range_rgb, uav_aerial, satellite, unknown\n"
        "Reply as: answer:confidence"
    )
    print(f"[ModelSelector] Q3 modality -> {modality} (confidence: {mod_conf}/10)")

    if mod_conf < CONFIDENCE_THRESHOLD or modality == "unknown":
        q = (
            "CLARIFY: Are these close-range leaf photos, UAV/drone aerial images, "
            "or satellite imagery?"
        )
        print(f"[ModelSelector] Low confidence on modality — asking user: {q}")
        return q

    # ── Lookup ────────────────────────────────────────────────────────────────
    key = (crop, task, modality)
    model_id = LOOKUP.get(key)

    if model_id:
        print(f"[ModelSelector] Match: {key} -> {model_id}")
        # ── Verification ──────────────────────────────────────────────────────
        from utils.verifier import verify_model_selection
        verdict = verify_model_selection(
            user_query, model_id, crop, task, modality, client, model_name
        )
        if not verdict["accepted"]:
            print(f"[ModelSelector] Verifier REJECTED — returning CLARIFY")
            return (
                f"CLARIFY: The model selection may be incorrect "
                f"({verdict['reason']}). Could you clarify your task?"
            )
    else:
        print(f"[ModelSelector] No match for {key}")
        print(f"[ModelSelector] Available: {list(LOOKUP.keys())}")

    return model_id
