"""
utils/ontology.py
=================
Contribution 2 — Agricultural Ontology Expansion

Expands user queries by mapping domain synonyms to canonical terms
BEFORE semantic retrieval. This makes retrieval robust to phrasing
variations without retraining the embedding model.

Architecture position:
  User query
      ↓
  Ontology expansion  ← this file
      ↓
  Hybrid retrieval (tool_selector.py)
      ↓
  Manager sees top-k tools
"""

import re

# ── Synonym maps ──────────────────────────────────────────────────────────────
# Each entry: synonym/alias -> canonical term that appears in tool descriptions

CROP_SYNONYMS = {
    "paddy":        "rice",
    "oryza":        "rice",
    "maize":        "maize corn",
    "corn":         "maize corn",
    "zea mays":     "maize corn",
    "coffea":       "coffee",
    "musa":         "banana",
    "triticum":     "wheat",
    "solanum":      "potato",
    "arabidopsis thaliana": "arabidopsis",
}

TASK_SYNONYMS = {
    "yellowing":        "nutrient deficiency classification",
    "chlorosis":        "nutrient deficiency classification",
    "leaf colour":      "nutrient deficiency classification",
    "leaf color":       "nutrient deficiency classification",
    "nutritional":      "nutrient deficiency",
    "stress":           "deficiency",
    "count":            "detect count",
    "how many":         "detect count",
    "number of":        "detect count",
    "ears":             "wheat spike detect count",
    "spike":            "wheat spike detect",
    "panicle":          "rice panicle detect",
    "head":             "spike detect count",
    "rosette":          "arabidopsis segment instance",
    "leaf area":        "segment regression",
    "lai":              "leaf area index regression",
    "biomass":          "regression predict",
    "fresh weight":     "regression predict",
    "dry weight":       "regression predict",
    "phenotype":        "segment classify",
    "phenotyping":      "segment classify",
    "plot":             "coding chart visualize",
    "bar chart":        "coding visualize",
    "graph":            "coding visualize",
    "script":           "coding",
    "visualize":        "coding chart",
    "train":            "finetune",
    "retrain":          "finetune",
    "fine-tune":        "finetune",
    "my dataset":       "finetune custom",
}

MODALITY_SYNONYMS = {
    "drone":        "uav aerial",
    "uav":          "uav aerial",
    "unmanned":     "uav aerial",
    "aerial":       "uav aerial",
    "overhead":     "uav aerial",
    "field camera": "close range rgb",
    "handheld":     "close range rgb",
    "close-range":  "close range rgb",
    "close range":  "close range rgb",
    "sentinel":     "satellite time series",
    "sentinel-2":   "satellite time series",
    "satellite":    "satellite time series",
    "multitemporal":"satellite time series temporal",
    "multi-temporal":"satellite time series temporal",
    "time series":  "temporal satellite",
}

# ── Constraint map ────────────────────────────────────────────────────────────
# Prevents impossible tool selections
# Format: detected_signal -> list of tools that are INCOMPATIBLE
HARD_CONSTRAINTS = {
    "satellite":        ["infer_instance_segmentation", "infer_image_classification",
                         "infer_object_detection", "infer_image_regression"],
    "sentinel":         ["infer_instance_segmentation", "infer_image_classification",
                         "infer_object_detection", "infer_image_regression"],
    "finetune":         ["infer_image_classification", "infer_object_detection",
                         "infer_instance_segmentation", "infer_image_regression",
                         "infer_temporal_analysis", "coding"],
    "coding":           ["infer_image_classification", "infer_object_detection",
                         "infer_instance_segmentation", "infer_image_regression",
                         "infer_temporal_analysis"],
}


def expand_query(query: str) -> str:
    """
    Expand a user query by appending canonical synonyms.
    The original query is preserved — we only ADD terms.

    Example:
        "my paddy leaves look yellow" 
        → "my paddy leaves look yellow rice nutrient deficiency classification"
    """
    q_lower = query.lower()
    expansions = []

    for synonym, canonical in CROP_SYNONYMS.items():
        if synonym in q_lower and canonical not in q_lower:
            expansions.append(canonical)

    for synonym, canonical in TASK_SYNONYMS.items():
        if synonym in q_lower:
            for term in canonical.split():
                if term not in q_lower:
                    expansions.append(term)

    for synonym, canonical in MODALITY_SYNONYMS.items():
        if synonym in q_lower and canonical not in q_lower:
            expansions.append(canonical)

    if expansions:
        expanded = query + " " + " ".join(dict.fromkeys(expansions))
        return expanded
    return query


def get_constraints(query: str) -> list:
    """
    Return list of tool names that should be EXCLUDED for this query
    based on hard logical constraints.

    Example: satellite imagery query → exclude segmentation/classification tools
    """
    q_lower = query.lower()
    excluded = set()
    for signal, incompatible_tools in HARD_CONSTRAINTS.items():
        if signal in q_lower:
            excluded.update(incompatible_tools)
    return list(excluded)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "My paddy leaves look yellow",
        "Count wheat ears in field photos",
        "Estimate the LAI from canopy images",
        "Analyse Sentinel-2 time series for harvest prediction",
        "Segment arabidopsis rosette leaves",
        "Train a model on my dataset",
        "Plot a bar chart of results",
    ]
    print("Ontology expansion test")
    print("=" * 70)
    for q in test_queries:
        expanded = expand_query(q)
        constraints = get_constraints(q)
        print(f"Original : {q}")
        print(f"Expanded : {expanded}")
        if constraints:
            print(f"Excluded : {constraints}")
        print()
