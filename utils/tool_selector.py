"""
utils/tool_selector.py
======================
Contribution 2 — Tool Selector Agent (Hybrid Retrieval)

Combines semantic similarity with keyword boosting for higher accuracy.
Final score = 0.7 * semantic_score + 0.3 * keyword_boost

This is neuro-symbolic retrieval:
  - Neural: sentence-transformers cosine similarity
  - Symbolic: domain keyword matching
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager

ALWAYS_INCLUDE = {"make_dir", "get_model_zoo", "get_best_model"}

# Keyword signals per tool — domain language that semantic alone may miss
KEYWORD_BOOSTS = {
    "infer_image_classification": [
        "classify", "classification", "deficiency", "nutrient", "identify",
        "stress", "disease", "paddy", "yellowing", "leaf colour", "nitrogen"
    ],
    "infer_instance_segmentation": [
        "segment", "segmentation", "count leaves", "rosette", "instance",
        "individual leaves", "leaf boundary", "mask", "organ"
    ],
    "infer_object_detection": [
        "detect", "count", "spike", "panicle", "head", "ear", "localize",
        "bounding box", "how many", "number of"
    ],
    "infer_image_regression": [
        "predict", "estimate", "regression", "continuous", "leaf area index",
        "height", "lai", "biomass", "fresh weight", "dry weight"
    ],
    "infer_temporal_analysis": [
        "sentinel", "satellite", "time series", "temporal", "harvest",
        "multitemporal", "multi-temporal", "crop mapping", "growth rate",
        "seasonal", "lombardia"
    ],
    "coding": [
        "code", "plot", "chart", "script", "python", "bar chart",
        "visualize", "write", "function", "csv", "merge", "compute"
    ],
    "finetune_image_classification": [
        "train", "finetune", "fine-tune", "my dataset", "new model",
        "custom", "retrain"
    ],
    "finetune_instance_segmentation": [
        "train segmentation", "finetune segmentation", "fine-tune segmentation",
        "train a new", "uploaded dataset"
    ],
    "perform_anova": [
        "anova", "statistical test", "significance", "treatment",
        "compare groups", "variance"
    ],
    "perform_tukey_test": [
        "tukey", "post-hoc", "posthoc", "pairwise", "multiple comparison"
    ],
    "RAG": [
        "what is", "explain", "how does", "describe", "paper", "literature",
        "phenotiki", "system", "method", "protocol"
    ],
    "plot_from_csv": [
        "plot", "visualize", "graph", "chart", "bar", "line", "scatter"
    ],
    "query_csv": [
        "query", "filter", "find rows", "highest", "lowest", "which plant",
        "how many rows"
    ],
    "compute_from_csv": [
        "compute", "calculate", "average", "mean", "sum", "aggregate"
    ],
}

SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3


def _keyword_score(query: str, tool_name: str) -> float:
    """Returns 1.0 if any keyword matches, 0.0 otherwise."""
    keywords = KEYWORD_BOOSTS.get(tool_name, [])
    q = query.lower()
    return 1.0 if any(kw in q for kw in keywords) else 0.0


class ToolSelectorIndex:
    def __init__(self, manager, model_name: str = "all-mpnet-base-v2"):
        self.manager = manager
        self.encoder = SentenceTransformer(model_name)
        self._build_index()

    def _build_index(self):
        tools = self.manager.llm_config["tools"]
        self.tools = tools
        self.tool_names = []
        self.tool_texts = []

        for t in tools:
            fn = t["function"]
            name = fn["name"]
            desc = fn.get("description", "")
            params = list(fn.get("parameters", {}).get("properties", {}).keys())
            text = f"{name}. {desc}. Parameters: {', '.join(params)}"
            self.tool_names.append(name)
            self.tool_texts.append(text)

        print(f"[ToolSelector] Encoding {len(self.tool_texts)} tools...")
        self.embeddings = self.encoder.encode(
            self.tool_texts, normalize_embeddings=True
        )
        print(f"[ToolSelector] Index ready. Shape: {self.embeddings.shape}")

    def get_top_k(self, query: str, k: int = 7) -> list:
        from utils.ontology import expand_query, get_constraints
        excluded = get_constraints(query)
        query = expand_query(query)
        query_vec = self.encoder.encode([query], normalize_embeddings=True)
        semantic_scores = (self.embeddings @ query_vec.T).flatten()

        # Hybrid score = semantic + keyword boost
        hybrid_scores = np.array([
            SEMANTIC_WEIGHT * semantic_scores[i] +
            KEYWORD_WEIGHT * _keyword_score(query, self.tool_names[i])
            for i in range(len(self.tool_names))
        ])

        ranked_indices = np.argsort(hybrid_scores)[::-1].tolist()
        selected_names = set()
        result = []
        pinned = []

        # Always-include tools first
        for i, name in enumerate(self.tool_names):
            if name in ALWAYS_INCLUDE:
                result.append(self.tools[i])
                selected_names.add(name)
                pinned.append(name)

        # Top-k by hybrid score (respecting hard constraints)
        added = 0
        for idx in ranked_indices:
            if added >= k:
                break
            name = self.tool_names[idx]
            if name not in selected_names and name not in excluded:
                result.append(self.tools[idx])
                selected_names.add(name)
                added += 1

        print(f"[ToolSelector] Query: '{query[:60]}'")
        print(f"[ToolSelector] Top-{k}: {[t['function']['name'] for t in result if t['function']['name'] not in ALWAYS_INCLUDE]}")
        return result

    def score_debug(self, query: str):
        """Show both semantic and hybrid scores for all tools."""
        query_vec = self.encoder.encode([query], normalize_embeddings=True)
        semantic_scores = (self.embeddings @ query_vec.T).flatten()

        rows = []
        for i, name in enumerate(self.tool_names):
            sem = float(semantic_scores[i])
            kw = _keyword_score(query, name)
            hybrid = SEMANTIC_WEIGHT * sem + KEYWORD_WEIGHT * kw
            rows.append((name, sem, kw, hybrid))

        rows.sort(key=lambda x: x[3], reverse=True)
        print(f"\nQuery: '{query}'")
        print(f"{'Rank':<5} {'Hybrid':<8} {'Semantic':<10} {'KW':<5} Tool")
        print("-" * 65)
        for rank, (name, sem, kw, hybrid) in enumerate(rows, 1):
            pin = " [pinned]" if name in ALWAYS_INCLUDE else ""
            print(f"{rank:<5} {hybrid:.4f}   {sem:.4f}     {kw:.1f}   {name}{pin}")

    @contextmanager
    def patch_manager(self, query: str, k: int = 7):
        full_tools = self.manager.llm_config["tools"]
        filtered = self.get_top_k(query, k=k)
        try:
            self.manager.llm_config["tools"] = filtered
            yield filtered
        finally:
            self.manager.llm_config["tools"] = full_tools
