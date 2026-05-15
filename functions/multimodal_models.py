import os
import json
import tempfile
from typing import Annotated, List, Dict, Any

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from utils.registry import pheno_tool


# =========================================================
# YOLO OBJECT DETECTION
# =========================================================

@pheno_tool(
    name="infer_yolo_object_detection",
    description="Run YOLO object detection models for agricultural phenotyping tasks such as rice panicle or wheat spike detection."
)
def infer_yolo_object_detection(
    image_paths: Annotated[List[str], "List of image paths"],
    repo_id: Annotated[str, "HuggingFace model repository ID"],
    conf_threshold: Annotated[float, "Confidence threshold"] = 0.25,
) -> Dict[str, Any]:

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    model_files = [
        x for x in os.listdir(os.path.dirname(config_path))
        if x.endswith(".pt")
    ]

    if len(model_files) == 0:
        raise ValueError("No YOLO checkpoint found.")

    model_path = os.path.join(
        os.path.dirname(config_path),
        model_files[0]
    )

    model = YOLO(model_path)

    outputs = []

    for image_path in image_paths:

        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):

                detections.append({
                    "bbox": box.tolist(),
                    "confidence": float(score),
                    "class_id": int(cls),
                    "class_name": config["classes"][int(cls)]
                })

        outputs.append({
            "image": image_path,
            "num_detections": len(detections),
            "detections": detections
        })

    return {
        "task": "object-detection",
        "repo_id": repo_id,
        "results": outputs
    }


# =========================================================
# TEMPORAL MULTITASK PLACEHOLDER
# =========================================================

@pheno_tool(
    name="infer_temporal_crop_model",
    description="Run temporal crop monitoring models for crop classification, growth prediction, and harvest timing estimation."
)
def infer_temporal_crop_model(
    sequence: Annotated[List[List[float]], "Temporal feature sequence"],
    repo_id: Annotated[str, "HuggingFace model repository ID"],
) -> Dict[str, Any]:

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json"
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    return {
        "task": "temporal-analysis",
        "repo_id": repo_id,
        "message": (
            "Temporal model weights downloaded successfully. "
            "Custom inference architecture required for deployment."
        ),
        "available_heads": list(config["heads"].keys()),
        "input_shape": config["input_shape"],
    }