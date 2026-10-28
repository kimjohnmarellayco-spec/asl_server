from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename


WEB_DIR = Path(__file__).resolve().parent
load_dotenv(WEB_DIR / ".env")

CATEGORIES = (
    "alphabet",
    "category_selector",
    "colors",
    "family(people)",
    "greetings",
    "numbers",
)
CATEGORY_TITLES = {
    "alphabet": "Alphabet",
    "category_selector": "Category selector",
    "colors": "Colors",
    "family(people)": "Family / people",
    "greetings": "Greetings",
    "numbers": "Numbers",
}
INPUT_SHAPE = (1, 2, 10, 75)
DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 * 1024


def configured_upload_limit() -> int:
    value = os.environ.get("MAX_CONTENT_LENGTH", str(DEFAULT_MAX_UPLOAD_BYTES))
    try:
        limit = int(value)
    except ValueError as error:
        raise ValueError("MAX_CONTENT_LENGTH must be an integer number of bytes.") from error
    if limit <= 0:
        raise ValueError("MAX_CONTENT_LENGTH must be greater than zero.")
    return limit


def configured_model_dir() -> Path:
    configured = Path(os.environ.get("MODEL_DIR", "models"))
    if not configured.is_absolute():
        configured = WEB_DIR / configured
    return configured.resolve()


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = configured_upload_limit()
MODEL_DIR = configured_model_dir()


@dataclass
class ModelRuntime:
    interpreter: Any
    input_detail: dict[str, Any]
    output_detail: dict[str, Any]
    label_to_id: dict[str, int]
    lock: threading.Lock


def model_paths(category: str) -> tuple[Path, Path]:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    category_dir = (MODEL_DIR / category).resolve()
    if category_dir.parent != MODEL_DIR:
        raise ValueError(f"Invalid model directory for category {category}.")
    return (
        category_dir / "stgcn_landmarks.tflite",
        category_dir / "stgcn_landmarks.json",
    )


def available_categories() -> dict[str, bool]:
    statuses = {}
    for category in CATEGORIES:
        model_path, labels_path = model_paths(category)
        statuses[category] = model_path.is_file() and labels_path.is_file()
    return statuses


def validate_label_mapping(metadata: Any, metadata_path: Path) -> dict[str, int]:
    if not isinstance(metadata, dict):
        raise ValueError(f"Label metadata must be a JSON object: {metadata_path.name}")
    label_to_id = metadata.get("label_to_id")
    if (
        not isinstance(label_to_id, dict)
        or not label_to_id
        or any(
            not isinstance(label, str)
            or type(class_id) is not int
            for label, class_id in label_to_id.items()
        )
        or sorted(label_to_id.values()) != list(range(len(label_to_id)))
    ):
        raise ValueError(f"Invalid label_to_id mapping in {metadata_path.name}.")
    return label_to_id


@lru_cache(maxsize=len(CATEGORIES))
def load_model(category: str) -> ModelRuntime:
    model_path, metadata_path = model_paths(category)
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing TFLite model: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing label metadata: {metadata_path}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {metadata_path.name}: {error.msg}") from error
    label_to_id = validate_label_mapping(metadata, metadata_path)

    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as error:
        raise RuntimeError(
            "LiteRT runtime is not installed. Install web/requirements.txt."
        ) from error

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError(f"Expected one input and one output in {model_path.name}.")

    input_detail = input_details[0]
    output_detail = output_details[0]
    if tuple(int(dimension) for dimension in input_detail["shape"]) != INPUT_SHAPE:
        raise ValueError(
            f"Unexpected input shape for {model_path.name}: "
            f"{input_detail['shape']}; expected {INPUT_SHAPE}."
        )
    if np.dtype(input_detail["dtype"]) != np.dtype(np.float32):
        raise ValueError(f"{model_path.name} must accept float32 input.")
    if int(np.prod(output_detail["shape"])) != len(label_to_id):
        raise ValueError(
            f"Output size in {model_path.name} does not match its label map."
        )
    return ModelRuntime(
        interpreter=interpreter,
        input_detail=input_detail,
        output_detail=output_detail,
        label_to_id=label_to_id,
        lock=threading.Lock(),
    )


def parse_landmark_json(raw: bytes) -> tuple[np.ndarray, str]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("The uploaded file is not valid UTF-8 JSON.") from error

    if not isinstance(data, dict):
        raise ValueError("The JSON root must be an object.")
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported landmark JSON schema_version; expected 1.")
    if data.get("dataset_type") != "normalized_landmark_features":
        raise ValueError(
            "Expected dataset_type 'normalized_landmark_features'."
        )
    if data.get("feature_count") != 150:
        raise ValueError("Expected feature_count 150.")

    frames = data.get("frames")
    if not isinstance(frames, list) or len(frames) != 10:
        raise ValueError("Expected exactly 10 landmark frames.")
    try:
        ordered_frames = sorted(frames, key=lambda frame: int(frame["sample_index"]))
        sample_indices = [int(frame["sample_index"]) for frame in ordered_frames]
        features = np.asarray(
            [frame["features"] for frame in ordered_frames],
            dtype=np.float32,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Each frame must contain an integer sample_index and a features array."
        ) from error

    if sample_indices != list(range(1, 11)):
        raise ValueError("Frame sample_index values must be 1 through 10.")
    if features.shape != (10, 150):
        raise ValueError(
            f"Expected features shaped (10, 150), received {features.shape}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Landmark features contain NaN or infinite values.")

    label = data.get("label")
    if not isinstance(label, str) or not label.strip():
        label = "Unknown"
    model_input = features.reshape(1, 10, 75, 2).transpose(0, 3, 1, 2).copy()
    return model_input, label.strip()


def run_prediction(category: str, model_input: np.ndarray) -> dict[str, Any]:
    runtime = load_model(category)
    with runtime.lock:
        runtime.interpreter.set_tensor(
            runtime.input_detail["index"], model_input.astype(np.float32, copy=False)
        )
        runtime.interpreter.invoke()
        logits = np.asarray(
            runtime.interpreter.get_tensor(runtime.output_detail["index"]),
            dtype=np.float64,
        ).reshape(-1)

    if logits.size != len(runtime.label_to_id):
        raise ValueError(f"Unexpected output size from the {category} model.")
    if not np.isfinite(logits).all():
        raise ValueError(f"The {category} model returned invalid scores.")

    shifted = logits - np.max(logits)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum()
    id_to_label = {class_id: label for label, class_id in runtime.label_to_id.items()}
    top_indices = np.argsort(probabilities)[::-1][: min(3, len(probabilities))]
    predictions = [
        {
            "label": id_to_label[int(index)],
            "confidence": float(probabilities[index]) * 100,
        }
        for index in top_indices
    ]
    return {
        "category": category,
        "title": CATEGORY_TITLES[category],
        "predictions": predictions,
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        categories=CATEGORIES,
        category_titles=CATEGORY_TITLES,
        statuses=available_categories(),
        results=None,
        filename=None,
        source_label=None,
        error=None,
        selected_category="",
    )


@app.post("/predict")
def predict():
    statuses = available_categories()
    uploaded_file = request.files.get("landmark_file")
    selected_category = request.form.get("category", "")
    filename = secure_filename(uploaded_file.filename) if uploaded_file else ""

    try:
        if selected_category != "all" and selected_category not in CATEGORIES:
            raise ValueError("Select a valid model category.")
        if uploaded_file is None or not uploaded_file.filename:
            raise ValueError("Choose a landmark JSON file to upload.")
        if not filename.lower().endswith(".json"):
            raise ValueError("Upload a file with the .json extension.")

        model_input, source_label = parse_landmark_json(uploaded_file.read())
        categories = CATEGORIES if selected_category == "all" else (selected_category,)
        results = []
        for category in categories:
            results.append(run_prediction(category, model_input))

        return render_template(
            "index.html",
            categories=CATEGORIES,
            category_titles=CATEGORY_TITLES,
            statuses=statuses,
            results=results,
            filename=filename,
            source_label=source_label,
            error=None,
            selected_category=selected_category,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        return (
            render_template(
                "index.html",
                categories=CATEGORIES,
                category_titles=CATEGORY_TITLES,
                statuses=statuses,
                results=None,
                filename=filename or None,
                source_label=None,
                error=str(error),
                selected_category=selected_category,
            ),
            400,
        )


@app.get("/healthz")
def healthz():
    statuses = available_categories()
    return {
        "status": "ok",
        "available_models": [category for category, ready in statuses.items() if ready],
        "missing_models": [category for category, ready in statuses.items() if not ready],
    }


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    upload_limit = app.config["MAX_CONTENT_LENGTH"]
    if upload_limit < 1024 * 1024:
        limit_description = f"{(upload_limit + 1023) // 1024} KB"
    else:
        limit_description = f"{(upload_limit + 1024 * 1024 - 1) // (1024 * 1024)} MB"
    return (
        render_template(
            "index.html",
            categories=CATEGORIES,
            category_titles=CATEGORY_TITLES,
            statuses=available_categories(),
            results=None,
            filename=None,
            source_label=None,
            error=(
                f"The uploaded file exceeds the {limit_description} limit."
            ),
            selected_category="",
        ),
        413,
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
        debug=False,
    )
