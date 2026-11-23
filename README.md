# Landmarks-only live inference web app

This Flask app runs the categorized hand/body-pose ST-GCN TFLite models using
normalized landmark JSON as input. The browser can capture a 10-frame gesture
from the camera, detect pose and hands locally with MediaPipe, and send only
landmark coordinates to this server for model inference. Camera video is not
uploaded or saved. Existing landmark JSON files can also be uploaded; those
uploads are validated and processed in memory and are not saved.

## Run locally

From this directory, install the dependencies and start the server:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:10000`. Allow camera access when prompted. Camera access
requires HTTPS or localhost, and the browser must be able to load MediaPipe
from jsDelivr and its task models from Google Cloud Storage. To change the port,
set `PORT`. Set `MODEL_DIR` to use model files outside `models`; relative paths
are resolved from this directory. `MAX_CONTENT_LENGTH` controls the upload
limit in bytes.

Use the **Mirror preview** checkbox to flip the camera display horizontally.
This only changes the preview; captured landmarks and predictions are unchanged.

The live tester captures 10 frames over 2 seconds. It requires a detected hand
and visible nose and shoulders in at least 6 frames, normalizes all 33 pose
points and 21 points for each hand, then posts the landmark JSON to `POST /predict`.
The server keeps the full 150-feature capture for the legacy category-selector
model and selects the 11 trained pose landmarks plus both hands for the newer
53-joint category models.

## Landmark input format

Upload one UTF-8 JSON capture with:

- `schema_version: 1`
- `dataset_type: "normalized_landmark_features"`
- `feature_count: 150`
- `feature_layout`: all 33 pose landmarks followed by all 21 landmarks for each
  of two hands, with x/y values for each point
- `frames`: exactly 10 frames with `sample_index` values 1 through 10 and 150
  finite numeric features per frame

The server reshapes the full capture to float32 input `[1, 2, 10, 75]` (batch,
x/y channels, frames, landmarks). For 53-joint categories, it selects the
trained pose landmarks and hands to produce `[1, 2, 10, 53]`.

## Models

Each category folder under `models/` contains `stgcn_landmarks.tflite` and
`stgcn_landmarks.json`. `alphabet`, `colors`, `family(people)`, `greetings`,
and `numbers` use the new 53-joint models; `category_selector` intentionally
remains on its 75-joint model. The JSON label map is used to turn model scores
into predictions. The app checks each model's input shape and output class
count before inference.

## HTTP API

`POST /predict` accepts either a multipart upload with fields `landmark_file`
and `category`, or an `application/json` request body containing the capture.
For JSON requests, add `"category": "all"` to run every available model, or
select one of `alphabet`, `category_selector`, `colors`, `family(people)`,
`greetings`, or `numbers`.

Successful requests return the source label and each model's top three
predictions. Invalid landmark input returns HTTP 400; unavailable or invalid
models return HTTP 503. `GET /healthz` reports available and missing model
categories.
