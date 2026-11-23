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

The live tester captures 10 frames over 2 seconds. It requires a detected hand
and visible nose and shoulders in at least 6 frames, normalizes all 33 pose
points and 21 points for each hand, then posts the landmark JSON to `POST /predict`.
Select one category or all available categories before starting.

## Landmark input format

Upload one UTF-8 JSON capture with:

- `schema_version: 1`
- `dataset_type: "normalized_landmark_features"`
- `feature_count: 150`
- `feature_layout`: all 33 pose landmarks followed by all 21 landmarks for each
  of two hands, with x/y values for each point
- `frames`: exactly 10 frames with `sample_index` values 1 through 10 and 150
  finite numeric features per frame

The features are reshaped to float32 model input `[1, 2, 10, 75]` (batch,
x/y channels, frames, landmarks). This is the schema used by the included
`live json landmarks` captures and matches the included model metadata.

## Models

Each category folder under `models/` contains `stgcn_landmarks.tflite` and
`stgcn_landmarks.json`. The JSON label map is used to turn model scores into
predictions. The app checks the model input shape and output class count before
inference.

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
