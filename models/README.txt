Place the converted category model folders here before deploying:

models/
  alphabet/
    stgcn_landmarks.tflite
    stgcn_landmarks.json
  category_selector/
    stgcn_landmarks.tflite
    stgcn_landmarks.json
  colors/
    stgcn_landmarks.tflite
    stgcn_landmarks.json
  family(people)/
    stgcn_landmarks.tflite
    stgcn_landmarks.json
  greetings/
    stgcn_landmarks.tflite
    stgcn_landmarks.json
  numbers/
    stgcn_landmarks.tflite
    stgcn_landmarks.json

Each JSON file must include the label_to_id mapping generated during TFLite conversion.
