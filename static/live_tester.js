import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

(() => {
  const video = document.getElementById("live-video");
  const overlay = document.getElementById("live-overlay");
  const context = overlay.getContext("2d");
  const drawingUtils = new DrawingUtils(context);
  const categorySelect = document.getElementById("category");
  const cameraStage = document.getElementById("camera-stage");
  const cameraEmpty = document.getElementById("camera-empty");
  const status = document.getElementById("camera-status");
  const handCount = document.getElementById("hand-count");
  const poseCount = document.getElementById("pose-count");
  const frameCount = document.getElementById("frame-count");
  const progress = document.querySelector(".capture-progress");
  const progressFill = document.getElementById("capture-progress-fill");
  const openButton = document.getElementById("open-camera");
  const testButton = document.getElementById("test-gesture");
  const closeButton = document.getElementById("close-camera");
  const resultsSection = document.getElementById("live-results");
  const resultsGrid = document.getElementById("live-results-grid");
  const resultSource = document.getElementById("result-source");
  const featureLayout = [
    ...Array.from({ length: 33 }, (_, index) => [
      `pose_1_${index}_x`,
      `pose_1_${index}_y`,
    ]).flat(),
    ...[1, 2].flatMap((handIndex) =>
      Array.from({ length: 21 }, (_, index) => [
        `hand_${handIndex}_${index}_x`,
        `hand_${handIndex}_${index}_y`,
      ]).flat()
    ),
  ];

  let handLandmarker = null;
  let poseLandmarker = null;
  let cameraStream = null;
  let animationFrame = 0;
  let lastVideoTime = -1;
  let lastTimestamp = 0;
  let testing = false;
  let latestDetection = { hands: [], poses: [], videoTime: -1 };

  const nextFrame = () => new Promise((resolve) => requestAnimationFrame(resolve));

  function setStatus(message, isError = false) {
    status.textContent = message;
    status.dataset.state = isError ? "error" : "";
  }

  function selectedModelIsReady() {
    if (categorySelect.value === "all") {
      return [...categorySelect.options].some((option) => option.dataset.ready === "true");
    }
    return categorySelect.selectedOptions[0]?.dataset.ready === "true";
  }

  function updateControls() {
    const modelReady = selectedModelIsReady();
    openButton.disabled = !handLandmarker || !poseLandmarker || !modelReady || Boolean(cameraStream) || testing;
    testButton.disabled = !cameraStream || !modelReady || testing;
    closeButton.disabled = !cameraStream || testing;
    categorySelect.disabled = testing;
  }

  function updateProgress(completed) {
    const percent = Math.round((completed / 10) * 100);
    progressFill.style.width = `${percent}%`;
    progress.setAttribute("aria-valuenow", String(percent));
    frameCount.textContent = `${completed} / 10`;
  }

  async function loadLandmarkModels() {
    setStatus("Loading hand and pose detection models...");
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      [handLandmarker, poseLandmarker] = await Promise.all([
        HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        }),
        PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        }),
      ]);
      setStatus(selectedModelIsReady()
        ? "Detection models ready. Open the camera to begin."
        : "No available TFLite models are ready. Check model availability below.", !selectedModelIsReady());
    } catch (error) {
      setStatus(`Could not load camera landmark models: ${error.message || error}`, true);
    }
    updateControls();
  }

  function updateDetection() {
    if (!cameraStream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.currentTime === lastVideoTime) {
      return;
    }
    lastVideoTime = video.currentTime;
    lastTimestamp = Math.max(performance.now(), lastTimestamp + 1);
    const handResult = handLandmarker.detectForVideo(video, lastTimestamp);
    const poseResult = poseLandmarker.detectForVideo(video, lastTimestamp);
    latestDetection = {
      hands: (handResult.landmarks || []).map((hand) => hand.map((point) => ({ ...point }))),
      poses: (poseResult.landmarks || []).map((pose) => pose.map((point) => ({ ...point }))),
      videoTime: video.currentTime,
    };
    handCount.textContent = String(latestDetection.hands.length);
    poseCount.textContent = String(latestDetection.poses.length);
  }

  function drawOverlay() {
    context.clearRect(0, 0, overlay.width, overlay.height);
    for (const pose of latestDetection.poses) {
      drawingUtils.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, {
        color: "#ff9d4f",
        lineWidth: 3,
      });
      drawingUtils.drawLandmarks(pose, {
        color: "#ffd38a",
        lineWidth: 1,
        radius: 3,
      });
    }
    for (const hand of latestDetection.hands) {
      drawingUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {
        color: "#4f9dff",
        lineWidth: 3,
      });
      drawingUtils.drawLandmarks(hand, {
        color: "#78ebc4",
        lineWidth: 1,
        radius: 4,
      });
    }
  }

  function drawCameraFrame() {
    if (!cameraStream) return;
    try {
      updateDetection();
      drawOverlay();
    } catch (error) {
      stopCamera();
      setStatus(`Live landmark detection failed: ${error.message || error}`, true);
      return;
    }
    animationFrame = requestAnimationFrame(drawCameraFrame);
  }

  function stopCamera() {
    if (animationFrame) cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      cameraStream = null;
    }
    video.srcObject = null;
    cameraEmpty.hidden = false;
    context.clearRect(0, 0, overlay.width, overlay.height);
    latestDetection = { hands: [], poses: [], videoTime: -1 };
    lastVideoTime = -1;
    handCount.textContent = "0";
    poseCount.textContent = "0";
    updateControls();
  }

  function waitForVideoMetadata() {
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        cleanup();
        reject(new Error("The camera did not provide video frames in time."));
      }, 10000);
      const onReady = () => {
        cleanup();
        resolve();
      };
      const onError = () => {
        cleanup();
        reject(new Error("The browser could not read frames from this camera."));
      };
      const cleanup = () => {
        window.clearTimeout(timeout);
        video.removeEventListener("loadeddata", onReady);
        video.removeEventListener("error", onError);
      };
      video.addEventListener("loadeddata", onReady, { once: true });
      video.addEventListener("error", onError, { once: true });
    });
  }

  openButton.addEventListener("click", async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("Camera access requires HTTPS (or localhost) and a supported browser.", true);
      return;
    }
    openButton.disabled = true;
    setStatus("Opening camera...");
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      video.srcObject = cameraStream;
      await video.play();
      await waitForVideoMetadata();
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      cameraStage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
      cameraEmpty.hidden = true;
      setStatus("Camera ready. Position yourself, then test a gesture.");
      animationFrame = requestAnimationFrame(drawCameraFrame);
    } catch (error) {
      stopCamera();
      setStatus(`Could not open camera: ${error.message || error}`, true);
    }
    updateControls();
  });

  closeButton.addEventListener("click", () => {
    stopCamera();
    setStatus("Camera closed.");
  });

  categorySelect.addEventListener("change", () => {
    updateControls();
    if (handLandmarker && poseLandmarker && selectedModelIsReady()) {
      setStatus(cameraStream ? "Camera ready. Position yourself, then test a gesture." : "Detection models ready. Open the camera to begin.");
    } else {
      setStatus("The selected model category is not available.", true);
    }
  });

  function hasVisiblePoseAnchors(pose) {
    return pose.length > 12
      && [0, 11, 12].every((index) =>
        pose[index] && (pose[index].visibility ?? 0) >= 0.5
      );
  }

  function nearestPopulatedIndex(samples, property, index) {
    const populated = samples
      .map((sample, sampleIndex) => sample[property].length ? sampleIndex : -1)
      .filter((sampleIndex) => sampleIndex >= 0);
    if (populated.length === 0) return -1;
    return populated.sort((left, right) =>
      Math.abs(left - index) - Math.abs(right - index)
      || (left > index ? -1 : 1)
    )[0];
  }

  function coordinateFrame(pose) {
    if (!hasVisiblePoseAnchors(pose)) return null;
    const nose = pose[0];
    const leftShoulder = pose[11];
    const rightShoulder = pose[12];
    if (Math.hypot(rightShoulder.x - leftShoulder.x, rightShoulder.y - leftShoulder.y) < 0.01) {
      return null;
    }
    const left = { x: leftShoulder.x - nose.x, y: leftShoulder.y - nose.y };
    const right = { x: rightShoulder.x - nose.x, y: rightShoulder.y - nose.y };
    const determinant = left.x * right.y - right.x * left.y;
    return Math.abs(determinant) < 0.0001 ? null : { nose, left, right, determinant };
  }

  function relativePoint(point, frame) {
    if (!frame) return [0, 0];
    const dx = point.x - frame.nose.x;
    const dy = point.y - frame.nose.y;
    const leftCoefficient = (dx * frame.right.y - frame.right.x * dy) / frame.determinant;
    const rightCoefficient = (frame.left.x * dy - dx * frame.left.y) / frame.determinant;
    return [
      Math.round((rightCoefficient - leftCoefficient) * 500) / 1000,
      Math.round((leftCoefficient + rightCoefficient) * 1000) / 1000,
    ];
  }

  function buildFeatures(samples) {
    return samples.map((sample, index) => {
      const handSourceIndex = nearestPopulatedIndex(samples, "hands", index);
      const poseSourceIndex = nearestPopulatedIndex(samples, "poses", index);
      const hands = handSourceIndex >= 0 ? samples[handSourceIndex].hands : [];
      const poses = poseSourceIndex >= 0 ? samples[poseSourceIndex].poses : [];
      const pose = poses[0] || [];
      const frame = coordinateFrame(pose);
      const features = [];
      for (let pointIndex = 0; pointIndex < 33; pointIndex++) {
        features.push(...(pose[pointIndex] ? relativePoint(pose[pointIndex], frame) : [0, 0]));
      }
      for (let handIndex = 0; handIndex < 2; handIndex++) {
        const hand = hands[handIndex] || [];
        for (let pointIndex = 0; pointIndex < 21; pointIndex++) {
          features.push(...(hand[pointIndex] ? relativePoint(hand[pointIndex], frame) : [0, 0]));
        }
      }
      return features;
    });
  }

  async function captureWindow() {
    const samples = [];
    let previousVideoTime = -1;
    const startedAt = performance.now();
    for (let index = 0; index < 10; index++) {
      const sampleAt = startedAt + (index + 1) * 200;
      while (performance.now() < sampleAt && cameraStream) await nextFrame();
      while (
        cameraStream
        && latestDetection.videoTime <= previousVideoTime
      ) {
        if (performance.now() - sampleAt > 5000) {
          throw new Error("The camera stopped providing new frames. Check the camera and try again.");
        }
        await nextFrame();
      }
      if (!cameraStream) throw new Error("The camera was closed during capture.");
      samples.push({
        hands: latestDetection.hands.map((hand) => hand.map((point) => ({ ...point }))),
        poses: latestDetection.poses.map((pose) => pose.map((point) => ({ ...point }))),
      });
      previousVideoTime = latestDetection.videoTime;
      updateProgress(samples.length);
    }

    const usableFrames = samples.filter((sample) =>
      sample.hands.length > 0 && hasVisiblePoseAnchors(sample.poses[0] || [])
    ).length;
    if (usableFrames < 6) {
      throw new Error(`Only ${usableFrames} of 10 frames had a detected hand plus visible nose and shoulders. Improve the lighting and framing, then try again.`);
    }
    return {
      schema_version: 1,
      dataset_type: "normalized_landmark_features",
      label: "Live camera",
      feature_count: featureLayout.length,
      feature_layout: featureLayout,
      feature_dtype: "float32",
      coordinate_frame: "nose_origin_shoulders_scaled",
      missing_landmarks: "nearest_detected_frame_then_zero",
      frames: buildFeatures(samples).map((features, index) => ({
        sample_index: index + 1,
        features,
      })),
    };
  }

  function renderResults(data) {
    resultsGrid.replaceChildren();
    for (const [index, result] of data.results.entries()) {
      const card = document.createElement("article");
      card.className = "result-card";
      const topLine = document.createElement("div");
      topLine.className = "card-topline";
      const marker = document.createElement("span");
      marker.className = "model-marker";
      marker.textContent = String(index + 1).padStart(2, "0");
      const title = document.createElement("h3");
      title.textContent = result.title;
      topLine.append(marker, title);
      card.append(topLine);

      const list = document.createElement("ol");
      list.className = "prediction-list";
      for (const [rank, prediction] of result.predictions.entries()) {
        const item = document.createElement("li");
        if (rank === 0) item.className = "top-prediction";
        const position = document.createElement("span");
        position.className = "prediction-rank";
        position.textContent = String(rank + 1);
        const label = document.createElement("span");
        label.className = "prediction-label";
        label.textContent = prediction.label;
        const confidence = document.createElement("span");
        confidence.className = "confidence";
        confidence.textContent = `${Number(prediction.confidence).toFixed(1)}%`;
        item.append(position, label, confidence);
        list.append(item);
      }
      card.append(list);
      resultsGrid.append(card);
    }
    resultSource.textContent = data.source_label && data.source_label !== "Unknown"
      ? `Capture label: ${data.source_label}`
      : "Camera capture";
    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  testButton.addEventListener("click", async () => {
    if (testing) return;
    testing = true;
    updateProgress(0);
    updateControls();
    setStatus("Capturing 10 frames...");
    try {
      const capture = await captureWindow();
      setStatus("Running the selected TFLite model...");
      capture.category = categorySelect.value;
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(capture),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Prediction failed (${response.status}).`);
      renderResults(data);
      setStatus("Prediction complete. Test another gesture whenever you're ready.");
    } catch (error) {
      setStatus(error.message || String(error), true);
    } finally {
      testing = false;
      updateControls();
    }
  });

  window.addEventListener("pagehide", stopCamera);
  loadLandmarkModels();
})();
