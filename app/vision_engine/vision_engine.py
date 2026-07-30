# app/vision_engine/vision_engine.py
"""
YOLO Real-Time Object Tracking Pipeline

Loads optimized ultralytics computer vision weights inside the container framework 
to scan camera inputs, detect registered belongings, and update the session status log.

Usage:
    Imported dynamically into the core framework via:
    from vision_engine.vision_tracker import VisionTracker

Dependencies:
    ultralytics==8.2.0
    torch
    
__original_author__ = "Anujj Saxena"
__license__ = "MIT"      
"""
__author__ = "Anujj Saxena"
__license__ = "MIT"
__version__ = "1.0.1"


import os
import random
from .vision_base import BaseVisionEngine

# Detections below this confidence score are discarded ("eliminate outlying" detections).
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Local default: smallest/fastest variant for CPU-only Docker inference.
# YOLO26n over YOLO11n: fewer params (2.4M vs 2.6M), higher mAP (40.9 vs 39.5),
# and ~30% faster CPU inference (38.9ms vs 56.1ms) at the same size class — see
# the Milestone 4 model-selection writeup for the full comparison. Override
# per-environment via YOLO_MODEL_PATH (e.g. a larger GPU-appropriate weights
# file in a cloud deployment) without touching this code.
DEFAULT_LOCAL_WEIGHTS = "yolo26n.pt"


class FallbackVisionEngine(BaseVisionEngine):
    """
    Heuristic-based mock engine used when YOLO dependencies, 
    weights, or CUDA allocations fail to initialize.
    """
    def __init__(self):
        print("[WARN] VisionAssist initialized FallbackVisionEngine. Running on Simulated Vision matrix.")
        # Common household objects to mock local identification loops
        self.simulated_pool = ["keys", "phone", "wallet", "sunglasses", "backpack"]

    def scan_frame(self, image_buffer) -> dict:
        """
        Simulates scanning a camera buffer by returning 1-2 random items
        from the tracking pool to keep the app working.
        """
        if image_buffer is None:
            return {"detections": [], "annotated_frame": None}

        # Simulate an automated detection confidence threshold pass
        detected_count = random.randint(1, 2)
        labels = random.sample(self.simulated_pool, k=detected_count)
        detections = [
            {"label": label, "confidence": round(random.uniform(0.5, 0.95), 2), "box": None}
            for label in labels
        ]
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        # No real image to draw on in the mock path — annotated_frame stays None.
        return {"detections": detections, "annotated_frame": None}


# --- REAL YOLO SYSTEM BOUNDARY ---
try:
    # Attempt to load your real computer vision stack
    import cv2
    import numpy as np
    from ultralytics import YOLO

    class YOLOVisionEngine(BaseVisionEngine):
        def __init__(self, weights_path=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
            # Resolution order: explicit constructor arg > YOLO_MODEL_PATH env var >
            # hardcoded local default. Mirrors OllamaMLEngine's OLLAMA_HOST pattern —
            # swapping models per environment (local vs. a bigger cloud/GPU variant)
            # is a config change, not a code change.
            self.weights_path = weights_path or os.getenv("YOLO_MODEL_PATH", DEFAULT_LOCAL_WEIGHTS)
            self.confidence_threshold = confidence_threshold
            # Runtime degrade path: reuse the mock engine if real weights fail to
            # load, instead of leaving the app with a crashed/None vision tracker.
            self._fallback = FallbackVisionEngine()
            try:
                self.model = YOLO(self.weights_path)
                print(f"[INFO] YOLOVisionEngine loaded '{self.weights_path}' (conf>={confidence_threshold}).")
            except Exception as e:
                print(f"[WARN] Failed to load YOLO weights '{self.weights_path}': {e}. "
                      f"YOLOVisionEngine will use simulated detections until this is fixed.")
                self.model = None

        def _decode_frame(self, image_buffer):
            """
            Normalize a Streamlit camera_input buffer (bytes-like UploadedFile),
            raw bytes, or an already-decoded np.ndarray into a BGR np.ndarray
            that ultralytics can run inference on.
            """
            if isinstance(image_buffer, np.ndarray):
                return image_buffer
            if hasattr(image_buffer, "getvalue"):
                raw_bytes = image_buffer.getvalue()  # st.camera_input()'s UploadedFile
            elif isinstance(image_buffer, (bytes, bytearray)):
                raw_bytes = image_buffer
            else:
                raw_bytes = image_buffer.read()
            np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        def scan_frame(self, image_buffer) -> dict:
            if image_buffer is None:
                return {"detections": [], "annotated_frame": None}

            if self.model is None:
                return self._fallback.scan_frame(image_buffer)

            try:
                frame = self._decode_frame(image_buffer)
                if frame is None:
                    return {"detections": [], "annotated_frame": None}

                # conf= enforces the confidence cutoff at the ultralytics/NMS level.
                results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)

                detections = []
                for result in results:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        # Defensive re-check: guarantees "eliminate outlying"
                        # detections regardless of ultralytics' internal conf= behavior.
                        if confidence < self.confidence_threshold:
                            continue
                        class_id = int(box.cls[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append({
                            "label": result.names[class_id],
                            "confidence": round(confidence, 4),
                            "box": (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
                        })
                detections.sort(key=lambda d: d["confidence"], reverse=True)

                # Ultralytics draws boxes/labels/confidence directly onto the frame —
                # convert BGR (OpenCV's native order) to RGB so callers (e.g. Streamlit's
                # st.image) don't need to know this engine's internal color convention.
                annotated_frame = None
                if results:
                    annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

                return {"detections": detections, "annotated_frame": annotated_frame}

            except Exception as e:
                print(f"[WARN] YOLO inference failed: {e}. Falling back to simulated detection.")
                return self._fallback.scan_frame(image_buffer)

    VisionTracker = YOLOVisionEngine

except ImportError as e:
    print(f"[CRITICAL] Vision dependencies unavailable: {e}")
    # Seamless substitution: App falls back cleanly to the mock engine
    VisionTracker = FallbackVisionEngine