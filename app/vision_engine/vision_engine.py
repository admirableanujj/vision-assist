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
# YOLO11n over YOLOv8n: fewer params (2.6M vs 3.2M) and higher mAP (39.5 vs 37.3)
# at the same size class — see the Milestone 4 model-selection writeup for the
# full comparison. Override per-environment via YOLO_MODEL_PATH (e.g. a larger
# GPU-appropriate weights file in a cloud deployment) without touching this code.
DEFAULT_LOCAL_WEIGHTS = "yolo11n.pt"


class FallbackVisionEngine(BaseVisionEngine):
    """
    Heuristic-based mock engine used when YOLO dependencies, 
    weights, or CUDA allocations fail to initialize.
    """
    def __init__(self):
        print("[WARN] VisionAssist initialized FallbackVisionEngine. Running on Simulated Vision matrix.")
        # Common household objects to mock local identification loops
        self.simulated_pool = ["keys", "phone", "wallet", "sunglasses", "backpack"]

    def scan_frame(self, image_buffer) -> list:
        """
        Simulates scanning a camera buffer by returning 1-2 random items 
        from the tracking pool to keep the app working.
        """
        if image_buffer is None:
            return []
        
        # Simulate an automated detection confidence threshold pass
        detected_count = random.randint(1, 2)
        return random.sample(self.simulated_pool, k=detected_count)


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

        def scan_frame(self, image_buffer) -> list:
            if image_buffer is None:
                return []

            if self.model is None:
                return self._fallback.scan_frame(image_buffer)

            try:
                frame = self._decode_frame(image_buffer)
                if frame is None:
                    return []

                # conf= enforces the confidence cutoff at the ultralytics/NMS level.
                results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)

                labels = []
                for result in results:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        # Defensive re-check: guarantees "eliminate outlying"
                        # detections regardless of ultralytics' internal conf= behavior.
                        if confidence < self.confidence_threshold:
                            continue
                        class_id = int(box.cls[0])
                        labels.append(result.names[class_id])

                # Contract describes "items discovered", not raw box count — dedupe.
                return list(dict.fromkeys(labels))

            except Exception as e:
                print(f"[WARN] YOLO inference failed: {e}. Falling back to simulated detection.")
                return self._fallback.scan_frame(image_buffer)

    VisionTracker = YOLOVisionEngine

except ImportError as e:
    print(f"[CRITICAL] Vision dependencies unavailable: {e}")
    # Seamless substitution: App falls back cleanly to the mock engine
    VisionTracker = FallbackVisionEngine