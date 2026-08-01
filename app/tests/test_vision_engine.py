import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from vision_engine.vision_base import BaseVisionEngine
from vision_engine.vision_engine import (
    FallbackVisionEngine,
    YOLOVisionEngine,
    YOLOEVisionEngine,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LOCAL_WEIGHTS,
    DEFAULT_YOLOE_WEIGHTS,
    DEFAULT_CUSTOM_CLASSES,
)


class FakeXYXY(list):
    """Stands in for ultralytics' xyxy tensor, which exposes .tolist()."""
    def tolist(self):
        return list(self)


class FakeBox:
    def __init__(self, conf, cls_id, xyxy=(0.0, 0.0, 10.0, 10.0)):
        self.conf = [conf]
        self.cls = [cls_id]
        self.xyxy = [FakeXYXY(xyxy)]


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names

    def plot(self):
        return np.zeros((10, 10, 3), dtype=np.uint8)


class TestABCContract:
    def test_subclass_missing_scan_frame_raises_type_error(self):
        class Incomplete(BaseVisionEngine):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_implementation_instantiates(self):
        assert isinstance(FallbackVisionEngine(), BaseVisionEngine)


class TestFallbackVisionEngine:
    def test_none_buffer_returns_empty_result(self):
        assert FallbackVisionEngine().scan_frame(None) == {"detections": [], "annotated_frame": None}

    def test_returns_items_from_known_pool(self):
        eng = FallbackVisionEngine()
        result = eng.scan_frame(b"fake-bytes")
        detections = result["detections"]
        assert 1 <= len(detections) <= 2
        assert result["annotated_frame"] is None
        for d in detections:
            assert d["label"] in eng.simulated_pool
            assert 0.0 <= d["confidence"] <= 1.0
            assert d["box"] is None

    def test_detections_sorted_by_confidence_descending(self):
        eng = FallbackVisionEngine()
        detections = eng.scan_frame(b"fake-bytes")["detections"]
        confidences = [d["confidence"] for d in detections]
        assert confidences == sorted(confidences, reverse=True)


class TestYOLOVisionEngineInit:
    def test_model_loads_once_with_default_threshold(self):
        with patch("vision_engine.vision_engine.YOLO") as mock_yolo_cls:
            eng = YOLOVisionEngine()
        mock_yolo_cls.assert_called_once_with(DEFAULT_LOCAL_WEIGHTS)
        assert eng.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
        assert eng.model is not None

    def test_custom_threshold_override(self):
        with patch("vision_engine.vision_engine.YOLO"):
            eng = YOLOVisionEngine(confidence_threshold=0.75)
        assert eng.confidence_threshold == 0.75

    def test_falls_back_gracefully_when_weights_fail_to_load(self):
        with patch("vision_engine.vision_engine.YOLO", side_effect=Exception("no network")):
            eng = YOLOVisionEngine()
        assert eng.model is None
        result = eng.scan_frame(b"fake-bytes")
        labels = {d["label"] for d in result["detections"]}
        assert labels.issubset(set(eng._fallback.simulated_pool))


class TestModelPathResolution:
    """
    Model choice is a config concern, not a code concern: local vs. a bigger
    cloud/GPU variant is selected via YOLO_MODEL_PATH, mirroring the OLLAMA_HOST
    resolution pattern already used in OllamaMLEngine.
    """

    def test_falls_back_to_hardcoded_default_when_no_env_var_or_arg(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("vision_engine.vision_engine.YOLO") as mock_yolo_cls:
            eng = YOLOVisionEngine()
        assert eng.weights_path == DEFAULT_LOCAL_WEIGHTS
        mock_yolo_cls.assert_called_once_with(DEFAULT_LOCAL_WEIGHTS)

    def test_env_var_used_when_no_explicit_arg(self):
        with patch.dict(os.environ, {"YOLO_MODEL_PATH": "yolo11m.pt"}, clear=True), \
             patch("vision_engine.vision_engine.YOLO") as mock_yolo_cls:
            eng = YOLOVisionEngine()
        assert eng.weights_path == "yolo11m.pt"
        mock_yolo_cls.assert_called_once_with("yolo11m.pt")

    def test_explicit_constructor_arg_overrides_env_var(self):
        with patch.dict(os.environ, {"YOLO_MODEL_PATH": "yolo11m.pt"}, clear=True), \
             patch("vision_engine.vision_engine.YOLO") as mock_yolo_cls:
            eng = YOLOVisionEngine(weights_path="yolov10n.pt")
        assert eng.weights_path == "yolov10n.pt"
        mock_yolo_cls.assert_called_once_with("yolov10n.pt")


class TestYOLOVisionEngineScanFrame:
    def _make_engine(self, mock_yolo_cls):
        with patch("vision_engine.vision_engine.YOLO", mock_yolo_cls):
            return YOLOVisionEngine()

    def test_none_buffer_returns_empty_result(self):
        eng = self._make_engine(MagicMock())
        assert eng.scan_frame(None) == {"detections": [], "annotated_frame": None}

    def test_filters_out_low_confidence_detections_and_reports_box(self):
        mock_model = MagicMock()
        names = {0: "keys", 1: "phone"}
        mock_model.predict.return_value = [
            FakeResult(
                [FakeBox(0.91, 0, xyxy=(10.0, 20.0, 30.0, 40.0)), FakeBox(0.10, 1)],
                names,
            )
        ]
        eng = self._make_engine(MagicMock(return_value=mock_model))

        result = eng.scan_frame(np.zeros((10, 10, 3), dtype=np.uint8))

        assert result["detections"] == [
            {"label": "keys", "confidence": 0.91, "box": (10.0, 20.0, 30.0, 40.0)}
        ]
        assert result["annotated_frame"] is not None
        _, kwargs = mock_model.predict.call_args
        assert kwargs["conf"] == DEFAULT_CONFIDENCE_THRESHOLD

    def test_keeps_repeated_labels_as_separate_detections(self):
        # Two distinct boxes of the same class (e.g. two keys in frame) must stay
        # separate now that each detection carries its own confidence/box — unlike
        # the old list[str] contract, which deduped by label.
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            FakeResult(
                [FakeBox(0.9, 0, xyxy=(0, 0, 5, 5)), FakeBox(0.8, 0, xyxy=(5, 5, 10, 10))],
                {0: "keys"},
            )
        ]
        eng = self._make_engine(MagicMock(return_value=mock_model))

        detections = eng.scan_frame(np.zeros((10, 10, 3), dtype=np.uint8))["detections"]

        assert len(detections) == 2
        assert [d["label"] for d in detections] == ["keys", "keys"]

    def test_detections_sorted_by_confidence_descending(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            FakeResult(
                [FakeBox(0.6, 0), FakeBox(0.95, 1)],
                {0: "keys", 1: "phone"},
            )
        ]
        eng = self._make_engine(MagicMock(return_value=mock_model))

        detections = eng.scan_frame(np.zeros((10, 10, 3), dtype=np.uint8))["detections"]

        assert [d["label"] for d in detections] == ["phone", "keys"]

    def test_inference_exception_falls_back_to_mock_pool(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CPU OOM")
        eng = self._make_engine(MagicMock(return_value=mock_model))

        result = eng.scan_frame(np.zeros((10, 10, 3), dtype=np.uint8))

        labels = {d["label"] for d in result["detections"]}
        assert labels.issubset(set(eng._fallback.simulated_pool))

    def test_accepts_streamlit_like_uploaded_file_buffer(self):
        # Simulates st.camera_input()'s UploadedFile: bytes-like object with .getvalue()
        fake_uploaded_file = MagicMock()
        fake_uploaded_file.getvalue.return_value = b"\x89PNG-fake-bytes"

        mock_model = MagicMock()
        mock_model.predict.return_value = [FakeResult([], {})]
        eng = self._make_engine(MagicMock(return_value=mock_model))

        result = eng.scan_frame(fake_uploaded_file)

        assert result["detections"] == []
        fake_uploaded_file.getvalue.assert_called_once()
        mock_model.predict.assert_called_once()


class TestYOLOEVisionEngineInit:
    """
    YOLOEVisionEngine is the open-vocabulary counterpart to YOLOVisionEngine —
    same scan_frame() contract (shared via _UltralyticsScanMixin), but loads a
    different ultralytics class and must call set_classes() after loading.
    """

    def test_loads_with_default_weights_and_classes(self):
        with patch("vision_engine.vision_engine.YOLOE") as mock_yoloe_cls:
            eng = YOLOEVisionEngine()
        mock_yoloe_cls.assert_called_once_with(DEFAULT_YOLOE_WEIGHTS)
        eng.model.set_classes.assert_called_once_with(DEFAULT_CUSTOM_CLASSES)
        assert eng.classes == DEFAULT_CUSTOM_CLASSES
        assert eng.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD

    def test_custom_classes_constructor_arg_overrides_default(self):
        with patch("vision_engine.vision_engine.YOLOE"):
            eng = YOLOEVisionEngine(classes=["shoe", "hat"])
        eng.model.set_classes.assert_called_once_with(["shoe", "hat"])
        assert eng.classes == ["shoe", "hat"]

    def test_custom_classes_env_var_used_when_no_explicit_arg(self):
        with patch.dict(os.environ, {"VISION_CUSTOM_CLASSES": "keys, wallet ,sunglasses"}, clear=True), \
             patch("vision_engine.vision_engine.YOLOE"):
            eng = YOLOEVisionEngine()
        # Whitespace around each comma-separated entry must be trimmed.
        assert eng.classes == ["keys", "wallet", "sunglasses"]
        eng.model.set_classes.assert_called_once_with(["keys", "wallet", "sunglasses"])

    def test_weights_path_resolution_matches_yolo_pattern(self):
        with patch.dict(os.environ, {"YOLO_MODEL_PATH": "yoloe-26l-seg.pt"}, clear=True), \
             patch("vision_engine.vision_engine.YOLOE") as mock_yoloe_cls:
            eng = YOLOEVisionEngine()
        assert eng.weights_path == "yoloe-26l-seg.pt"
        mock_yoloe_cls.assert_called_once_with("yoloe-26l-seg.pt")

    def test_falls_back_gracefully_when_weights_fail_to_load(self):
        with patch("vision_engine.vision_engine.YOLOE", side_effect=Exception("no network")):
            eng = YOLOEVisionEngine()
        assert eng.model is None
        result = eng.scan_frame(b"fake-bytes")
        labels = {d["label"] for d in result["detections"]}
        assert labels.issubset(set(eng._fallback.simulated_pool))

    def test_falls_back_gracefully_when_set_classes_raises(self):
        # Loading succeeds but the open-vocabulary prompt step fails — must not
        # leave the engine half-initialized with a model that was never prompted.
        mock_model = MagicMock()
        mock_model.set_classes.side_effect = RuntimeError("prompt encoder failed")
        with patch("vision_engine.vision_engine.YOLOE", return_value=mock_model):
            eng = YOLOEVisionEngine()
        assert eng.model is None


class TestYOLOEVisionEngineScanFrame:
    """
    scan_frame() itself is shared with YOLOVisionEngine via _UltralyticsScanMixin
    and already covered thoroughly there — these confirm the mixin actually wires
    up correctly for YOLOE too, not a full re-test of every branch.
    """

    def _make_engine(self, mock_model):
        with patch("vision_engine.vision_engine.YOLOE", return_value=mock_model):
            return YOLOEVisionEngine()

    def test_detects_a_custom_class_not_in_coco(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            FakeResult([FakeBox(0.82, 0, xyxy=(1.0, 2.0, 3.0, 4.0))], {0: "wallet"})
        ]
        eng = self._make_engine(mock_model)

        result = eng.scan_frame(np.zeros((10, 10, 3), dtype=np.uint8))

        assert result["detections"] == [
            {"label": "wallet", "confidence": 0.82, "box": (1.0, 2.0, 3.0, 4.0)}
        ]
        assert result["annotated_frame"] is not None

    def test_inference_exception_falls_back_to_mock_pool(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CPU OOM")
        eng = self._make_engine(mock_model)

        result = eng.scan_frame(np.zeros((10, 10, 3), dtype=np.uint8))

        labels = {d["label"] for d in result["detections"]}
        assert labels.issubset(set(eng._fallback.simulated_pool))
