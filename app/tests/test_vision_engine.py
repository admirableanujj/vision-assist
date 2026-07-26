import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from vision_engine.vision_base import BaseVisionEngine
from vision_engine.vision_engine import (
    FallbackVisionEngine,
    YOLOVisionEngine,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LOCAL_WEIGHTS,
)


class FakeBox:
    def __init__(self, conf, cls_id):
        self.conf = [conf]
        self.cls = [cls_id]


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class TestABCContract:
    def test_subclass_missing_scan_frame_raises_type_error(self):
        class Incomplete(BaseVisionEngine):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_implementation_instantiates(self):
        assert isinstance(FallbackVisionEngine(), BaseVisionEngine)


class TestFallbackVisionEngine:
    def test_none_buffer_returns_empty_list(self):
        assert FallbackVisionEngine().scan_frame(None) == []

    def test_returns_items_from_known_pool(self):
        eng = FallbackVisionEngine()
        result = eng.scan_frame(b"fake-bytes")
        assert 1 <= len(result) <= 2
        assert set(result).issubset(set(eng.simulated_pool))


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
        assert set(result).issubset(set(eng._fallback.simulated_pool))


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

    def test_none_buffer_returns_empty_list(self):
        eng = self._make_engine(MagicMock())
        assert eng.scan_frame(None) == []

    def test_filters_out_low_confidence_detections(self):
        mock_model = MagicMock()
        names = {0: "keys", 1: "phone"}
        mock_model.predict.return_value = [
            FakeResult([FakeBox(0.91, 0), FakeBox(0.10, 1)], names)
        ]
        eng = self._make_engine(MagicMock(return_value=mock_model))

        result = eng.scan_frame(np.zeros((10, 10, 3), dtype=np.uint8))

        assert result == ["keys"]
        _, kwargs = mock_model.predict.call_args
        assert kwargs["conf"] == DEFAULT_CONFIDENCE_THRESHOLD

    def test_dedupes_repeated_labels(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            FakeResult([FakeBox(0.9, 0), FakeBox(0.8, 0)], {0: "keys"})
        ]
        eng = self._make_engine(MagicMock(return_value=mock_model))

        assert eng.scan_frame(np.zeros((10, 10, 3), dtype=np.uint8)) == ["keys"]

    def test_inference_exception_falls_back_to_mock_pool(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CPU OOM")
        eng = self._make_engine(MagicMock(return_value=mock_model))

        result = eng.scan_frame(np.zeros((10, 10, 3), dtype=np.uint8))

        assert set(result).issubset(set(eng._fallback.simulated_pool))

    def test_accepts_streamlit_like_uploaded_file_buffer(self):
        # Simulates st.camera_input()'s UploadedFile: bytes-like object with .getvalue()
        fake_uploaded_file = MagicMock()
        fake_uploaded_file.getvalue.return_value = b"\x89PNG-fake-bytes"

        mock_model = MagicMock()
        mock_model.predict.return_value = [FakeResult([], {})]
        eng = self._make_engine(MagicMock(return_value=mock_model))

        result = eng.scan_frame(fake_uploaded_file)

        assert result == []
        fake_uploaded_file.getvalue.assert_called_once()
        mock_model.predict.assert_called_once()
