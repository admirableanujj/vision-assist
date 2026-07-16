import os
import pytest
from unittest.mock import MagicMock, patch

from ml_engine.ml_base_engine import BaseMLEngine
from ml_engine.ml_engine import OllamaMLEngine
from ml_engine.query_classifier import QueryClassifier


# ─── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """
    OllamaMLEngine with all external connections mocked.
    Env is cleared so fallback_enabled=False (no OpenAI key) by default.
    """
    mock_client = MagicMock()
    with patch("ml_engine.ml_engine.ollama.Client", return_value=mock_client), \
         patch.dict(os.environ, {}, clear=True):
        eng = OllamaMLEngine()
    eng.local_client = mock_client
    return eng, mock_client


@pytest.fixture
def classifier():
    """QueryClassifier backed by a mock Ollama client."""
    mock_client = MagicMock()
    return QueryClassifier(mock_client), mock_client


# ─── ABC Contract ──────────────────────────────────────────────────────────────

class TestABCContract:
    def test_subclass_missing_generate_general_response_raises_type_error(self):
        """The ABC must not allow instantiation without generate_general_response."""
        class Incomplete(BaseMLEngine):
            def tokenize_text(self, text): return []
            def generate_response(self, text, ctx): return ""
            # generate_general_response intentionally absent

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_implementation_instantiates_without_error(self, engine):
        eng, _ = engine
        assert isinstance(eng, OllamaMLEngine)


# ─── Hostname Resolution ───────────────────────────────────────────────────────

class TestHostnameResolution:
    def _make(self, env):
        mock_client = MagicMock()
        with patch("ml_engine.ml_engine.ollama.Client", return_value=mock_client), \
             patch.dict(os.environ, env, clear=True):
            return OllamaMLEngine(), mock_client

    def test_env_var_takes_precedence_over_default(self):
        eng, _ = self._make({"OLLAMA_HOST": "http://custom-host:9999"})
        assert eng.host == "http://custom-host:9999"

    def test_falls_back_to_hardcoded_default_when_no_env_var(self):
        eng, _ = self._make({})
        assert eng.host == "http://vision_assist_llm_local:11434"

    def test_explicit_constructor_arg_overrides_env_var(self):
        mock_client = MagicMock()
        with patch("ml_engine.ml_engine.ollama.Client", return_value=mock_client), \
             patch.dict(os.environ, {"OLLAMA_HOST": "http://env-host:11434"}, clear=True):
            eng = OllamaMLEngine(host="http://explicit-host:11434")
        assert eng.host == "http://explicit-host:11434"

    def test_langchain_initialized_when_openai_key_present(self):
        """LangChain fallback is set up in __init__ when OPENAI_API_KEY is set."""
        mock_client = MagicMock()
        mock_llm_cls = MagicMock()
        with patch("ml_engine.ml_engine.ollama.Client", return_value=mock_client), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True), \
             patch("langchain_openai.ChatOpenAI", mock_llm_cls):
            eng = OllamaMLEngine()
        assert eng.fallback_enabled is True
        assert eng.fallback_llm is not None
        mock_llm_cls.assert_called_once_with(model="gpt-4o-mini", temperature=0.1)

    def test_fallback_disabled_when_chatopenai_init_raises(self):
        """
        If ChatOpenAI() construction fails (bad key format, package issue, etc.),
        fallback_enabled must stay False so callers never see fallback_enabled=True
        with a missing fallback_llm (which would raise AttributeError downstream).
        """
        mock_client = MagicMock()
        mock_llm_cls = MagicMock(side_effect=Exception("invalid api key"))
        with patch("ml_engine.ml_engine.ollama.Client", return_value=mock_client), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "sk-bad"}, clear=True), \
             patch("langchain_openai.ChatOpenAI", mock_llm_cls):
            eng = OllamaMLEngine()

        assert eng.fallback_enabled is False
        assert eng.fallback_llm is None

        # generate_general_response must not raise AttributeError — it should
        # go straight to the local Ollama path instead.
        eng.local_client = mock_client
        mock_client.generate.return_value = {"response": "Local answer."}
        result = eng.generate_general_response("Some question?")
        assert result == "Local answer."


# ─── tokenize_text ─────────────────────────────────────────────────────────────

class TestTokenizeText:
    def test_empty_string_returns_empty_list(self, engine):
        eng, _ = engine
        assert eng.tokenize_text("") == []

    def test_returns_list_of_integers(self, engine):
        eng, _ = engine
        result = eng.tokenize_text("hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_capped_at_twelve_tokens(self, engine):
        eng, _ = engine
        result = eng.tokenize_text(" ".join(["word"] * 20))
        assert len(result) <= 12


# ─── generate_response ────────────────────────────────────────────────────────

class TestGenerateResponse:
    def test_local_ollama_returns_response(self, engine):
        eng, mock_client = engine
        mock_client.generate.return_value = {"response": "Your keys are on the desk."}

        result = eng.generate_response("Where are my keys?", "{'keys': 'on desk'}")

        assert result == "Your keys are on the desk."
        mock_client.generate.assert_called_once()

    def test_falls_back_to_cloud_when_local_fails(self, engine):
        eng, mock_client = engine
        mock_client.generate.side_effect = Exception("Ollama down")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Cloud answer."
        eng.fallback_enabled = True
        eng.fallback_llm = mock_llm

        result = eng.generate_response("Where are my keys?", "{}")

        assert result == "Cloud answer."
        mock_llm.invoke.assert_called_once()

    def test_returns_error_string_when_both_paths_fail(self, engine):
        eng, mock_client = engine
        mock_client.generate.side_effect = Exception("Ollama down")
        eng.fallback_enabled = False

        result = eng.generate_response("Where are my keys?", "{}")

        assert isinstance(result, str)
        assert len(result) > 0


# ─── generate_general_response ───────────────────────────────────────────────────

class TestGenerateGeneralResponse:
    def test_uses_langchain_when_openai_key_is_set(self, engine):
        """Cloud path: LangChain invoke is called and its response is returned."""
        eng, _ = engine
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Paris is the capital of France."
        eng.fallback_enabled = True
        eng.fallback_llm = mock_llm

        result = eng.generate_general_response("What is the capital of France?")

        assert result == "Paris is the capital of France."
        mock_llm.invoke.assert_called_once()

    def test_uses_local_ollama_when_no_api_key(self, engine):
        """Local path: Ollama generate() is called when fallback_enabled=False."""
        eng, mock_client = engine
        eng.fallback_enabled = False
        mock_client.generate.return_value = {"response": "Water boils at 100°C."}

        result = eng.generate_general_response("At what temperature does water boil?")

        assert result == "Water boils at 100°C."
        mock_client.generate.assert_called_once()

    def test_falls_back_to_ollama_when_cloud_raises(self, engine):
        """Resilience: cloud failure triggers the local Ollama fallback."""
        eng, mock_client = engine
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("OpenAI timeout")
        eng.fallback_enabled = True
        eng.fallback_llm = mock_llm
        mock_client.generate.return_value = {"response": "Local fallback answer."}

        result = eng.generate_general_response("Some question?")

        assert result == "Local fallback answer."
        mock_client.generate.assert_called_once()

    def test_graceful_error_message_when_ollama_also_offline(self, engine):
        """Both paths down: returns a non-empty error string instead of raising."""
        eng, mock_client = engine
        eng.fallback_enabled = False
        mock_client.generate.side_effect = Exception("Connection refused")

        result = eng.generate_general_response("Some question?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_response_is_stripped_of_surrounding_whitespace(self, engine):
        """Ollama sometimes returns padded output — ensure it's trimmed."""
        eng, mock_client = engine
        eng.fallback_enabled = False
        mock_client.generate.return_value = {"response": "  Trimmed answer.  "}

        result = eng.generate_general_response("Question?")

        assert result == "Trimmed answer."


# ─── QueryClassifier ──────────────────────────────────────────────────────────

class TestQueryClassifier:
    def test_empty_input_returns_general_intent(self, classifier):
        clf, _ = classifier
        result = clf.classify("")
        assert result["intent"] == "general"
        assert result["payload"] == ""

    def test_locate_intent_parsed_correctly(self, classifier):
        clf, mock_client = classifier
        mock_client.generate.return_value = {
            "response": '{"intent": "locate", "extracted_target": "keys"}'
        }
        result = clf.classify("Where are my keys?")
        assert result["intent"] == "locate"
        assert result["payload"] == "keys"

    def test_note_intent_parsed_correctly(self, classifier):
        clf, mock_client = classifier
        mock_client.generate.return_value = {
            "response": '{"intent": "note", "extracted_target": "buy milk"}'
        }
        result = clf.classify("Remember to buy milk")
        assert result["intent"] == "note"
        assert result["payload"] == "buy milk"

    def test_alarm_intent_parsed_correctly(self, classifier):
        clf, mock_client = classifier
        mock_client.generate.return_value = {
            "response": '{"intent": "alarm", "extracted_target": "8am meeting"}'
        }
        result = clf.classify("Set an alarm for 8am")
        assert result["intent"] == "alarm"

    def test_general_intent_parsed_correctly(self, classifier):
        clf, mock_client = classifier
        mock_client.generate.return_value = {
            "response": '{"intent": "general", "extracted_target": "weather"}'
        }
        result = clf.classify("What is the weather like?")
        assert result["intent"] == "general"

    def test_falls_back_on_llm_exception(self, classifier):
        clf, mock_client = classifier
        mock_client.generate.side_effect = Exception("Ollama offline")
        result = clf.classify("Where is my wallet?")
        assert result["intent"] == "general"
        assert result["payload"] == "Where is my wallet?"

    def test_falls_back_on_malformed_json(self, classifier):
        clf, mock_client = classifier
        mock_client.generate.return_value = {"response": "not valid json {{{"}
        result = clf.classify("Some query")
        assert result["intent"] == "general"
