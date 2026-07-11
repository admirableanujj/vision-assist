import sys
import os
from unittest.mock import MagicMock

# Add app/ to the path so ml_engine.* imports resolve when pytest runs from app/tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Stub out packages that are only available inside Docker.
# This lets the test suite run locally without the full dependency stack.
_DOCKER_ONLY_DEPS = [
    "ollama",
    "dotenv",
    "langchain_openai",
    "langchain_core",
    "langchain_core.messages",
]
for _mod in _DOCKER_ONLY_DEPS:
    sys.modules.setdefault(_mod, MagicMock())
