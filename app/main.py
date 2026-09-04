# app/main.py
"""
VisionCore Orchestrator

Consolidates system startup by running database migrations/seeding 
and launching the Streamlit interface programmatically.

Usage:
    python app/main.py
"""
import os
import sys
import time
import subprocess
from precheck_db import run_migrations_and_seeding
from logging_config import get_logger

# Orchestrator runs non-interactively inside containers — avoid console spam
logger = get_logger(__name__, console=False)

def load_docker_secrets():
    """Reads Docker secret files and injects them safely into os.environ at runtime."""
    secrets_map = {
        "/run/secrets/postgres_password": "POSTGRES_PASSWORD",
        "/run/secrets/qdrant_api_key": "QDRANT_API_KEY"  # Map path to target env variable
    }
    
    for secret_path, env_name in secrets_map.items():
        if os.path.exists(secret_path):
            try:
                with open(secret_path, "r", encoding="utf-8") as f:
                    # Strip hidden Windows/Linux newlines and trailing whitespace
                    os.environ[env_name] = f.read().strip()
                logger.info(f"[INFO] Successfully loaded secret into {env_name} from {secret_path}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to read secret at {secret_path}: {e}")
        else:
            logger.info(f"[INFO] No secret file at {secret_path}. Using fallback environment string.")


def main():
    load_docker_secrets()
    logger.info("[SYSTEM] Running database precheck and migrations...")
    try:
        run_migrations_and_seeding()
    except Exception as e:
        logger.exception(f"Database precheck/migration failed: {e}")
        sys.exit(1)
    
    # Ensure current working directory is anchored to the app directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    logger.info("[SYSTEM] Starting Streamlit interface...")
    
    # Using sys.executable ensures we use the exact same Python binary and path 
    # where all pip packages (like streamlit) are installed inside the container.
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    
    process = None
    try:
        # PYTHONFAULTHANDLER=1 makes CPython dump the active Python-level stack
        # to stderr when a fatal native signal (SIGSEGV, SIGABRT, SIGFPE, SIGBUS)
        # kills the process — without it, a segfault just kills Streamlit with no
        # trace at all.
        env = {**os.environ, "PYTHONFAULTHANDLER": "1"}

        # We use subprocess.Popen to let Streamlit take over the process space cleanly,
        # or we keep subprocess.run but make sure it handles signals and doesn't swallow stdout.
        process = subprocess.Popen(streamlit_cmd, env=env)

        # Wait for the process to exit or be killed (Ctrl+C). Propagate its real
        # return code (negative when killed by a signal, e.g. -9 for SIGKILL) —
        # without this, `main()` falls through and the container always reports
        # a clean exit 0 to Docker no matter how Streamlit actually died, which
        # makes `docker inspect`'s ExitCode useless for diagnosing crashes.
        returncode = process.wait()
        if returncode != 0:
            logger.fatal(f"Streamlit exited unexpectedly with code {returncode}.")
        sys.exit(returncode)

    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt. Terminating child processes...")
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Streamlit failed to launch or crashed unexpectedly: {e}")
        if process and process.poll() is None:
            process.kill()
        sys.exit(1)

if __name__ == "__main__":
    main()