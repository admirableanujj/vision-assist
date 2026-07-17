"""
VisionCore Orchestrator

Consolidates system startup by running database migrations/seeding 
and launching the Streamlit interface programmatically.
"""
import sys
import subprocess
from precheck_db import run_migrations_and_seeding

def main():
    print("[SYSTEM] Running database precheck and migrations...")
    try:
        run_migrations_and_seeding()
    except Exception as e:
        print(f"[FATAL ERROR] Database precheck/migration failed: {e}")
        sys.exit(1)
    
    print("[SYSTEM] Starting Streamlit interface...")
    
    # Using sys.executable ensures we use the exact same Python binary and path 
    # where all pip packages (like streamlit) are installed inside the container.
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    
    try:
        # We use subprocess.Popen to let Streamlit take over the process space cleanly,
        # or we keep subprocess.run but make sure it handles signals and doesn't swallow stdout.
        process = subprocess.Popen(streamlit_cmd)
        
        # Wait for the process to exit or be killed (Ctrl+C)
        process.wait()
        
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down gracefully.")
        if 'process' in locals():
            process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Streamlit failed to launch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()