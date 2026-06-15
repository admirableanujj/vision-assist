import os
import sys

def print_header(title):
    print("\n" + "=" * 50)
    print(f"🔍 TESTING: {title}")
    print("=" * 50)

def test_python_and_env():
    print_header("System & Environment Variables")
    print(f"✔ Python Version: {sys.version}")
    
    # Check if .env variables mapped into Docker correctly
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        # Mask key for security
        masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "Present"
        print(f"✔ OPENAI_API_KEY: Found ({masked})")
    else:
        print("❌ OPENAI_API_KEY: Missing! Ensure your .env file exists and docker-compose mapped it.")

def test_pytorch():
    print_header("Deep Learning Backend (PyTorch)")
    try:
        import torch
        print(f"✔ PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"💡 CUDA (GPU Acceleration) Available: {cuda_available}")
        if cuda_available:
            print(f"✔ Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ Running on CPU mode (Expected for standard/Mac/Windows WSL2 baseline setups)")
    except ImportError:
        print("❌ PyTorch is not installed properly!")

def test_computer_vision():
    print_header("Computer Vision Pipeline (OpenCV & YOLO)")
    try:
        import cv2
        print(f"✔ OpenCV Version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV (opencv-python-headless) is missing!")
        return

    try:
        from ultralytics import YOLO
        import numpy as np
        
        print("🔄 Loading pre-cached YOLOv8 model weights...")
        model = YOLO("yolov8n.pt")
        print("✔ Model loaded successfully!")
        
        # Create a fake blank image array to test the mathematical forward-pass inference matrix
        print("🔄 Running sanity-check inference on dummy matrix...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = model.predict(dummy_frame, verbose=False)
        print("✔ Vision forward pass successful! Object detection architecture operational.")
        
    except ImportError:
        print("❌ Ultralytics (YOLO) library is missing!")
    except Exception as e:
        print(f"❌ Error during vision processing test: {e}")

def test_vector_db():
    print_header("Vector Database Backend (ChromaDB)")
    try:
        import chromadb
        print(f"✔ ChromaDB Version: {chromadb.__version__}")
        client = chromadb.EphemeralClient()
        print("✔ Client initialization successful! Memory storage working.")
    except ImportError:
        print("❌ ChromaDB is missing!")
    except Exception as e:
        print(f"❌ Error initializing Vector DB: {e}")

if __name__ == "__main__":
    print("==================================================")
    print("🚀 VISION ASSIST ENVIRONMENT VERIFICATION SCRIPT 🚀")
    print("==================================================")
    
    test_python_and_env()
    test_pytorch()
    test_computer_vision()
    test_vector_db()
    
    print("\n==================================================")
    print("🏁 Diagnostics Complete!")
    print("==================================================\n")