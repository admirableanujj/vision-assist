import sys
import torch
import cv2

def run_sanity_check():
    print("=" * 50)
    print("🌍 VOICE-VISION ASSISTANT: ENVIRONMENT SANITY CHECK 🌍")
    print("=" * 50)

    # 1. Verify Python & PyTorch
    print(f"🐍 Python Version: {sys.version.split()[0]}")
    print(f"🔥 PyTorch Version: {torch.__version__}")
    
    # Check for GPU acceleration (CUDA for Nvidia, MPS for Apple Silicon)
    cuda_available = torch.cuda.is_available()
    print(f"🚀 GPU Acceleration Available (CUDA): {cuda_available}")
    if cuda_available:
        print(f"   └─ Device Name: {torch.cuda.get_device_name(0)}")

    # 2. Verify OpenCV
    print(f"📸 OpenCV Version: {cv2.__version__}")

    # 3. Simulate Camera Input (Creating a dummy image matrix)
    # This ensures OpenCV can manipulate pixel matrices inside Docker
    print("\n⏳ Testing CV Matrix manipulation...")
    try:
        # Create a black square image (400x400 pixels, 3 color channels)
        dummy_frame = cv2.rectangle(
            cv2.copyMakeBorder(cv2.UMat(), 0, 400, 0, 400, cv2.BORDER_CONSTANT, value=[0, 0, 0]), 
            (50, 50), (350, 350), (0, 255, 0), -1
        )
        print("✅ Success! OpenCV matrix operations are working flawlessly.")
    except Exception as e:
        print(f"❌ OpenCV Test Failed: {e}")
        
    print("=" * 50)

if __name__ == "__main__":
    run_sanity_check()