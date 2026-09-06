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
        
        print("🔄 Loading pre-cached YOLO26n model weights...")
        model = YOLO("yolo26n.pt")
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

def test_qdrant_db():
    print_header("Vector Database Backend (Qdrant)")
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            print(f"✔ qdrant-client Version: {version('qdrant-client')}")
        except PackageNotFoundError:
            print("✔ qdrant-client is installed (version metadata unavailable).")
    except ImportError:
        pass

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        print("❌ qdrant-client is missing! (pip install qdrant-client)")
        return

    # --- Core client sanity check: full upsert → search round trip in local
    # in-memory mode, so this passes with no live server required. Vector size
    # matches the clip-ViT-B-32 embedding dimension.
    EMBED_DIM = 512
    try:
        client = QdrantClient(location=":memory:")
        client.create_collection(
            collection_name="verify_probe",
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        client.upsert(
            collection_name="verify_probe",
            points=[
                PointStruct(id=1, vector=[0.1] * EMBED_DIM, payload={"label": "keys"}),
                PointStruct(id=2, vector=[0.2] * EMBED_DIM, payload={"label": "wallet"}),
            ],
        )
        # query_points is the modern API (qdrant-client >= 1.10); fall back to the
        # deprecated .search() so this stays correct on older client pins.
        try:
            hits = client.query_points(
                collection_name="verify_probe", query=[0.1] * EMBED_DIM, limit=1
            ).points
        except AttributeError:
            hits = client.search(
                collection_name="verify_probe", query_vector=[0.1] * EMBED_DIM, limit=1
            )
        assert hits, "search returned no hits"
        print(
            "✔ In-memory client OK — collection + upsert + cosine search round trip "
            f"working (top hit id={hits[0].id}, score={hits[0].score:.3f})."
        )
    except Exception as e:
        print(f"❌ Error exercising in-memory Qdrant client: {e}")
        return

    # --- Optional live-server probe: only when a running Qdrant is reachable
    # (e.g. inside docker-compose, service host "qdrant"). Never fails the check
    # when standalone — the in-memory round trip above is the real gate.
    host = os.getenv("QDRANT_HOST", "qdrant")
    port = os.getenv("QDRANT_PORT", "6333")
    url = os.getenv("QDRANT_URL", f"http://{host}:{port}")
    api_key = os.getenv("QDRANT_API_KEY") or os.getenv("QDRANT__SERVICE__API_KEY")
    try:
        live = QdrantClient(url=url, api_key=api_key, timeout=2.0)
        collections = live.get_collections().collections
        print(f"✔ Live Qdrant server reachable at {url} ({len(collections)} collection(s)).")
    except Exception:
        print(f"ℹ No live Qdrant server at {url} (expected outside docker-compose).")

def test_embeddings():
    print_header("Multimodal Embeddings (CLIP / sentence-transformers)")
    try:
        import sentence_transformers
        print(f"✔ sentence-transformers Version: {sentence_transformers.__version__}")
    except ImportError:
        print("❌ sentence-transformers is missing!")
        return

    try:
        import os
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("EMBEDDING_MODEL_NAME", "clip-ViT-B-32")
        print(f"🔄 Loading pre-cached embedding model '{model_name}'...")
        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()
        print(f"✔ Model loaded successfully! (embedding dim: {dim})")

        print("🔄 Encoding a sanity text sample...")
        vec = model.encode("a black leather wallet", normalize_embeddings=True)
        print(f"✔ Text embedding produced (length {len(vec)}). CLIP encoder operational.")
    except Exception as e:
        print(f"❌ Error during embedding model test: {e}")

if __name__ == "__main__":
    print("==================================================")
    print("🚀 VISION ASSIST ENVIRONMENT VERIFICATION SCRIPT 🚀")
    print("==================================================")
    
    test_python_and_env()
    test_pytorch()
    test_computer_vision()
    test_vector_db()
    test_qdrant_db()
    test_embeddings()

    print("\n==================================================")
    print("🏁 Diagnostics Complete!")
    print("==================================================\n")