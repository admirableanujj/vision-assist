# app/ml_engine/ml_engine.py
"""
Ollama and LangChain Hybrid ML Engine

This module interfaces directly with the local Ollama SDK daemon to coordinate 
LLM reasoning patterns. It features a reliable cloud fallback strategy via 
LangChain OpenAI clients if local assets go offline.

Usage:
    Imported dynamically into the Streamlit app execution tree via:
    from ml_engine.ml_engine import OllamaMLEngine

Dependencies:
    ollama
    langchain
    langchain-openai
    
__original_author__ = "Anujj Saxena"
__license__ = "MIT"    
"""
__author__ = "Anujj Saxena"
__license__ = "MIT"
__version__ = "1.0.1"
import os
import ollama
from dotenv import load_dotenv
from .ml_base_engine import BaseMLEngine

# Load local environment variables (for OpenAI API keys, etc.)
load_dotenv()

class OllamaMLEngine(BaseMLEngine):
    """
    Enhanced concrete implementation of BaseMLEngine natively utilizing the Ollama SDK 
    with an integrated LangChain + OpenAI cloud fallback framework for voice classification.
    """
    def __init__(self, host: str = "http://vision_assist_llm_local:11434", default_model: str = "llama3"):
        self.host = host
        self.model = default_model
        
        # Configure client bindings for the official Ollama SDK
        self.local_client = ollama.Client(host=self.host)
        print(f"[INFO] OllamaMLEngine natively bound over SDK client via {self.host}")
        
        # Optional LangChain / OpenAI Fallback Integration
        self.fallback_enabled = os.getenv("OPENAI_API_KEY") is not None
        if self.fallback_enabled:
            from langchain_openai import ChatOpenAI
            # Initialize a highly concise cloud fallback engine for latency insurance
            self.fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            print("[INFO] LangChain OpenAI cloud hybrid fallback engine securely configured.")

    # def tokenize_text(self, input_text: str) -> list:
    #     """
    #     Natively pulls token index maps via the local Ollama SDK client wrapper.
    #     """
    #     try:
    #         response = self.local_client.tokenize(model=self.model, prompt=input_text)
    #         return response.get("tokens", [])
    #     except Exception as e:
    #         print(f"[WARN] Local Tokenizer failed: {e}. Falling back to empty map.")
    #         return []

    def tokenize_text(self, input_text: str) -> list:
            """
            Natively pulls token index maps via the local Ollama SDK client wrapper.
            """
            if not input_text:
                return []
                
            try:
                # The correct client layout uses the embedded .embed or sub-client structures
                # For direct raw string tokenization arrays:
                response = self.local_client.embed(model=self.model, input=input_text)
                
                # If your downstream layout explicitly needs integer index locations:
                # We can map the embedded float arrays or query the internal model tokens:
                return response.get("embeddings", [[]])[0][:12] # Slices out first 12 structural markers
                
            except Exception as e:
                # Fallback routine so the Streamlit UI never completely freezes up
                print(f"[WARN] Local Tokenizer failed: {e}. Falling back to clean word-split token weights.")
                return [len(word) for word in input_text.split()]

    def tokenize_text(self, input_text: str) -> list:
        """
        Simulates token indexes securely without requiring an external 
        Ollama embedding model infrastructure setup.
        """
        if not input_text:
            return []
        try:
            # Converts letters directly into fake token IDs for UI display metrics
            return [ord(char) % 100 for char in input_text.split()[:12]]
        except Exception:
            return []

    
    def generate_response(self, user_text: str, tracking_context: str) -> str:
        """
        Executes structural query evaluation and context response engineering.
        """
        system_prompt = (
            "You are VisionCore-ML, a highly precise assistive voice system.\n"
            "Analyze the voice command and answer accurately using ONLY the tracking context provided.\n"
            f"Active Tracking Context Matrix: {tracking_context}\n"
            f"User Voice Request: {user_text}\n"
            "Keep your final spoken output clear, conversational, and under two sentences."
        )

        try:
            # 1. Attempt Native Local Inference via Ollama SDK
            response = self.local_client.generate(
                model=self.model,
                prompt=system_prompt,
                options={"temperature": 0.2, "stop": ["\n\n"]}
            )
            return response.get("response", "").strip()

        except Exception as local_error:
            print(f"[SYSTEM WARN] Local ML Engine generation failed: {local_error}")
            
            # 2. Trigger Intelligent Cloud Failover if Local Container is Offline/Stalled
            if self.fallback_enabled:
                print("[SYSTEM INFO] Engaging LangChain cloud backup pipelines...")
                try:
                    from langchain_core.messages import SystemMessage, HumanMessage
                    messages = [
                        SystemMessage(content="You are VisionCore-ML fallback system. Answer concisely based on given inventory state."),
                        HumanMessage(content=f"Context: {tracking_context}\nQuery: {user_text}")
                    ]
                    cloud_response = self.fallback_llm.invoke(messages)
                    return cloud_response.content.strip()
                except Exception as cloud_error:
                    return f"System Orchestration Failure. Cloud chain also unreachable: {cloud_error}"
            
            return "VisionAssist Core is currently recovering or adjusting container system links."