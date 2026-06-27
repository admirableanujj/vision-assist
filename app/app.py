# app/app.py
"""
VisionCore Engine Frontend

This script serves as the primary Streamlit web application dashboard for 
VisionAssist, orchestrating local speech-to-text, computer vision, 
and local LLM orchestration logic in a single interface.

Usage:
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0

Dependencies:
    streamlit
    pandas
    numpy
    
__original_author__ = "Anujj Saxena"
__license__ = "MIT"
"""
__author__ = "Anujj Saxena"
__license__ = "MIT"
__version__ = "1.0.1"
import streamlit as st
import os

# Clean, package-level module imports
from ml_engine import OllamaMLEngine
from voice_engine.voice_stt import SpeechToTextConverter
from voice_engine.voice_tts import TextToSpeechConverter
from vision_engine import VisionTracker
from ml_engine.query_classifier import QueryClassifier

# SYSTEM CONFIGURATION FLAGS
VISION_ENABLED = False  # Set to True to turn the camera tracking system back on

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="VisionAssist L-F-A-I", page_icon="🔍", layout="wide")

# Initialize our core engine abstractions inside Streamlit's resource cache
@st.cache_resource
def boot_system_core(enable_vision: bool):
    brain = OllamaMLEngine()
    return (
        brain, 
        TextToSpeechConverter(speech_rate=165),
        SpeechToTextConverter(),  # Whisper Engine initialization
        VisionTracker() if enable_vision else None,
        QueryClassifier(brain.local_client)  # <-- CACHE THE ROUTER INSTANCE
    )

# Pass our configuration flag down into the boot initialization
ml_brain, speaker, whisper_stt, tracker, router = boot_system_core(VISION_ENABLED)



st.title("🔍 FoundItGini — Lost & Found AI")
st.caption("Never lose sight of what matters. Powered by Whisper, Computer Vision & Local LLM.")
st.markdown("---")

# Maintain our operational tracking database matrix via local session memory
if "tracked_items" not in st.session_state:
    st.session_state.tracked_items = {
        "keys": "on the workspace desk near the monitor (detected 10 mins ago)",
        "phone": "on the kitchen counter next to the coffee maker",
        "sunglasses": "on the dining room table"
    }

# --- SIDEBAR: ITEM REGISTRATION ---
with st.sidebar:
    st.header("📦 Inventory & Registration")
    st.write("Register personal items to keep track of their status:")
    new_item = st.text_input("Item Name (e.g., Wallet)", placeholder="Enter item name...").lower().strip()
    new_loc = st.text_input("Expected Location", placeholder="e.g., Bedroom side table")
    
    
    if st.button("Register Belonging", use_container_width=True):
        if new_item and new_loc:
            st.session_state.tracked_items[new_item] = f"in the {new_loc} (Manually registered)"
            st.success(f"Registered '{new_item}' successfully!")
        else:
            st.error("Please fill out both fields.")

# --- MAIN INTERFACE LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎙️ Ask Gini via Voice")
    audio_file = st.audio_input("Click the microphone icon below to record your search command:")

    if audio_file is not None:
        st.audio(audio_file)
        
        with st.spinner("Whisper is processing natural language patterns..."):
            # Process raw buffer through your custom modular Whisper client wrapper
            transcribed_text = whisper_stt.execute(audio_file)
            
            if transcribed_text and not transcribed_text.startswith("Error"):
                st.success(f"🗣️ **Whisper Transcribed:** \"{transcribed_text}\"")
                
                with st.spinner("Analyzing intent signatures..."):
                    # 1. Instantly tap into your cached router layer
                    classification = router.classify(transcribed_text)             
                    intent = classification["intent"]
                    payload = classification["payload"]
                    
                st.caption(f"🎯 **System Intent Routing Detected:** `{intent.upper()}`")

                # 2. Dynamic Operational Execution Routes
                if intent == "locate":
                    with st.spinner("Searching tracking database matrix..."):
                        current_context = str(st.session_state.tracked_items)
                        ml_response = ml_brain.generate_response(transcribed_text, current_context)
                        st.info(f"🤖 **VisionCore-ML [Locate Mode]:** {ml_response}")

                elif intent == "note":
                    # Placeholder route for note-taking feature expansions
                    st.success(f"📝 **Note-Taking Module Triggered:** Logging payload: \"{payload}\"")
                    ml_response = f"I've noted that down for you: {payload}"

                elif intent == "alarm":
                    # Placeholder route for clock/scheduling activations
                    st.warning(f"⏰ **Alarm/Scheduling Triggered:** Setting event parameters for: \"{payload}\"")
                    ml_response = f"Handling your scheduling request for {payload} now."

                else: # intent == "general"
                    with st.spinner("Engaging LangChain cloud backup pipelines for general inquiry..."):
                        # If it's a general question or weather request, let the cloud fallback model answer it
                        ml_response = ml_brain.generate_cloud_response(transcribed_text)
                        st.info(f"🌐 **Cloud Hybrid Assistant [General Mode]:** {ml_response}")

                # 3. Deliver Audio back to user browser
                audio_output_path = speaker.execute(ml_response)
                if audio_output_path and os.path.exists(audio_output_path):
                    st.audio(audio_output_path, format="audio/mp3", autoplay=True)
            else:
                # Intercept API/Key omissions gracefully directly inside UI space
                st.error(f"❌ Transcription Failure: {transcribed_text}")

with col2:
    st.subheader("👁️ Live Camera Workspace")
    
    # Check our feature configuration flag before building UI nodes
    if VISION_ENABLED and tracker is not None:
        st.write("Optical environment frame scanner ready.")
        cam_frame = st.camera_input("Environmental Scanner Feed")
        if cam_frame:
            st.image(cam_frame, caption="Processing live visual frames...")
            
            with st.spinner("Scanning frame targets..."):
                detected_items = tracker.scan_frame(cam_frame)
                
                if detected_items:
                    st.success(f"🎯 **Detected on Feed:** {', '.join([item.capitalize() for item in detected_items])}")
                    for item in detected_items:
                        st.session_state.tracked_items[item.lower()] = "Detected in live workspace sweep (Just now)"
                else:
                    st.caption("No registered tracking assets found in the current scene context.")
    else:
        # 🛡️ Clean interface fallback presentation when disabled
        st.warning("⚠️ Vision Engine Module has been set to disabled in application settings. Object scanning via web camera is inactive.")

# --- INVENTORY FOOTPRINT LOG ---
st.markdown("---")
st.subheader("📋 System Status Log")
df_data = [{"Belonging": item.capitalize(), "Last Seen Tracking Status": status} for item, status in st.session_state.tracked_items.items()]
st.table(df_data)