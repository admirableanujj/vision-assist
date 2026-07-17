# app/app.py
"""
VisionCore Engine Frontend

Streamlit dashboard for VisionAssist. Orchestrates local speech-to-text, 
computer vision, and local LLM orchestration logic with database-backed 
user session authentication and persistent inventory tracking.

Usage:
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0

__original_author__ = "Anujj Saxena"
__license__ = "MIT"
"""
__author__ = "Anujj Saxena"
__license__ = "MIT"
__version__ = "1.0.1"

import streamlit as st
import os
import psycopg2
from psycopg2.extras import RealDictCursor

# Clean, package-level module imports
from ml_engine import OllamaMLEngine
from voice_engine.voice_stt import SpeechToTextConverter
from voice_engine.voice_tts import TextToSpeechConverter
from vision_engine import VisionTracker
from ml_engine.query_classifier import QueryClassifier
from user_module.user_manager import UserManager  # <-- Imported your polished module

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
        QueryClassifier(brain.local_client),
        UserManager()  # <-- Cached your User Manager instance
    )

# Pass our configuration flag down into the boot initialization
ml_brain, speaker, whisper_stt, tracker, router, user_mgr = boot_system_core(VISION_ENABLED)


# --- DATABASE UTILITY HELPERS ---
def fetch_user_items(user_id: int) -> list:
    """Retrieves all registered tracked items for the active user from Postgres."""
    conn = user_mgr._get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT item_name, description, last_modified_on 
                FROM items 
                WHERE owner_id = %s 
                ORDER BY last_modified_on DESC;
            """, (user_id,))
            return cur.fetchall()
    except Exception as e:
        st.error(f"Error fetching items from database: {e}")
        return []
    finally:
        conn.close()

def register_db_item(user_id: int, item_name: str, description: str) -> bool:
    """Persists a new registered item into the Postgres database."""
    conn = user_mgr._get_connection()
    try:
        with conn.cursor() as cur:
            # Generate a clean system ID for the item asset
            item_id = f"item_{user_id}_{int(os.getpid())}_{item_name.replace(' ', '_')}"
            cur.execute("""
                INSERT INTO items (id, owner_id, item_name, description)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET description = EXCLUDED.description, last_modified_on = CURRENT_TIMESTAMP;
            """, (item_id, user_id, item_name, description))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Failed to register item in database: {e}")
        return False
    finally:
        conn.close()


# --- SESSION AUTHENTICATION GUARD ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None

if not st.session_state.authenticated:
    st.title("🔑 VisionAssist Security Gateway")
    st.subheader("Login to access your personalized tracking workspace")
    
    tab_login, tab_register = st.tabs(["Sign In", "Create Account"])
    
    with tab_login:
        login_user = st.text_input("Username", key="login_user_field")
        login_pass = st.text_input("Password", type="password", key="login_pass_field")
        if st.button("Authenticate Session", use_container_width=True):
            if user_mgr.authenticate_user(login_user, login_pass):
                # Retrieve the newly validated user's primary database ID
                conn = user_mgr._get_connection()
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT id FROM users WHERE username = %s;", (login_user,))
                    u_rec = cur.fetchone()
                conn.close()
                
                st.session_state.authenticated = True
                st.session_state.username = login_user
                st.session_state.user_id = u_rec["id"]
                st.success(f"Access granted! Welcoming session token for {login_user}.")
                st.rerun()
            else:
                st.error("Invalid username or password credentials. Please retry.")
                
    with tab_register:
        reg_user = st.text_input("Choose Username", key="reg_user_field")
        reg_pass = st.text_input("Choose Password", type="password", key="reg_pass_field")
        if st.button("Register Account Credentials", use_container_width=True):
            success, msg = user_mgr.register_user(reg_user, reg_pass)
            if success:
                st.success(f"{msg} You can now log in using the 'Sign In' tab.")
            else:
                st.error(msg)
                
    st.stop()  # Stop rendering dashboard content until verified


# --- AUTHENTICATED SYSTEM DASHBOARD ---
st.title("🔍 FoundItGini — Lost & Found AI")
st.caption(f"Authenticated Session: {st.session_state.username} | Powered by Whisper, Computer Vision & Postgres.")

# Logout control inside sidebar top boundary
with st.sidebar:
    st.markdown(f"**👤 Connected as:** `{st.session_state.username}`")
    if st.button("Logout of Workspace", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()
    st.markdown("---")

# --- SIDEBAR: DATABASE ITEM REGISTRATION ---
with st.sidebar:
    st.header("📦 Inventory & Registration")
    st.write("Register personal items to keep track of their status:")
    new_item = st.text_input("Item Name (e.g., Wallet)", placeholder="Enter item name...").lower().strip()
    new_loc = st.text_input("Expected Location", placeholder="e.g., Bedroom side table")
    
    if st.button("Register Belonging", use_container_width=True):
        if new_item and new_loc:
            description = f"located in the {new_loc} (Manually registered)"
            if register_db_item(st.session_state.user_id, new_item, description):
                st.success(f"Registered '{new_item}' successfully in Postgres!")
                st.rerun()
        else:
            st.error("Please fill out both fields.")

    st.sidebar.subheader("🔌 Connection Status")
    if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY").startswith("sk-..."):
        st.sidebar.success("🌐 Cloud API: Enabled")
    else:
        st.sidebar.info("🏡 Local Engine Mode: Enabled (Offline Safe)")

# --- MAIN INTERFACE LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎙️ Ask Gini via Voice")
    audio_file = st.audio_input("Click the microphone icon below to record your search command:")

    if audio_file is not None:
        st.audio(audio_file)
        
        with st.spinner("Whisper is processing natural language patterns..."):
            transcribed_text = whisper_stt.execute(audio_file)
            
            if transcribed_text and not transcribed_text.startswith("Error"):
                st.success(f"🗣️ **Whisper Transcribed:** \"{transcribed_text}\"")
                
                with st.spinner("Analyzing intent signatures..."):
                    classification = router.classify(transcribed_text)             
                    intent = classification["intent"]
                    payload = classification["payload"]
                    
                st.caption(f"🎯 **System Intent Routing Detected:** `{intent.upper()}`")

                # Dynamic Operational Execution Routes
                if intent == "locate":
                    with st.spinner("Searching tracking database matrix..."):
                        # Convert database records into a string context format for the LLM
                        user_db_items = fetch_user_items(st.session_state.user_id)
                        db_items_context = {item["item_name"]: item["description"] for item in user_db_items}
                        
                        ml_response = ml_brain.generate_response(transcribed_text, str(db_items_context))
                        st.info(f"🤖 **VisionCore-ML [Locate Mode]:** {ml_response}")

                elif intent == "note":
                    st.success(f"📝 **Note-Taking Module Triggered:** Logging payload: \"{payload}\"")
                    ml_response = f"I've noted that down for you: {payload}"

                elif intent == "alarm":
                    st.warning(f"⏰ **Alarm/Scheduling Triggered:** Setting event parameters for: \"{payload}\"")
                    ml_response = f"Handling your scheduling request for {payload} now."

                else: # intent == "general"
                    with st.spinner("Engaging LangChain cloud backup pipelines for general inquiry..."):
                        ml_response = ml_brain.generate_general_response(transcribed_text)
                        st.info(f"🌐 **Cloud Hybrid Assistant [General Mode]:** {ml_response}")

                # Deliver Audio back to user browser
                audio_output_path = speaker.execute(ml_response)
                if audio_output_path and os.path.exists(audio_output_path):
                    st.audio(audio_output_path, format="audio/mp3", autoplay=True)
            else:
                st.error(f"❌ Transcription Failure: {transcribed_text}")

with col2:
    st.subheader("👁️ Live Camera Workspace")
    
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
                        desc = "Detected in live workspace sweep (Just now)"
                        register_db_item(st.session_state.user_id, item.lower(), desc)
                    st.rerun()
                else:
                    st.caption("No registered tracking assets found in the current scene context.")
    else:
        st.warning("⚠️ Vision Engine Module has been set to disabled in application settings. Object scanning via web camera is inactive.")

# --- INVENTORY FOOTPRINT LOG ---
st.markdown("---")
st.subheader("📋 System Status Log")

# Pull items directly from our Postgres relational system matching the current session's user ID
raw_items = fetch_user_items(st.session_state.user_id)
df_data = [
    {
        "Belonging": item["item_name"].capitalize(), 
        "Last Seen Tracking Status": item["description"],
        "Last System Log": item["last_modified_on"].strftime("%Y-%m-%d %H:%M:%S")
    } 
    for item in raw_items
]

if df_data:
    st.table(df_data)
else:
    st.info("No belongings currently registered in your database profile. Use the sidebar to register your first item!")