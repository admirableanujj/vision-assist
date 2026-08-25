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
import hashlib
import html
import os
import gc
import psycopg2
from psycopg2.extras import RealDictCursor

# Clean, package-level module imports
from ml_engine import OllamaMLEngine
from voice_engine.voice_stt import SpeechToTextConverter
from voice_engine.voice_tts import TextToSpeechConverter
from vision_engine import FallbackVisionEngine
from ml_engine.query_classifier import QueryClassifier
from user_module.user_manager import UserManager  # <-- Imported your polished module

# SYSTEM CONFIGURATION FLAGS
VISION_ENABLED = True  # Set to False to disable the camera tracking system

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="VisionAssist L-F-A-I", page_icon="🔍", layout="wide")

# Initialize our core engine abstractions inside Streamlit's resource cache
@st.cache_resource
def boot_system_core():
    brain = OllamaMLEngine()
    return (
        brain,
        TextToSpeechConverter(speech_rate=165),
        SpeechToTextConverter(),  # Whisper Engine initialization
        QueryClassifier(brain.local_client),
        UserManager()  # <-- Cached your User Manager instance
    )

ml_brain, speaker, whisper_stt, router, user_mgr = boot_system_core()


def get_vision_engine(engine_choice: str):
    """
    Builds exactly one vision engine at a time (session_state-cached),
    evicting the previous one on switch — switching engines costs a real
    reload, not instant, in exchange for only ever holding one model's
    native memory resident. Falls back to the mock engine if the real
    vision stack (cv2/ultralytics) isn't importable at all, same degrade
    path YOLOVisionEngine/YOLOEVisionEngine use internally for load failures.
    """
    if not VISION_ENABLED:
        return None

    if (
        st.session_state.get("_vision_engine_choice") == engine_choice
        and "_vision_engine" in st.session_state
    ):
        return st.session_state["_vision_engine"]

    try:
        if engine_choice == "YOLOE":
            from vision_engine import YOLOEVisionEngine as EngineCls
        else:
            from vision_engine import YOLOVisionEngine as EngineCls
    except ImportError:
        EngineCls = FallbackVisionEngine

    # Drop the previous engine before building the new one so only one
    # model's native memory is ever resident at a time.
    st.session_state.pop("_vision_engine", None)
    gc.collect()

    engine = EngineCls()
    st.session_state["_vision_engine"] = engine
    st.session_state["_vision_engine_choice"] = engine_choice
    return engine


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

# No eager pre-warm here — get_vision_engine() now loads exactly one engine
# lazily, on first selection, to keep only one model resident at a time.

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

    if VISION_ENABLED:
        engine_choice = st.radio(
            "Detection engine",
            options=["YOLO", "YOLOE"],
            index=0,
            horizontal=True,
            help=(
                "YOLO: fixed 80 COCO classes, most accurate. "
                "YOLOE: open-vocabulary, much broader class list (see YOLO_VS_YOLOE_GUIDE.md), "
                "lower accuracy per class. Switching reloads the model (a few seconds) — "
                "only one engine is kept resident at a time."
            ),
        )
        already_loaded = st.session_state.get("_vision_engine_choice") == engine_choice
        if already_loaded:
            tracker = get_vision_engine(engine_choice)
        else:
            with st.spinner(f"Loading {engine_choice}..."):
                tracker = get_vision_engine(engine_choice)
    else:
        tracker = None

    if VISION_ENABLED and tracker is not None:
        st.write("Optical environment frame scanner ready.")
        st.caption(
            f"Engine: {type(tracker).__name__} · "
            f"Confidence threshold: {getattr(tracker, 'confidence_threshold', 'N/A')}"
        )
        cam_frame = st.camera_input("Environmental Scanner Feed")
        if cam_frame:
            # st.camera_input keeps returning the same captured photo across
            # every script rerun until it's explicitly retaken — without this
            # guard, a successful detection below would scan_frame() the exact
            # same image again on each rerun, forever. Track which photo (by
            # content hash) was actually scanned so each capture is processed
            # exactly once, regardless of how many times the script reruns.
            # Engine choice is part of the key too — switching YOLO<->YOLOE on
            # an already-scanned photo must trigger a fresh scan with the newly
            # selected engine, not silently keep showing the old engine's result.
            frame_hash = hashlib.md5(cam_frame.getvalue()).hexdigest() + f"|{engine_choice}"
            if st.session_state.get("last_scanned_frame_hash") != frame_hash:
                st.session_state["last_scanned_frame_hash"] = frame_hash
                with st.spinner("Scanning frame targets..."):
                    scan_result = tracker.scan_frame(cam_frame)
                st.session_state["last_scan_result"] = scan_result
                for d in scan_result["detections"]:
                    desc = "Detected in live workspace sweep (Just now)"
                    register_db_item(st.session_state.user_id, d["label"].lower(), desc)

            # Render from the cached result for this photo — the script reruns
            # naturally on every interaction anyway (e.g. the inventory table
            # below always reflects the latest registration), so no st.rerun()
            # is needed here, and results stay visible instead of flashing away.
            cached_result = st.session_state.get("last_scan_result")
            if cached_result:
                if cached_result["annotated_frame"] is not None:
                    st.image(cached_result["annotated_frame"], caption="Detected objects highlighted")
                else:
                    st.image(cam_frame, caption="Processing live visual frames...")

                if cached_result["detections"]:
                    summary = ", ".join(
                        f"{d['label'].capitalize()} ({d['confidence']:.0%})"
                        for d in cached_result["detections"]
                    )
                    st.success(f"🎯 **Detected on Feed:** {summary}")
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
    # Rendered as plain HTML rather than st.table()/st.dataframe() — both
    # route through pyarrow's Arrow serialization, which segfaulted here.
    rows_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['Belonging']))}</td>"
        f"<td>{html.escape(str(row['Last Seen Tracking Status']))}</td>"
        f"<td>{html.escape(str(row['Last System Log']))}</td>"
        "</tr>"
        for row in df_data
    )
    st.markdown(
        f"""
        <table style="width:100%; border-collapse: collapse;">
          <thead>
            <tr>
              <th style="text-align:left; border-bottom:1px solid #666; padding:6px;">Belonging</th>
              <th style="text-align:left; border-bottom:1px solid #666; padding:6px;">Last Seen Tracking Status</th>
              <th style="text-align:left; border-bottom:1px solid #666; padding:6px;">Last System Log</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("No belongings currently registered in your database profile. Use the sidebar to register your first item!")