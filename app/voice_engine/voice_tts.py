#app/voice_engine/voice_tts.py
"""
Browser-Side Text-To-Speech (TTS) Engine

Converts model response text strings into streaming audio assets using gTTS.
This bypasses headless Linux container audio card errors by handing off audio playback 
directly to the client's local browser window.

Usage:
    Imported dynamically into the core framework via:
    from voice_engine.voice_tts import TextToSpeechConverter

Dependencies:
    gTTS==2.5.1
    
__original_author__ = "Anujj Saxena"
__license__ = "MIT"       
"""
__author__ = "Anujj Saxena"
__license__ = "MIT"
__version__ = "1.0.1"
import os
import sys
from .voice_base import VoiceEngineAC
from gtts import gTTS

class TextToSpeechConverter(VoiceEngineAC):
    """
    Converts string responses into auditory files suitable for web UI streaming,
    bypassing headless container sound card limitations.
    """
    def __init__(self, speech_rate: int = 165):
        self.speech_rate = speech_rate
        self.output_filename = "response_vocal.mp3"
        self.initialize_engine()

    def initialize_engine(self) -> None:
        print("[INFO] TextToSpeech pipeline configured for browser-side streaming delivery.")

    def execute(self, response_text: str) -> str:
        """
        Generates an audio file from text.
        Returns the string path to the audio file if successful, or an empty string.
        """
        if not response_text:
            return ""

        try:
            # Clean up old audio files before generating a new one
            if os.path.exists(self.output_filename):
                os.remove(self.output_filename)

            # Use edge-tts or gtts via pip to create file buffers without a sound card
            tts = gTTS(text=response_text, lang='en', tld='com')
            tts.save(self.output_filename)
            
            return self.output_filename
        except Exception as e:
            print(f"[ERROR] Audio file generation failure: {e}")
            return ""