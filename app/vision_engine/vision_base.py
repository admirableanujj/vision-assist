# app/vision_engine/vision_base.py
from abc import ABC, abstractmethod

class BaseVisionEngine(ABC):
    """
    Abstract Base Class for environmental scanner architectures.
    """
    @abstractmethod
    def scan_frame(self, image_buffer) -> list:
        """
        Process a visual frame to extract detected items.
        
        Returns:
            list: A list of string labels discovered in the frame.
        """
        pass