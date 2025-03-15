from abc import ABC, abstractmethod
from typing import Any, Optional

class VLMIterface(ABC):
    @abstractmethod
    def execution(self) -> Optional[Any]:
        """Method execution VLM model based on Gemini"""
        pass
    @abstractmethod
    def model_name(self) -> str:
        """Method for checking model"""
        pass