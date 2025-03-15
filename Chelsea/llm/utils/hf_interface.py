from abc import ABC, abstractmethod
from typing import Any, Optional

class HFInterface(ABC):
    @abstractmethod
    def execution(self) -> Optional[Any]:
        """Method execution LLM model based on HuggingFace or others"""
        pass