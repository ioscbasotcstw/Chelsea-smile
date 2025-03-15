import os
from typing import Any
from vlm.vlm_interface import VLMIterface
import google.generativeai as genai
# Fetch Google API key from environment variables
API = os.getenv("GOOGLE_API_KEY")
NAME = "gemini-1.5-flash"

class VLM(VLMIterface):
    """
    This class implements the interface for a Visual Language Model (VLM) using Google's generative AI API.
    It inherits from the VLMInterface and configures the Google API to interact with the generative model.
    """

    def __init__(self) -> None:
        """
        Initialize the VLM object by configuring the Google generative AI model using the provided API key.
        """
        super().__init__()
        # Configure the generative AI model with the provided API key
        genai.configure(api_key=API)
        # Initialize the model (Gemini 1.5 Flash) from Google's generative AI
        self.vlm = genai.GenerativeModel(NAME)

    def execution(self) -> Any | None:
        """
        Execute the VLM model and return the model object.
        If there is an error (e.g., invalid API key or connection issue), handle the exception gracefully.
        """
        try:
            return self.vlm
        except Exception as e:
            print(f"Something went wrong with the API: {e}")
            return None
    
    def model_name(self) -> str:
        return NAME

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the VLM object.
        Useful for informal descriptions or when printing the object.
        """
        return f"VLM Model: {self.vlm.model_name}"

    def __repr__(self) -> str:
        """
        Return an official string representation of the VLM object.
        This should ideally be a valid Python expression that can recreate the object.
        """
        return f"VLM(api_key='****', model_name='{self.vlm.model_name}')"