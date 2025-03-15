import os

from llm.utils.hf_interface import HFInterface

from langchain_google_genai import GoogleGenerativeAI
from abc import ABC

# 429 - You've exceeded the rate limit.
# 400 - The request body is malformed.
# 403 - Your API key doesn't have the required permissions.
# 404 - The requested resource wasn't found.
# 500 - An unexpected error occurred on Google's side.
# 503 - The service may be temporarily overloaded or down.

# Якщо трапиться одна з цих помилок , то слід перемкнутися на HF , якщо якась модель занадто повільна 
# перемкнутися на іншу після закінчення виконання текущої, якщо  трабли з HF перемкнутися на локальну, 
# якщо ж у користувача відсутнє інтернет зʼєднання, то нічим не зарадиш, 
# хіба що пропонувати скачати репозиторій.

_api = os.environ.get("GOOGLE_API_KEY")


class Gemini(HFInterface, ABC):
    """
    This class represents a Gemini large language model interface.

    It inherits from `HFInterface` (likely an interface from a Hugging Face library)
    and `ABC` (for abstract base class) to enforce specific functionalities.
    """

    def __init__(self):
        """
        Initializer for the Gemini class.

        - Raises a `ValueError` if the provided API key is None or an empty string.
        - Creates an instance of `GoogleGenerativeAI` using the specified model name
          ("gemini-1.5-flash") and the stored API key.
        """

        if not _api:
            raise ValueError(f"Your api is None or empty string {_api}, please provide a Gemini API")

        #{
        #   'model': 'gemini-1.5-flash', 'temperature': 0.7, 'top_p': None, 
        #   'top_k': None, 'max_output_tokens': None, 'candidate_count': 1
        #}
        self.llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=_api)

    def execution(self) -> GoogleGenerativeAI:
        """
        This method attempts to return the underlying `llm` (likely a language model object).

        It wraps the retrieval in a `try-except` block to catch potential exceptions.
        On success, it returns the `llm` object.
        On failure, it logs an error message with the exception details using a logger
        (assumed to be available elsewhere).
        """
        try:
            return self.llm
        except Exception as e:
            print(f"Something wrong with Gemini api: {e}")

    def model_name(self):
        """
        Simple method that returns the hardcoded model name ("gemini-1.5-flash").

        This can be useful for identifying the specific model being used.
        """
        return "gemini-1.5-flash"

    def __str__(self):
        """
        Defines the string representation of the Gemini object for human readability.

        It returns a string indicating that it's a "Gemini model" and appends the model name
        obtained from the `model_name` method.
        """
        return f"Gemini model: {self.model_name()}"

    def __repr__(self):
        """
        Defines the representation of the Gemini object for debugging purposes.

        It uses `hasattr` to check if the `llm` attribute is set.
        - If `llm` exists, it returns a string like `Gemini(llm=GoogleGenerativeAI(...))`,
          showing the class name and the `llm` object information.
        - If `llm` is not yet set (during initialization), it returns
          `Gemini(llm=not initialized)`, indicating the state.
        """
        llm_info = f"llm={self.llm}" if hasattr(self, 'llm') else 'llm=not initialized'
        return f"{self.__class__.__name__}({llm_info})"


