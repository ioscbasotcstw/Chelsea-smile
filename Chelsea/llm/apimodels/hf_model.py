import os

from abc import ABC
from typing import Any

from llm.utils.hf_interface import HFInterface
from llm.utils.config import config

from langchain_community.llms import HuggingFaceEndpoint

_api = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

class HF_Mistaril(HFInterface, ABC):
    """
    This class represents an interface for the Mistaril large language model from Hugging Face.

    It inherits from `HFInterface` (likely an interface from a Hugging Face library)
    and `ABC` (for abstract base class) to enforce specific functionalities.
    """

    def __init__(self):
        """
        Initializer for the `HF_Mistaril` class.

        - Retrieves configuration values for the Mistaril model from a `config` dictionary:
            - `repo_id`: The ID of the repository containing the Mistaril model on Hugging Face.
            - `max_length`: Maximum length of the generated text.
            - `temperature`: Controls randomness in the generation process.
            - `top_k`: Restricts the vocabulary used for generation.
        - Raises a `ValueError` if the `api` key (presumably stored elsewhere) is missing.
        - Creates an instance of `HuggingFaceEndpoint` using the retrieved configuration
          and the `api` key.
        """

        repo_id = config["HF_Mistrail"]["model"]
        max_length = config["HF_Mistrail"]["max_new_tokens"]
        temperature = config["HF_Mistrail"]["temperature"]
        top_k = config["HF_Mistrail"]["top_k"]

        if not _api:
            raise ValueError(f"API key not provided {_api}")

        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, max_length=max_length, temperature=temperature, top_k=top_k, token=_api
        )

    def execution(self) -> Any:
        """
        This method attempts to return the underlying `llm` (likely a language model object).

        It wraps the retrieval in a `try-except` block to catch potential exceptions.
        On success, it returns the `llm` object.
        On failure, it logs an error message with the exception details using a logger
        (assumed to be available elsewhere).
        """
        try:
            return self.llm  # `invoke()`
        except Exception as e:
            print(f"Something wrong with API or HuggingFaceEndpoint: {e}")

    def model_name(self):
        """
        Simple method that returns the Mistaril model name from the configuration.

        This can be useful for identifying the specific model being used.
        """
        return config["HF_Mistrail"]["model"]

    def __str__(self):
        """
        Defines the string representation of the `HF_Mistaril` object for human readability.

        It combines the class name and the model name retrieved from the `model_name` method
        with an underscore separator.
        """
        return f"{self.__class__.__name__}_{self.model_name()}"

    def __repr__(self):
        """
        Defines the representation of the `HF_Mistaril` object for debugging purposes.

        It uses `hasattr` to check if the `llm` attribute is set.
        - If `llm` exists, it returns a string like `HF_Mistaril(llm=HuggingFaceEndpoint(...))`,
          showing the class name and the `llm` object information.
        - If `llm` is not yet set (during initialization), it returns
          `HF_Mistaril(llm=not initialized)`, indicating the state.
        """
        llm_info = f"llm={self.llm}" if hasattr(self, 'llm') else 'llm=not initialized'
        return f"{self.__class__.__name__}({llm_info})"

class HF_TinyLlama(HFInterface, ABC):
    """
    This class represents an interface for the TinyLlama large language model from Hugging Face.

    It inherits from `HFInterface` (likely an interface from a Hugging Face library)
    and `ABC` (for abstract base class) to enforce specific functionalities.
    """
        
    def __init__(self):
        """
        Initializer for the `HF_TinyLlama` class.

        - Retrieves configuration values for the TinyLlama model from a `config` dictionary:
            - `repo_id`: The ID of the repository containing the TinyLlama model on Hugging Face.
            - `max_length`: Maximum length of the generated text.
            - `temperature`: Controls randomness in the generation process.
            - `top_k`: Restricts the vocabulary used for generation.
        - Raises a `ValueError` if the `api` key (presumably stored elsewhere) is missing.
        - Creates an instance of `HuggingFaceEndpoint` using the retrieved configuration
          and the `api` key.
        """

        repo_id = config["HF_TinyLlama"]["model"]
        max_length = config["HF_TinyLlama"]["max_new_tokens"]
        temperature = config["HF_TinyLlama"]["temperature"]
        top_k = config["HF_TinyLlama"]["top_k"]

        if not _api:
            raise ValueError(f"API key not provided {_api}")

        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, max_length=max_length, temperature=temperature, top_k=top_k, token=_api
        )

    def execution(self) -> Any:
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
            print(f"Something wrong with API or HuggingFaceEndpoint: {e}")
        
    def model_name(self):
        """
        Simple method that returns the TinyLlama model name from the configuration.
        This can be useful for identifying the specific model being used.
        """
        return config["HF_TinyLlama"]["model"]
        
    def __str__(self):
        """
        Defines the string representation of the `HF_TinyLlama` object for human readability.
        It combines the class name and the model name retrieved from the `model_name` method
        with an underscore separator.
        """
        return f"{self.__class__.__name__}_{self.model_name()}"

    def __repr__(self):
        """
        Defines the representation of the `HF_TinyLlama` object for debugging purposes.
        It uses `hasattr` to check if the `llm` attribute is set.
        - If `llm` exists, it returns a string like `HF_TinyLlama(llm=HuggingFaceEndpoint(...))`,
          showing the class name and the `llm` object information.
        - If `llm` is not yet set (during initialization), it returns
          `HF_TinyLlama(llm=not initialized)`, indicating the state.
        """
        llm_info = f"llm={self.llm}" if hasattr(self, 'llm') else 'llm=not initialized'
        return f"{self.__class__.__name__}({llm_info})"

class HF_SmolLM135(HFInterface, ABC):
    """
    This class represents an interface for the SmolLm tiny language model from Hugging Face.
    It inherits from `HFInterface` (likely an interface from a Hugging Face library)
    and `ABC` (for abstract base class) to enforce specific functionalities.
    """

    def __init__(self):
        """
        Initializer for the `HF_SmolLM135` class.
        - Retrieves configuration values for the SmolLM135 model from a `config` dictionary:
            - `repo_id`: The ID of the repository containing the SmolLM135 model on Hugging Face.
            - `max_length`: Maximum length of the generated text.
            - `temperature`: Controls randomness in the generation process.
            - `top_k`: Restricts the vocabulary used for generation.
        - Raises a `ValueError` if the `api` key (presumably stored elsewhere) is missing.
        - Creates an instance of `HuggingFaceEndpoint` using the retrieved configuration
          and the `api` key.
        """

        repo_id = config["HF_SmolLM135"]["model"]
        max_length = config["HF_SmolLM135"]["max_new_tokens"]
        temperature = config["HF_SmolLM135"]["temperature"]
        top_k = config["HF_SmolLM135"]["top_k"]

        if not _api:
            raise ValueError(f"API key not provided {_api}")

        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, max_length=max_length, temperature=temperature, top_k=top_k, token=_api
        )

    def execution(self) -> Any:
        """
        This method attempts to return the underlying `llm` (likely a language model object).
        It wraps the retrieval in a `try-except` block to catch potential exceptions.
        On success, it returns the `llm` object.
        On failure, it logs an error message with the exception details using a logger
        (assumed to be available elsewhere).
        """
        try:
            return self.llm  # `invoke()`
        except Exception as e:
            print(f"Something wrong with API or HuggingFaceEndpoint: {e}")

    def model_name(self):
        """
        Simple method that returns the SmolLM135 model name from the configuration.
        This can be useful for identifying the specific model being used.
        """
        return config["HF_SmolLM135"]["model"]

    def __str__(self):
        """
        Defines the string representation of the `HF_SmolLM135` object for human readability.
        It combines the class name and the model name retrieved from the `model_name` method
        with an underscore separator.
        """
        return f"{self.__class__.__name__}_{self.model_name()}"

    def __repr__(self):
        """
        Defines the representation of the `HF_SmolLM135` object for debugging purposes.
        It uses `hasattr` to check if the `llm` attribute is set.
        - If `llm` exists, it returns a string like `HF_SmolLM135(llm=HuggingFaceEndpoint(...))`,
          showing the class name and the `llm` object information.
        - If `llm` is not yet set (during initialization), it returns
          `HF_SmolLM135(llm=not initialized)`, indicating the state.
        """
        llm_info = f"llm={self.llm}" if hasattr(self, 'llm') else 'llm=not initialized'
        return f"{self.__class__.__name__}({llm_info})"

class HF_SmolLM360(HFInterface, ABC):
    """
    This class represents an interface for the SmolLm tiny language model from Hugging Face.
    It inherits from `HFInterface` (likely an interface from a Hugging Face library)
    and `ABC` (for abstract base class) to enforce specific functionalities.
    """

    def __init__(self):
        """
        Initializer for the `HF_SmolLM360` class.
        - Retrieves configuration values for the SmolLM360 model from a `config` dictionary:
            - `repo_id`: The ID of the repository containing the SmolLM360 model on Hugging Face.
            - `max_length`: Maximum length of the generated text.
            - `temperature`: Controls randomness in the generation process.
            - `top_k`: Restricts the vocabulary used for generation.
        - Raises a `ValueError` if the `api` key (presumably stored elsewhere) is missing.
        - Creates an instance of `HuggingFaceEndpoint` using the retrieved configuration
          and the `api` key.
        """

        repo_id = config["HF_SmolLM360"]["model"]
        max_length = config["HF_SmolLM360"]["max_new_tokens"]
        temperature = config["HF_SmolLM360"]["temperature"]
        top_k = config["HF_SmolLM360"]["top_k"]

        if not _api:
            raise ValueError(f"API key not provided {_api}")

        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, max_length=max_length, temperature=temperature, top_k=top_k, token=_api
        )

    def execution(self) -> Any:
        """
        This method attempts to return the underlying `llm` (likely a language model object).
        It wraps the retrieval in a `try-except` block to catch potential exceptions.
        On success, it returns the `llm` object.
        On failure, it logs an error message with the exception details using a logger
        (assumed to be available elsewhere).
        """
        try:
            return self.llm  # `invoke()`
        except Exception as e:
            print(f"Something wrong with API or HuggingFaceEndpoint: {e}")

    def model_name(self):
        """
        Simple method that returns the SmolLM360 model name from the configuration.
        This can be useful for identifying the specific model being used.
        """
        return config["HF_SmolLM360"]["model"]

    def __str__(self):
        """
        Defines the string representation of the `HF_SmolLM360` object for human readability.
        It combines the class name and the model name retrieved from the `model_name` method
        with an underscore separator.
        """
        return f"{self.__class__.__name__}_{self.model_name()}"

    def __repr__(self):
        """
        Defines the representation of the `HF_SmolLM360` object for debugging purposes.
        It uses `hasattr` to check if the `llm` attribute is set.
        - If `llm` exists, it returns a string like `HF_SmolLM360(llm=HuggingFaceEndpoint(...))`,
          showing the class name and the `llm` object information.
        - If `llm` is not yet set (during initialization), it returns
          `HF_SmolLM360(llm=not initialized)`, indicating the state.
        """
        llm_info = f"llm={self.llm}" if hasattr(self, 'llm') else 'llm=not initialized'
        return f"{self.__class__.__name__}({llm_info})"

class HF_SmolLM(HFInterface, ABC):
    """
    This class represents an interface for the SmolLm small language model from Hugging Face.
    It inherits from `HFInterface` (likely an interface from a Hugging Face library)
    and `ABC` (for abstract base class) to enforce specific functionalities.
    """

    def __init__(self):
        """
        Initializer for the `HF_SmolLM` class.
        - Retrieves configuration values for the SmolLM model from a `config` dictionary:
            - `repo_id`: The ID of the repository containing the SmolLM model on Hugging Face.
            - `max_length`: Maximum length of the generated text.
            - `temperature`: Controls randomness in the generation process.
            - `top_k`: Restricts the vocabulary used for generation.
        - Raises a `ValueError` if the `api` key (presumably stored elsewhere) is missing.
        - Creates an instance of `HuggingFaceEndpoint` using the retrieved configuration
          and the `api` key.
        """

        repo_id = config["HF_SmolLM"]["model"]
        max_length = config["HF_SmolLM"]["max_new_tokens"]
        temperature = config["HF_SmolLM"]["temperature"]
        top_k = config["HF_SmolLM"]["top_k"]

        if not _api:
            raise ValueError(f"API key not provided {_api}")

        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, max_length=max_length, temperature=temperature, top_k=top_k, token=_api
        )

    def execution(self) -> Any:
        """
        This method attempts to return the underlying `llm` (likely a language model object).
        It wraps the retrieval in a `try-except` block to catch potential exceptions.
        On success, it returns the `llm` object.
        On failure, it logs an error message with the exception details using a logger
        (assumed to be available elsewhere).
        """
        try:
            return self.llm  # `invoke()`
        except Exception as e:
            print(f"Something wrong with API or HuggingFaceEndpoint: {e}")

    def model_name(self):
        """
        Simple method that returns the SmolLM model name from the configuration.
        This can be useful for identifying the specific model being used.
        """
        return config["HF_SmolLM"]["model"]

    def __str__(self):
        """
        Defines the string representation of the `HF_SmolLM` object for human readability.
        It combines the class name and the model name retrieved from the `model_name` method
        with an underscore separator.
        """
        return f"{self.__class__.__name__}_{self.model_name()}"

    def __repr__(self):
        """
        Defines the representation of the `HF_SmolLM` object for debugging purposes.
        It uses `hasattr` to check if the `llm` attribute is set.
        - If `llm` exists, it returns a string like `HF_SmolLM(llm=HuggingFaceEndpoint(...))`,
          showing the class name and the `llm` object information.
        - If `llm` is not yet set (during initialization), it returns
          `HF_SmolLM(llm=not initialized)`, indicating the state.
        """
        llm_info = f"llm={self.llm}" if hasattr(self, 'llm') else 'llm=not initialized'
        return f"{self.__class__.__name__}({llm_info})"

class HF_Gemma2(HFInterface, ABC):
    """
    This class represents an interface for the Gemma2 small language model from Hugging Face.
    It inherits from `HFInterface` (likely an interface from a Hugging Face library)
    and `ABC` (for abstract base class) to enforce specific functionalities.
    """

    def __init__(self):
        """
        Initializer for the `HF_Gemma2` class.
        - Retrieves configuration values for the Gemma2 model from a `config` dictionary:
            - `repo_id`: The ID of the repository containing the Gemma2 model on Hugging Face.
            - `max_length`: Maximum length of the generated text.
            - `temperature`: Controls randomness in the generation process.
            - `top_k`: Restricts the vocabulary used for generation.
        - Raises a `ValueError` if the `api` key (presumably stored elsewhere) is missing.
        - Creates an instance of `HuggingFaceEndpoint` using the retrieved configuration
          and the `api` key.
        """

        repo_id = config["HF_Gemma2"]["model"]
        max_length = config["HF_Gemma2"]["max_new_tokens"]
        temperature = config["HF_Gemma2"]["temperature"]
        top_k = config["HF_Gemma2"]["top_k"]

        if not _api:
            raise ValueError(f"API key not provided {_api}")

        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, max_length=max_length, temperature=temperature, top_k=top_k, token=_api
        )

    def execution(self) -> Any:
        """
        This method attempts to return the underlying `llm` (likely a language model object).
        It wraps the retrieval in a `try-except` block to catch potential exceptions.
        On success, it returns the `llm` object.
        On failure, it logs an error message with the exception details using a logger
        (assumed to be available elsewhere).
        """
        try:
            return self.llm  # `invoke()`
        except Exception as e:
            print(f"Something wrong with API or HuggingFaceEndpoint: {e}")

    def model_name(self):
        """
        Simple method that returns the Gemma2 model name from the configuration.
        This can be useful for identifying the specific model being used.
        """
        return config["HF_Gemma2"]["model"]

    def __str__(self):
        """
        Defines the string representation of the `HF_Gemma2` object for human readability.
        It combines the class name and the model name retrieved from the `model_name` method
        with an underscore separator.
        """
        return f"{self.__class__.__name__}_{self.model_name()}"

    def __repr__(self):
        """
        Defines the representation of the `HF_Gemma2` object for debugging purposes.
        It uses `hasattr` to check if the `llm` attribute is set.
        - If `llm` exists, it returns a string like `HF_Gemma2(llm=HuggingFaceEndpoint(...))`,
          showing the class name and the `llm` object information.
        - If `llm` is not yet set (during initialization), it returns
          `HF_Gemma2(llm=not initialized)`, indicating the state.
        """
        llm_info = f"llm={self.llm}" if hasattr(self, 'llm') else 'llm=not initialized'
        return f"{self.__class__.__name__}({llm_info})"

class HF_Qwen2(HFInterface, ABC):
    """
    This class represents an interface for the Qwen2 small language model from Hugging Face.
    It inherits from `HFInterface` (likely an interface from a Hugging Face library)
    and `ABC` (for abstract base class) to enforce specific functionalities.
    """

    def __init__(self):
        """
        Initializer for the `HF_Qwen2` class.
        - Retrieves configuration values for the Qwen2 model from a `config` dictionary:
            - `repo_id`: The ID of the repository containing the Qwen2 model on Hugging Face.
            - `max_length`: Maximum length of the generated text.
            - `temperature`: Controls randomness in the generation process.
            - `top_k`: Restricts the vocabulary used for generation.
        - Raises a `ValueError` if the `api` key (presumably stored elsewhere) is missing.
        - Creates an instance of `HuggingFaceEndpoint` using the retrieved configuration
          and the `api` key.
        """

        repo_id = config["HF_Qwen2"]["model"]
        max_length = config["HF_Qwen2"]["max_new_tokens"]
        temperature = config["HF_Qwen2"]["temperature"]
        top_k = config["HF_Qwen2"]["top_k"]

        if not _api:
            raise ValueError(f"API key not provided {_api}")

        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, max_length=max_length, temperature=temperature, top_k=top_k, token=_api
        )

    def execution(self) -> Any:
        """
        This method attempts to return the underlying `llm` (likely a language model object).
        It wraps the retrieval in a `try-except` block to catch potential exceptions.
        On success, it returns the `llm` object.
        On failure, it logs an error message with the exception details using a logger
        (assumed to be available elsewhere).
        """
        try:
            return self.llm  # `invoke()`
        except Exception as e:
            print(f"Something wrong with API or HuggingFaceEndpoint: {e}")

    def model_name(self):
        """
        Simple method that returns the Qwen2 model name from the configuration.
        This can be useful for identifying the specific model being used.
        """
        return config["HF_Qwen2"]["model"]

    def __str__(self):
        """
        Defines the string representation of the `HF_Qwen2` object for human readability.
        It combines the class name and the model name retrieved from the `model_name` method
        with an underscore separator.
        """
        return f"{self.__class__.__name__}_{self.model_name()}"

    def __repr__(self):
        """
        Defines the representation of the `HF_Qwen2` object for debugging purposes.
        It uses `hasattr` to check if the `llm` attribute is set.
        - If `llm` exists, it returns a string like `HF_Qwen2(llm=HuggingFaceEndpoint(...))`,
          showing the class name and the `llm` object information.
        - If `llm` is not yet set (during initialization), it returns
          `HF_Qwen2(llm=not initialized)`, indicating the state.
        """
        llm_info = f"llm={self.llm}" if hasattr(self, 'llm') else 'llm=not initialized'
        return f"{self.__class__.__name__}({llm_info})"