import time

from llm.apimodels.gemini_model import Gemini
from llm.apimodels.hf_model import HF_Mistaril, HF_TinyLlama, HF_SmolLM135, HF_SmolLM360, HF_SmolLM, HF_Gemma2, HF_Qwen2

from typing import Optional, Any

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

def prettify(raw_text: str) -> str:        
    pretty = raw_text.replace("**", "")
    return pretty.strip()


memory: ConversationBufferWindowMemory = ConversationBufferWindowMemory(k=3, ai_prefix="Chelsea")

DELAY: int = 300  # 5 minutes

def has_failed(conversation, prompt) -> Optional[str]:
    """
    Checks if the LLM conversation prediction fails and returns None if so.

    Args:
        conversation: The LLM conversation object used for prediction.
        prompt: The prompt to be used for prediction.

    Returns:
        None, otherwise the prettified response.
    """

    try:
        response = conversation.predict(input=prompt)
        print(f"response: {response}")
        result = prettify(raw_text=response)
        return result
    except Exception as e:
        print(f"Error during prediction with conversation in has_failed function: {e}")
        return None


def has_delay(conversation, prompt) -> Optional[str]:
    """
    Checks if the LLM conversation prediction takes longer than a set delay.

    Args:
        conversation: The LLM conversation object used for prediction.
        prompt: The prompt to be used for prediction.

    Returns:
        None if the execution time exceeds the delay,
        otherwise, the prettified response from the conversation object.
    """

    start_time = time.perf_counter()  # Start timer before prediction
    try:
        response = conversation.predict(input=prompt)
        execution_time = time.perf_counter() - start_time  # Calculate execution time

        if execution_time > DELAY: 
            return None  # Return None if delayed
        
        result = prettify(raw_text=response)  # Prettify the response
        return result  # Return the prettified response

    except Exception as e:
        print(f"Error during prediction with conversation in has_delay function: {e}")


class Conversation:
    def __init__(self):
        """
        Initializes the Conversation class with a prompt and a list of LLM model classes.

        Args:
            model_classes (list, optional): A list of LLM model classes to try in sequence.
                Defaults to [Gemini, HF_SmolLM135, HF_SmolLM360, HF_TinyLlama, HF_SmolLM, HF_Gemma2, HF_Mistaril, HF_Qwen2].
        """

        self.model_classes = [Gemini, HF_Gemma2, HF_SmolLM, HF_SmolLM360, HF_Mistaril, HF_Qwen2, HF_TinyLlama, HF_SmolLM135]
        self.current_model_index = 0

    def _get_conversation(self) -> Any:
        """
        Creates a ConversationChain object using the current model class.
        """
        try:
            current_model_class = self.model_classes[self.current_model_index]
            print("current model class is: ", current_model_class)
            return ConversationChain(llm=current_model_class().execution(), memory=memory, return_final_only=True)
        except Exception as e:
            print(f"Error during conversation chain in get_conversation function: {e}")

    def chatting(self, prompt: str) -> str:
        """
        Carries out the conversation with the user, handling errors and delays.

        Args: 
            prompt(str): The prompt to be used for prediction.

        Returns:
            str: The final conversation response or None if all models fail.
        """

        if prompt is None or prompt == "":
            raise Exception(f"Prompt must be string not None or empty string: {prompt}")

        while self.current_model_index < len(self.model_classes):
            conversation = self._get_conversation()

            result = has_failed(conversation=conversation, prompt=prompt)
            if result is not None:
                return result
            print(f"chat - chatting result : {result}")

            result = has_delay(conversation=conversation, prompt=prompt)
            if result is None:
                self.current_model_index += 1  # Switch to next model after delay
                continue

            return result

        return "All models failed conversation. Please, try again"
    
    def __str__(self) -> str:
        return f"prompt: {type(self.prompt)}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(prompt: {type(self.prompt)})"
    