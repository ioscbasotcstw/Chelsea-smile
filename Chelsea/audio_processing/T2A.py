from typing import Optional
from streamlit_TTS import auto_play, text_to_audio


class T2A:
    def autoplay(self, input_text: Optional[str] = None, lang: str = "en") -> None:
        """
        Plays audio based on the provided input text.
    
        Args:
            input_text (Optional[str], optional): Text to convert to audio. Defaults to None.
            lang (str, optional): Language for text-to-speech conversion. Defaults to "en".
        """
        
        if input_text is None:
            text = "Please check the input text you have provided, it has a value of None"
            audio = text_to_audio(text, language=lang)
            auto_play(audio)
        
        if not isinstance(input_text, str):
            text = f"The text you provided is of data type {type(input_text)}, only string type is accepted"
            audio = text_to_audio(text, language=lang)
            auto_play(audio)

        audio = text_to_audio(input_text, language=lang)
        auto_play(audio)
