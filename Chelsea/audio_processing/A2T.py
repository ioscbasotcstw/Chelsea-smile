import io
import librosa
import numpy as np

from typing import Optional

from .config import pipe

TASK = "transcribe"
BATCH_SIZE = 8

class A2T:
    def __init__(self, mic):
        self.mic = mic

    def __generate_text(self, inputs, task: Optional[str] = None) -> str:
        if inputs is None:
            raise ValueError(f"Input audio is None {inputs}, please provide audio")

        transcribed_text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
        return transcribed_text

    @staticmethod
    def __preprocess(raw: bytes) -> np.ndarray:
        print(f"Raw type: {type(raw)}")

        if not isinstance(raw, bytes):
            raise ValueError("Expected raw audio data as bytes")
            
        try:
            chunk = io.BytesIO(raw)
            print(f"Chunk type: {type(chunk)}")
            audio, sample_rate = librosa.load(chunk, sr=16000)
            print(f"Sample rate : {sample_rate}")
            return audio
        except Exception as e:
            print(f"Error loading audio in the preprocess function in the A2T class: {e}")
 
    def predict(self) -> str:
        try:
            if self.mic is not None:
                raw = self.mic
                audio = self.__preprocess(raw=raw)
                print(f"audio type : {type(audio)} \n shape : {audio.shape} \n audio max value : {np.max(audio)}")
            else:
                raise ValueError(f"Please provide audio your audio {self.mic}")

            if isinstance(audio, np.ndarray):
                return self.__generate_text(inputs=audio, task=TASK)
            else:
                raise ValueError("Audio is not np array")
                
        except Exception as e:
            print(f"An error occurred in the predict function in the A2T class: {e}")