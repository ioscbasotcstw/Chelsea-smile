# ArcticMonkey:19.03.24:1700 example of version name in plaintext  will be converted into hex using this site ->
# https://magictool.ai/tool/text-to-hex-converter/ Here ArcticMonkey is name of version and rest of all is data and time

import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else "cpu"

checkpoint_whisper = "openai/whisper-medium"

pipe = pipeline(
    "automatic-speech-recognition",
    model=checkpoint_whisper,
    device=device,
    chunk_length_s=30, 
)