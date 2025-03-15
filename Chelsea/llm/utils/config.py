config = {
    "HF_Mistrail": {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "temperature": 0.5,
        "max_new_tokens": 1024,
        "top_k": 5,
        "load_in_8bit": True
    },
    "HF_TinyLlama": {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "temperature": 0.5,
        "max_new_tokens": 1024,
        "top_k": 5,
        "top_p":0.95,
        "load_in_8bit": True,
        "do_sample": True
    },
    "HF_SmolLM135": {
        "model": "HuggingFaceTB/SmolLM-135M-Instruct",
        "temperature": 0.5,
        "max_new_tokens": 1024,
        "top_k": 5,
        "top_p":0.95,
        "load_in_8bit": True,
        "do_sample": True
    },
    "HF_SmolLM360": {
        "model": "HuggingFaceTB/SmolLM-360M-Instruct",
        "temperature": 0.5,
        "max_new_tokens": 1024,
        "top_k": 5,
        "top_p":0.95,
        "load_in_8bit": True,
        "do_sample": True
    },
    "HF_SmolLM": {
        "model": "HuggingFaceTB/SmolLM-1.7B-Instruct",
        "temperature": 0.5,
        "max_new_tokens": 1024,
        "top_k": 5,
        "top_p":0.95,
        "load_in_8bit": True,
        "do_sample": True
    },
    "HF_Gemma2": {
        "model": "google/gemma-2-2b",
        "temperature": 0.5,
        "max_new_tokens": 1024,
        "top_k": 5,
        "top_p":0.95,
        "load_in_8bit": True,
        "do_sample": True
    },
    "HF_Qwen2": {
        "model": "Qwen/Qwen2-7B-Instruct",
        "temperature": 0.5,
        "max_new_tokens": 1024,
        "top_k": 5,
        "top_p":0.95,
        "load_in_8bit": True,
        "do_sample": True
    },
}
