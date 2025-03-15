# python core libraries
import re
import psutil
import time
import random
# streamlit
import streamlit as st
import streamlit.components.v1 as components
# components from other authors
from streamlit_mic_recorder import mic_recorder
# core modules
from audio_processing.A2T import A2T
from audio_processing.T2A import T2A
from llm.utils.chat import Conversation
from vlm.vlm import VLM
# utils modules
from utils.keywords import keywords
from utils.prompt_toggle import select_prompt, load_prompts
from utils.image_caption import ImageCaption
from utils.documentation import html_content
from utils.payment import html_doge_wallet

prompts = load_prompts()
chat = Conversation()
t2a = T2A()
vlm = VLM()
ic = ImageCaption()
text_dict = {}

def remove_labels_with_regex(text: str):
    pattern = r'^(Human:|AI:|Chelsea:)\s*'
    cleaned_text = re.sub(pattern, '', text, flags=re.MULTILINE)
    return cleaned_text

def exctrator(sentence, phrase):
    extracted_text = sentence.split(phrase)[1].strip() if phrase in sentence else ""
    return extracted_text

def switching(text):
    result = None

    if re.search("show me your image", text.lower(), re.IGNORECASE):
        prompt = exctrator(text.lower(), phrase="show me your image")
        # Завантажуємо зображення
        uploaded_image = ic.load_image()

        if uploaded_image is not None:
            # Якщо зображення завантажено, виконуємо обробку
            result = ic.send2ai(model=vlm, prompt=prompt)
        else:
            # Якщо зображення ще не завантажене, показуємо попередження
            st.warning("No image uploaded yet. Please upload an image to continue.")
    elif re.search("show me documentation", text.lower(), re.IGNORECASE):
        components.html(html_content, height=800, scrolling=True)
    elif re.search("pay the ghost", text.lower(), re.IGNORECASE):
        components.html(html_doge_wallet, height=600, scrolling=False)  
    else:
        prompt = select_prompt(input_text=text, prompts=prompts, keywords=keywords)
        result = chat.chatting(prompt=prompt if prompt is not None else text)

    print(f"Prompt:\n{prompt}")
    return result

def get_text():
    try:
        mic = mic_recorder(start_prompt="Record", stop_prompt="Stop", just_once=False, use_container_width=True)
        start_time = time.perf_counter()
        a2t = A2T(mic["bytes"])
        text = a2t.predict()
        print(f"Text from A2T:\n{text}")
        execution_time = time.perf_counter() - start_time
        print(f"App.py -> get_text() -> time of execution A2T -> {execution_time}s")
        text_dict['text'] = text

        return text
    except Exception as e:
        print(f"An error occurred in get_text function, reason is: {e}")
        return None  # Повертаємо None у випадку помилки

def speaking(text):
    try:
        if text and text.strip() != "":
            print(f"Checking for execution this part {random.randint(0, 5)}")
            output = switching(text)
            response = remove_labels_with_regex(text=output)
            start_time_t2a = time.perf_counter()
            t2a.autoplay(response)
            execution_time_t2a = time.perf_counter() - start_time_t2a
            print(f"App.py -> speaking() -> time of execution T2A -> {execution_time_t2a}s")
            print(ic.pil_image)

            if response:
                st.markdown(f"Your input: {text}")
                st.markdown(f"Chelsea response: {response}")

    except Exception as e:
        print(f"An error occurred in speaking function, reason is: {e}")

def main():
    text = get_text()  
    
    if text is None and 'text' in text_dict:
        text = text_dict['text']
    
    print(f"Text dict: {text_dict}")
    print(f"Print text: s{text}s")
    speaking(text)
    print(f"Checking for execution main func {random.randint(0, 10)}")

if __name__ == "__main__":
    main()