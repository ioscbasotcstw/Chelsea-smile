import streamlit as st
from PIL import Image

class ImageCaption:
    def __init__(self):
        self.pil_image = None

    def load_image(self):
        # Використовуємо унікальний ключ для file_uploader
        self.image = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'], key="uploader_1")
        
        if self.image is not None:
            self.pil_image = Image.open(self.image)
            st.image(self.pil_image, caption="Uploaded Image", use_column_width=True)

        return self.pil_image

    def send2ai(self, model, prompt):
        if not model.model_name() == "gemini-1.5-flash":
            raise Exception(f"VLM should be gemini-1.5-flash but got {model.model_name()}")
        
        if self.pil_image is None:
            raise ValueError("Image should be np.ndarray or PIL.Image but got None")
        
        if prompt is None:
            raise ValueError("Prompt should be str but got None")
    
        response = model.execution().generate_content([prompt, self.pil_image])
        return response.text