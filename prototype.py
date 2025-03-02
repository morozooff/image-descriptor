#TODO: 
# separate prototype logic
# add some features

import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image, min_length=10, max_length=50, temperature=1.0, num_beams=5):
    inputs = processor(image, return_tensors="pt")

    out = model.generate(
        **inputs,
        min_length=min_length,
        max_length=max_length,
        temperature=temperature,
        num_beams=num_beams,
        early_stopping=True
    )
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

image_descriptor_view = gr.Interface(
    fn=generate_caption, 
    inputs=[
        gr.Image(type="pil", label="Загрузите изображение"),
        gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Минимальная длина текста"),
        gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Максимальная длина текста"),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Случайность слов генерации"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Качество генерации)")
    ],
    outputs=gr.Textbox(label="Текстовое описание"),
    title="Генерация текстового описания изображения",
    description="Загрузите изображение и настройте параметры генерации текста."
)

image_descriptor_view.launch()