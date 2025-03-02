from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_BLIP_description(image, max_length=50, min_length=10, temperature=1.0, num_beams=5):
    """
    Генерация текстового описания для изображения.
    """
    inputs = processor(image, return_tensors="pt")
    out = model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        temperature=temperature,
        num_beams=num_beams,
        early_stopping=True
    )
    description = processor.decode(out[0], skip_special_tokens=True)
    return description