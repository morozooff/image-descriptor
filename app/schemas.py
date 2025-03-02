from pydantic import BaseModel

class InputBLIPParameters(BaseModel):
    """Схема для валидации параметров входа модели."""
    max_length: int = 50
    min_length: int = 10
    temperature: float = 1.0
    num_beams: int = 5

class ImageDescription(BaseModel):
    """Схема для валидации описания, сгенерированного моделью."""
    text: str