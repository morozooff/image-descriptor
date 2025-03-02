import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from utils.nlp_integrations import generate_BLIP_description
from .schemas import InputBLIPParameters, ImageDescription

app = FastAPI(
    title="Image Descriptor API",
    description="API для генерации текстового описания изображений с использованием модели BLIP.",
    version="1.0.0"
)

@app.post("/generate-description/", response_model=ImageDescription)
async def create_description(
    request_image: UploadFile = File(..., description="Изображение для генерации описания"),
    params: str = Form(..., description="Параметры в формате JSON")
):
    """
    Генерация текстового описания для загруженного изображения.

    - **request_image**: Изображение в формате JPEG или PNG.
    - **params**: Параметры генерации текста (макс. длина, мин. длина, температура и т.д.).
    """

    try:
        params_dict = json.loads(params)
        params_model = InputBLIPParameters(**params_dict)
    except json.JSONDecodeError:
        return JSONResponse(
            content={"error": "Invalid JSON format for params"},
            status_code=400
        )
    except Exception as e:
        return JSONResponse(
            content={"error": f"Invalid params: {str(e)}"},
            status_code=400
        )
    image = Image.open(request_image.file)
    
    description = generate_BLIP_description(
        image,
        max_length=params_model.max_length,
        min_length=params_model.min_length,
        temperature=params_model.temperature,
        num_beams=params_model.num_beams
    )
    
    return JSONResponse(content={"description": description})