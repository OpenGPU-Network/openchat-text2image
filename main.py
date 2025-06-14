
from pydantic import BaseModel

import torch
from diffusers import DiffusionPipeline
import base64
from io import BytesIO
import threading

import ogpu.service

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
if torch.cuda.is_available():
    DEVICE = "cuda"
    TORCH_DTYPE = torch.float16
    VARIANT = "fp16"
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32
    VARIANT = None

PIPE = None
PIPE_LOCK = threading.Lock()  # Lock to ensure thread-safe access to PIPE

@ogpu.service.init()
def setup():
    global PIPE
    ogpu.service.logger.info(f"Pulling {MODEL_ID} model...")
    PIPE = DiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=TORCH_DTYPE, 
        use_safetensors=True, 
        variant=VARIANT
    )
    PIPE = PIPE.to(DEVICE)
    ogpu.service.logger.info(f"{MODEL_ID} pulled.")


class InputData(BaseModel):
    prompt: str = "a photo of an astronaut riding a horse on mars"

class OutputData(BaseModel):
    image_base64: str

@ogpu.service.expose()
def text2image(input_data: InputData) -> OutputData:
    global PIPE
    
    output_data = {}
    try:
        # Use lock to ensure only one request processes at a time
        with PIPE_LOCK:
            ogpu.service.logger.info(f"Processing request: {input_data.prompt[:50]}...")
            image = PIPE(input_data.prompt).images[0]
            img_bytes = BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")
            output_data = OutputData(image_base64=img_base64)
            ogpu.service.logger.info(f"Task completed.")
    except Exception as e:
        ogpu.service.logger.error(f"An error occurred: {e}")
        raise e

    return output_data

ogpu.service.start()