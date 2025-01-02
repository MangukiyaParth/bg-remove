from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import shutil
import os
import io
import time  # For simulating progress updates
import asyncio
import spaces
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from loadimg import load_img

app = FastAPI()

# Define the upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Create the directory if it doesn't exist
PROCESSED_DIR = "processed_uploads"
os.makedirs(PROCESSED_DIR, exist_ok=True)  # Create the directory if it doesn't exist

torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
birefnet.to(device)

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

progress = {}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file and save it to the server.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    task_id = file.filename  # Task ID based on filename for simplicity
    progress[task_id] = 0
    try:
        # Save the file to the server
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Update progress to indicate processing has started
        progress[task_id] = 10

        # Process the file
        processed_image = process_file(file_path, task_id)

        return StreamingResponse(
            processed_image,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename={file.filename.rsplit('.', 1)[0]}.png"
            },
        )

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )

    finally:
        file.file.close()
        if task_id in progress:
            del progress[task_id]


@app.get("/progress/{task_id}")
async def progress_updates(task_id: str):
    """
    Send real-time progress updates for the given task ID using SSE.
    """
    async def event_generator():
        while task_id in progress:
            yield {"data": str(progress[task_id])}
            await asyncio.sleep(1)  # Adjust interval as needed
        yield {"data": "100"}  # Indicate completion

    return EventSourceResponse(event_generator())


@spaces.GPU
def process(image, task_id):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to(device)

    # Simulate progress
    progress[task_id] = 30

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()

    progress[task_id] = 70

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)

    progress[task_id] = 90

    return image


def process_file(f, task_id):
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im, task_id)

    # Save the processed image to an in-memory buffer
    buffer = io.BytesIO()
    transparent.save(buffer, format="PNG")
    buffer.seek(0)  # Reset buffer pointer to the beginning

    progress[task_id] = 100  # Mark task as complete

    return buffer


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
