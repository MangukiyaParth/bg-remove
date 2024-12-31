from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
import shutil
import os
import io
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

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file and save it to the server.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save the file to the server
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        processed_image = process_file(file_path)
        # return JSONResponse(
        #     content={"new_file": new_file, "message": "Background removed successfully."},
        #     status_code=200,
        # )
        return StreamingResponse(
            processed_image, 
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={file.filename.rsplit('.', 1)[0]}.png"}
        )

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )

    finally:
        file.file.close()

@spaces.GPU
def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to(device)
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

def process_file(f):
    # filename = os.path.basename(f).rsplit(".", 1)[0] + ".png"
    # processed_path = os.path.join(PROCESSED_DIR, filename)
    
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    # transparent.save(processed_path)
    # return processed_path
    
    # Save the processed image to an in-memory buffer
    buffer = io.BytesIO()
    transparent.save(buffer, format="PNG")
    buffer.seek(0)  # Reset buffer pointer to the beginning

    return buffer

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
