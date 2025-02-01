import base64
from fastapi import FastAPI, File, UploadFile
import torch
import io
import cv2
import numpy as np
from PIL import Image
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# อนุญาตให้ Frontend เชื่อมต่อ API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล YOLOv5
MODEL_PATH = "./models/CBC.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
model.conf = 0.4  

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)
        results = model(img_array)
        
        # Render ภาพ
        results.render()
        detected_classes = [model.names[int(cls)] for cls in results.xyxy[0][:, -1]]

        # นับจำนวนแต่ละคลาส
        cell_counts = {cls: detected_classes.count(cls) for cls in set(detected_classes)}

        # แปลงภาพเป็น Base64
        _, img_encoded = cv2.imencode('.jpg', results.ims[0])
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return JSONResponse(content={"cell_counts": cell_counts, "image": img_base64})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "YOLOv5 API is running!"}
