from fastapi import FastAPI, File, UploadFile
import torch
import io
import cv2
import numpy as np
from PIL import Image
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI()

# อนุญาตให้ Frontend (Next.js) เชื่อมต่อ API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือกำหนดเป็น ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📌 แก้ไขโมเดลให้โหลดจากไฟล์โดยตรง (ไม่ใช้ torch.hub)
MODEL_PATH = "./models/CBC.pt"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))  # โหลดโมเดลจากไฟล์
model.eval()  # ตั้งค่าให้โมเดลเป็นโหมดประเมินผล

@app.get("/")
def root():
    return {"message": "YOLOv5 API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # แปลงภาพเป็น numpy array
        img_array = np.array(image)
        results = model(img_array)

        # Render ผลลัพธ์จาก YOLOv5
        results.render()

        # แปลงรูปกลับเป็น JPEG
        for img in results.ims:  
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, img_encoded = cv2.imencode('.jpg', img_bgr)
            return Response(content=img_encoded.tobytes(), media_type="image/jpeg")

    except Exception as e:
        return {"error": str(e)}

# สำหรับรัน Local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
