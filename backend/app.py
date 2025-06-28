from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import json
import base64

# IMPORT HÀM INFERENCE TỪ PIPELINE CỦA BẠN
from pipeline import Inference

app = FastAPI()

# Cấu hình CORS (giữ nguyên, đã đúng)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "AI Server is running and ready to process images."}


@app.post("/api")
async def process_image_and_get_services(
    file: UploadFile = File(...),
    services: str = Form(...),  # Nhận chuỗi JSON config từ frontend
):
    """
    Nhận ảnh và cấu hình dịch vụ, chạy pipeline AI, và trả về kết quả.
    :param file: Tập tin ảnh được tải lên.
    :param services: Chuỗi JSON chứa mapping từ service -> list of messes.
    :return: JSON chứa ảnh kết quả (base64) và danh sách dịch vụ cần thiết.
    """
    try:
        # 1. Đọc và xử lý input
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Chuyển chuỗi JSON từ frontend thành dictionary Python
        service_to_mess_config = json.loads(services)

        # 2. GỌI PIPELINE: Đây là phần kết nối chính
        # Hàm Inference của bạn trả về (ảnh đã vẽ, set các tuple (service, score))
        result_img, services_with_scores = Inference(image, service_to_mess_config)

        # 3. Chuẩn bị response để gửi về frontend
        # Chuyển ảnh kết quả (PIL.Image) thành chuỗi base64
        buffered = io.BytesIO()
        result_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Tạo chuỗi data URI để frontend có thể hiển thị trực tiếp
        base64_image = f"data:image/jpeg;base64,{img_str}"

        # *** MODIFICATION START ***
        # Chuyển set các tuple (service, score) thành list các dictionary
        # để frontend dễ dàng xử lý và sắp xếp.
        # Ví dụ: {('cleaning', 0.8), ('laundry', 0.5)} -> [{'name': 'cleaning', 'score': 0.8}, {'name': 'laundry', 'score': 0.5}]
        services_list = [
            {"name": service, "score": score} for service, score in services_with_scores
        ]
        # *** MODIFICATION END ***

        # 4. Trả về kết quả
        return JSONResponse(
            content={
                "result_image": base64_image,
                "services_needed": services_list,  # Gửi đi danh sách có cấu trúc
                "filename": file.filename,
            }
        )

    except Exception as e:
        # Ghi lại lỗi để debug
        import traceback

        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred in the pipeline: {str(e)}"},
        )


# Hàm để chạy bằng python app.py (giữ nguyên)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
