FROM python:3.11-slim
WORKDIR /app

# เพิ่มบรรทัดนี้ เพื่อติดตั้ง System Libraries ที่ Mediapipe ต้องการ
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "kaya:app"]