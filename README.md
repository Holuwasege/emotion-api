# Emotion Detection API (FastAPI + Transformers)

### Install dependencies (Windows, Mac, Linux)

pip install -r requirements.txt -f https://download.pytorch.org/whl/cpu

### Run server
uvicorn app:app --reload

### Test on browser:
http://127.0.0.1:8000

### Test with Postman:
POST http://127.0.0.1:8000/predict
Body (JSON):
{
  "text": "I am very happy today!"
}
