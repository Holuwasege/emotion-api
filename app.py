# -------------------------------------------------------------
# app.py — Text-Only Emotion Recognition (EmoBERTa)
# FastAPI API
# -------------------------------------------------------------
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -------------------------------------------------------------
# 1️⃣ Initialize FastAPI
# -------------------------------------------------------------
app = FastAPI(
    title="Text Emotion Recognition API",
    description="Detect emotions from text using EmoBERTa",
    version="1.0.0"
)

# Allow any frontend or Postman to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# 2️⃣ Load EmoBERTa Model (global = faster)
# -------------------------------------------------------------
MODEL_NAME = "tae898/emoberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# -------------------------------------------------------------
# 3️⃣ FastAPI Route → /predict (TEXT ONLY)
# -------------------------------------------------------------
@app.post("/predict")
async def predict(text: str = Form(...)):
    """
    Accepts text input only.
    Returns emotion and probabilities.
    """

    try:
        # Tokenize the input
        inputs = tokenizer(text, return_tensors="pt", truncation=True)

        # Run model
        with torch.no_grad():
            outputs = model(**inputs)

        # Softmax → probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Predicted label
        label_id = torch.argmax(probs).item()
        emotion = model.config.id2label[label_id]

        return {
            "mode": "text",
            "text": text,
            "emotion": emotion,
            "probabilities": {
                model.config.id2label[i]: float(round(p, 4))
                for i, p in enumerate(probs[0].tolist())
            }
        }

    except Exception as e:
        return JSONResponse({"error": f"Text error: {e}"}, status_code=500)

# -------------------------------------------------------------
# Test Route
# -------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Text Emotion API is running!"}
