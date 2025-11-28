# -------------------------------------------------------------
# app.py — Multimodal Emotion Recognition API (Audio + Text)
# -------------------------------------------------------------

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torchaudio
import torch
import io

# -------------------------------------------------------------
# Initialize FastAPI
# -------------------------------------------------------------
app = FastAPI(
    title="Emotion Recognition API",
    description="Detect emotions from Speech or Text",
    version="1.0.0"
)

# Allow frontend everywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Load Models Once
# -------------------------------------------------------------
print("Loading models...")

speech_classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)

text_tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-base")
text_model = AutoModelForSequenceClassification.from_pretrained(
    "tae898/emoberta-base"
)

print("Models loaded successfully.")

# -------------------------------------------------------------
# Main Endpoint
# -------------------------------------------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(None),
    text: str = Form(None)
):
    """
    Provide:
    - audio file (wav/mp3)
    - OR text
    """
    # AUDIO MODE
    if file:
        try:
            audio_bytes = await file.read()
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

            preds = speech_classifier(waveform.squeeze().numpy(), sampling_rate=sr)
            return {
                "mode": "audio",
                "emotion": preds[0]["label"],
                "top_predictions": preds
            }

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # TEXT MODE
    if text and text.strip():
        try:
            inputs = text_tokenizer(text, return_tensors="pt", truncation=True)
            outputs = text_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            idx = torch.argmax(probs).item()
            label = text_model.config.id2label[idx]

            return {
                "mode": "text",
                "emotion": label,
                "probabilities": {
                    text_model.config.id2label[i]: float(probs[0][i])
                    for i in range(len(probs[0]))
                }
            }

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # NOTHING PROVIDED
    return JSONResponse(
        {"error": "Provide audio or text."}, status_code=400
    )


@app.get("/")
def home():
    return {"message": "Emotion API is running!"}
