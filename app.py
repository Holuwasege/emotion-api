from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torchaudio
import torch
import io
from typing import Optional

app = FastAPI(
    title="Multimodal Emotion Recognition API",
    description="Detect emotions from speech or text using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loading for low memory
speech_model = None
text_tokenizer = None
text_model = None

def load_speech_model():
    global speech_model
    if speech_model is None:
        speech_model = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    return speech_model

def load_text_model():
    global text_tokenizer, text_model
    if text_model is None:
        text_tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-base")
        text_model = AutoModelForSequenceClassification.from_pretrained("tae898/emoberta-base")
    return text_tokenizer, text_model


@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    # Audio input
    if file:
        try:
            audio_bytes = await file.read()
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

            model = load_speech_model()
            preds = model(waveform.squeeze().numpy(), sampling_rate=sr, top_k=3)

            return {"mode": "audio", "emotion": preds[0]["label"], "top_predictions": preds}
        except Exception as e:
            return JSONResponse({"error": f"Audio error: {e}"}, status_code=500)

    # Text input
    if text:
        try:
            tokenizer, model = load_text_model()
            inputs = tokenizer(text, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            label_id = torch.argmax(probs).item()
            emotion = model.config.id2label[label_id]

            return {
                "mode": "text",
                "emotion": emotion,
                "probabilities": {
                    model.config.id2label[i]: float(round(p, 4))
                    for i, p in enumerate(probs[0].tolist())
                }
            }
        except Exception as e:
            return JSONResponse({"error": f"Text error: {e}"}, status_code=500)

    return JSONResponse({"error": "No valid input. Provide text or audio."}, status_code=400)

