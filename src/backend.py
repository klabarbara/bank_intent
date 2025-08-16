import json
import os
from pathlib import Path
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from starlette.responses import JSONResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from dotenv import load_dotenv

# config
load_dotenv()  # optional in case of model residing in .env

MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/models/deberta_lora/merged")).expanduser().resolve()
LABELS_FILE = MODEL_DIR / "labels.json"

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"MODEL_DIR={MODEL_DIR} does not exist. "
        "Set correct path via environment variable or .env file."
    )

# load model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    with LABELS_FILE.open() as fp:
        maps: Dict[str, Dict[str, str]] = json.load(fp)
        id2label = {int(k): v for k, v in maps["id2label"].items()}
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(f"Model initialisation failed: {exc}") from exc


app = FastAPI(title="Bank77 Intent API")


class Query(BaseModel):
    text: str


@app.post("/predict/")
def predict(q: Query) -> Dict[str, str]:
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty!")

    inputs = tokenizer(
        q.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    with torch.inference_mode():
        logits = model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=-1))
    return {"prediction": id2label[pred_id]}


@app.exception_handler(Exception)
def unhandled_exc_handler(_, exc: Exception): 
    
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Unhandled error: {exc}"},
    )
