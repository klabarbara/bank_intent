
import logging
from pathlib import Path
from typing import List

import mlflow
import numpy as np
import pandas as pd
import torch
import typer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from peft import AutoPeftModelForSequenceClassification, PeftModel


app = typer.Typer(add_completion=False)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # TODO why doesn't setting it in basicConfig work

def print_eval_summary(acc: float, macro_f1: float, n: int, fig_path: Path):
    # Minimal, dependency-free table
    line = "-" * 50
    print("\n" + line)
    print(f"{'Banking77 Evaluation Summary':^48}")
    print(line)
    print(f"{'Samples':<24}{n:>24}")
    print(f"{'Accuracy':<24}{acc:>24.3f}")
    print(f"{'Macro-F1':<24}{macro_f1:>24.3f}")
    print(line)
    print(f"Confusion matrix image: {fig_path}")
    print(line + "\n")

@app.command()
def run(
    checkpoint_path: str = "artifacts/models/deberta_lora",
    data_path: str = "data/processed/test.parquet",
    batch_size: int = 64,
    max_length: int = 128,
    device_id: int = 0,
    plots_dir: str = "artifacts/plots",
) -> None:
    
    # data
    df = pd.read_parquet(data_path)
    texts = df["text"].tolist()
    # groundtruth as label_text
    y_true_text = df["label_text"].tolist()

    # tokenizer and model
    device = torch.device("cuda", device_id) if (device_id >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else None

    cp = Path(checkpoint_path)
    merged_dir  = cp / "merged"
    adapter_dir = cp / "adapter"
    base77_dir  = cp / "base_77"

    # checks for models from 'best case' scenario to worst
    if merged_dir.exists():
        # single standard HF model, adapter already merged
        tokenizer = AutoTokenizer.from_pretrained(merged_dir)
        model = AutoModelForSequenceClassification.from_pretrained(merged_dir, torch_dtype=dtype)

    elif adapter_dir.exists() and base77_dir.exists():
        # evaluates via saved 77 label label base and adapter
        tokenizer = AutoTokenizer.from_pretrained(base77_dir)
        base = AutoModelForSequenceClassification.from_pretrained(base77_dir, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, str(adapter_dir))

    elif adapter_dir.exists() and not base77_dir.exists():
        # adapter found but no matching base_77, steer user to merged or provide base_77
        raise RuntimeError(
            f"Found adapter at {adapter_dir} but no base_77 directory. "
            f"Evaluate the merged model ({merged_dir}) or provide a base_77 to attach the adapter."
        )

    else:
        # plain checkpoint dir, no layout/subfolder assumption made
        tokenizer = AutoTokenizer.from_pretrained(cp)
        model = AutoModelForSequenceClassification.from_pretrained(cp, torch_dtype=dtype)

    model.to(device)
    model.eval()



    # mappings derived from modals, normalizing labels/true tags
    id2label = {int(k): v for k, v in model.config.id2label.items()} if isinstance(model.config.id2label, dict) else model.config.id2label
    label2id = {v.strip().lower(): int(k) for k, v in id2label.items()}
    y_true = np.array([label2id[y.strip().lower()] for y in y_true_text])

    

    # sort numerically by key to prevent out of order keys from dict
    classes = [v for k, v in sorted(id2label.items(), key=lambda kv: int(kv[0]))]
    # sorted numeric keys for classification report and confusion mtx
    labels = [int(k) for k in sorted(id2label.keys(), key=int)]

    # map y_true to ids 

    # batched inference
    preds: List[int] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())

    y_pred = np.array(preds, dtype=np.int64)

    # metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report_txt = classification_report(y_true, y_pred, labels=labels, target_names=classes, digits=3, zero_division=0)

    # confusion mtx, normalized by true class
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    # save plot 
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(cm, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)")
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    plt.xticks(rotation=90, fontsize=7); plt.yticks(fontsize=7)
    plt.tight_layout()

    plots_dir = Path(plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plots_dir / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print_eval_summary(acc, macro_f1, n=len(y_true), fig_path=fig_path)

    mlflow.set_experiment("bank_intent")
    with mlflow.start_run():
        mlflow.set_tags({"phase": "eval", "dataset": "banking77", "split":"test"})
        mlflow.log_params({
            "checkpoint_path": checkpoint_path,
            "batch_size": batch_size,
            "max_length": max_length,
        })
        mlflow.log_metrics({"accuracy": acc, "macro_f1": macro_f1})
        mlflow.log_artifact(str(fig_path))
        mlflow.log_text(report_txt, "classification_report.txt")

    logger.info("Eval — Acc: %.3f | Macro-F1: %.3f", acc, macro_f1)

if __name__ == "__main__":
    app()
