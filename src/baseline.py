import os

import mlflow, typer, pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

app = typer.Typer()

@app.command()
def run(data_path: str = "data/processed/test.parquet",
        label_map_path: str = "data/processed/train.parquet"):    
    mlflow.set_experiment("baseline")
    with mlflow.start_run():
        test = pd.read_parquet(data_path)

        train = pd.read_parquet(label_map_path)
        id2label = sorted(train["label"].unique().tolist())
        classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli",
                              device_map="auto")
        batch_size = 16
        texts = list(test["text"])
        preds = []
        
        # note that HF's pipeline() treats the batches as sequence of calls
        # will give erroneous "You seem to be using the pipelines 
        # sequentially on GPU warning"
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Zero-shot classifying"):
            batch = texts[i:i+batch_size]
            out_batch = classifier(batch, candidate_labels=id2label, multi_label=False)
            preds.extend([out["labels"][0] for out in out_batch])
        acc = accuracy_score(test["label"], preds)
        f1 = f1_score(test["label"], preds, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_f1", f1)

        report = classification_report(test["label"], preds, output_dict=False)
        
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/baseline_report.txt","w") as f:
            f.write(report)
        mlflow.log_artifact("artifacts/baseline_report.txt")
        typer.echo(f"Baseline acc={acc:.3f} macro-F1={f1:.3f}")

if __name__ == "__main__":
    app()
