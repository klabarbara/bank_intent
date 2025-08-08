"""
zero-shot benchmark using typeform/distilbert-base-uncased-mnli for Banking77 ds : (distilbert used for dev, will use facebook/bart-large-mnli for production)

    1. reads Parquet test set and class list from the training split.
    2. creates id2label mapping to ensure labels' semantic meaning is preserved while allowing int label usage for metrics
    2. feeds natural language input to a transformers zero shot pipeline.
    3. logs Accuracy & Macro-F1 to MLflow, plus a full classification report.

"""
import os, time

import mlflow, typer, pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

app = typer.Typer()

@app.command()
def run(
        data_path: str = "data/processed/test.parquet",
        label_map_path: str = "data/processed/train.parquet",
        timeit: bool = False,
        sample_n: int = 0, # enables cli level sample ctrl for dev
    ):    
    mlflow.set_experiment("baseline")
    with mlflow.start_run():
        test = pd.read_parquet(data_path)

        if sample_n > 0:
            test = test.iloc[:sample_n].reset_index(drop=True)
            typer.echo(f"running on sample of {sample_n} rows")

        train = pd.read_parquet(label_map_path)
        id2label = sorted(train["label_text"].unique().tolist())
        classifier = pipeline("zero-shot-classification",
                              model="typeform/distilbert-base-uncased-mnli",
                            #   device_map="auto",
                              device=0,)
        batch_size = 16
        texts = list(test["text"])
        preds = []

        # speed test to confirm appropriate gpu usage
        if timeit:
            sample = "I want to withdraw cash"
            print(f"Timing classification for: \"{sample}\"")
            times = []
            for _ in range(5):
                start = time.time()
                classifier(sample, candidate_labels=id2label)
                times.append(time.time() - start)
            avg = sum(times) / len(times)
            print(f"Average inference time: {avg:.3f} seconds over 5 runs")
            return  
        
        # note that HF's pipeline() treats the batches as sequence of calls
        # will give erroneous "You seem to be using the pipelines 
        # sequentially on GPU warning"
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Zero-shot classifying"):
            batch = texts[i:i+batch_size]
            out_batch = classifier(batch, candidate_labels=id2label, multi_label=False)
            preds.extend([out["labels"][0] for out in out_batch])

        encoder = LabelEncoder().fit(train["label_text"])   
        y_true = encoder.transform(test["label_text"])
        y_pred = encoder.transform(preds)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_f1", f1)

        # generating list of all possible labels for dev sets < len(labels)
        all_labels = list(range(len(encoder.classes_)))

        report = classification_report(
            y_true,
            y_pred,
            labels=all_labels,
            target_names=encoder.classes_,
            digits=3,
            zero_division=0, # avoids warnings when class is missing
        )
        
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/baseline_report.txt","w") as f:
            f.write(report)
        mlflow.log_artifact("artifacts/baseline_report.txt")
        typer.echo(f"Baseline acc={acc:.3f} macro-F1={f1:.3f}")

if __name__ == "__main__":
    app()
