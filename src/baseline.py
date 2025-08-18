"""
zero-shot benchmark now with:
 - domain-aware hypothesis_template
 - label text normalisation  (atm_support -> "atm support")
 - MNLI-tuned backbone by default (facebook/bart-large-mnli) rather than general BERT architecture used in multiregime

"""
import os, time

import mlflow, typer, pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import torch

app = typer.Typer()

HYPOTHESIS_TEMPLATE = "This banking query is about {}."

@app.command()
def run(
        data_path: str = "data/processed/test.parquet",
        label_map_path: str = "data/processed/train.parquet",
        timeit: bool = False,
        sample_n: int = 0, # enables cli level sample ctrl for dev
        model_name: str = "facebook/bart-large-mnli", # microsoft/deberta-v3-base used in multiregime not useful for zero shot
        batch_size: int = 64,
    ):    
    mlflow.set_experiment("baseline")
    with mlflow.start_run():
        # tags to distinguish from lora peft
        mlflow.set_tags({
            "phase": "eval",
            "dataset": "banking77",
            "split": "test",
            "baseline_type": "zero_shot_nli",  
            "model_name": model_name
        })

        # data
        test = pd.read_parquet(data_path)
        train = pd.read_parquet(label_map_path)

        if sample_n > 0:
            test = test.iloc[:sample_n].reset_index(drop=True)
            typer.echo(f"running on sample of {sample_n} rows")

        id2label = (
            train["label_text"]
            .str.strip() # TODO underscore replacement in etl.py, consider moving the rest over as well
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        # pipeline
        device_id = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("zero-shot-classification",
                              model=model_name,
                              hypothesis_template=HYPOTHESIS_TEMPLATE,
                            #   device_map="auto",
                              device=device_id)
        
        texts, preds = test["text"].tolist(), []

        # inference
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

        # metrics 
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
            labels=list(range(len(encoder.classes_))),
            target_names=encoder.classes_,
            digits=3, zero_division=0, # avoids warnings when class is missing
        )
        
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/baseline_report.txt","w") as f:
            f.write(report)
        mlflow.log_artifact("artifacts/baseline_report.txt")
        typer.echo(f"Baseline acc={acc:.3f} macro-F1={f1:.3f}")

if __name__ == "__main__":
    app()
