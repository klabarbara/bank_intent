import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

test = pd.read_parquet("data/processed/test.parquet")
train = pd.read_parquet("data/processed/train.parquet")

id2label = sorted(train["label"].unique().tolist())

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli",
                      device_map="auto")

batch_size = 8
texts = list(test["text"])
preds = []

for i in tqdm(range(0, len(texts), batch_size), desc="Zero-shot classifying"):
    batch = texts[i:i+batch_size]
    out_batch = classifier(batch, candidate_labels=id2label, multi_label=False)
    preds.extend([out["labels"][0] for out in out_batch])

acc = accuracy_score(test["label"], preds)
f1 = f1_score(test["label"], preds, average="macro")
report = classification_report(test["label"], preds)

print(f"Accuracy: {acc:.3f}")
print(f"Macro F1: {f1:.3f}")
print("\nClassification report:\n", report)

