
import logging
from pathlib import Path
from typing import Dict

import json
import mlflow
import pandas as pd
import torch
import typer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import MLflowCallback

from .config import load_config, TrainCfg

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", level=logging.INFO
)

app = typer.Typer(add_completion=False)


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    # hacky torch use to reuse argmax as logits is a np.ndarray
    preds = torch.from_numpy(logits).argmax(dim=-1).cpu().numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

def save_labels_json(path: Path, classes) -> Path:
    id2label = {i: str(lbl) for i, lbl in enumerate(classes)}
    label2id = {str(lbl): i for i, lbl in enumerate(classes)}
    payload = {"id2label": id2label, "label2id": label2id}
    path.mkdir(parents=True, exist_ok=True)
    f = path / "labels.json"
    f.write_text(json.dumps(payload, indent=2))
    return f

@app.command()
def main(
    train_config: str = "configs/train.yaml",
    sample_n: int = 0,
    regime: str = typer.Option(
        "lora", "--regime", "-r", help="head-only, lora, full-ft"
    ),
) -> None:
    cfg: TrainCfg = load_config(train_config, TrainCfg)

    mlflow.set_experiment("bank_intent")
    with mlflow.start_run():
        # tags for MLFlow tables
        mlflow.set_tags(
            {"dataset": "banking77", "backbone": cfg.model_name, "regime": regime}
        )

        # raw data
        full_train_df = pd.read_parquet("data/processed/train.parquet")
        val_df = pd.read_parquet("data/processed/val.parquet")

        if sample_n > 0:
            train_df = full_train_df.iloc[:sample_n].reset_index(drop=True)
            val_df = val_df.iloc[: max(sample_n // 4, 1)].reset_index(drop=True)
            logger.info("dev run: train=%d  val=%d", len(train_df), len(val_df))
        else:
            train_df = full_train_df

        # label encoding fit on all labels in training regardless of sample-n
        encoder = LabelEncoder().fit(full_train_df["label_text"])
        train_df["labels"] = encoder.transform(train_df["label_text"])
        val_df["labels"] = encoder.transform(val_df["label_text"])
        num_labels = len(encoder.classes_)

        # tokenization
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        def tok(batch):
            return tokenizer(
                batch["text"], truncation=True, padding="max_length", max_length=64
            )

        train_ds = Dataset.from_pandas(train_df[["text", "labels"]]).map(
            tok, batched=True
        )
        val_ds = Dataset.from_pandas(val_df[["text", "labels"]]).map(tok, batched=True)
        cols = ["input_ids", "attention_mask", "labels"]
        train_ds.set_format("torch", columns=cols)
        val_ds.set_format("torch", columns=cols)

        # model init
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=num_labels,
            id2label={i: l for i, l in enumerate(encoder.classes_)},
            label2id={l: i for i, l in enumerate(encoder.classes_)},
        )

        # choose your fighter
        if regime == "lora":
            lora_cfg = LoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=cfg.lora.target_modules,
                bias="none",
                task_type="SEQ_CLS",
                modules_to_save=["classifier"],  # keep head trainable/savable
            )
            model = get_peft_model(base, lora_cfg)
        elif regime == "head-only":
            model = base
            for name, p in model.named_parameters():
                if not (name.startswith("classifier") or name.startswith("pooler")):
                    p.requires_grad = False
        elif regime == "full-ft":
            model = base
            for p in model.parameters():
                p.requires_grad = True
        else:
            raise typer.BadParameter("regime must be one of: head-only | lora | full-ft")

        # log params flatten lora block
        params = cfg.model_dump()
        params.update({f"lora.{k}": v for k, v in params.pop("lora").items()})
        params.update(
            {
                "num_labels": num_labels,
                "sample_n": sample_n,
            }
        )
        mlflow.log_params(params)

        # trainable vs total
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        mlflow.log_params({"trainable_params": int(trainable), "total_params": int(total)})
        logger.info("trainable params: %d / %d", trainable, total)

        args = TrainingArguments(
            disable_tqdm=True,          
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay,
            num_train_epochs=cfg.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            logging_steps=50,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[MLflowCallback()],
        )

        trainer.train()
        final = trainer.evaluate()

        # explicit final log
        mlflow.log_metrics(final)

        # persist artifacts
        out_root = Path(cfg.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        # freeze label schema 
        labels_file = save_labels_json(out_root, encoder.classes_)

        if regime == "lora":
            adapter_dir = out_root / "adapter"    # lora weights/peft
            base_dir    = out_root / "base_77"    # label base for reattach
            merged_dir  = out_root / "merged"     # fully merged hf model (no peft required in evbal now)

            # save the peft adapter, model is PeftModel because we used get_peft_model()
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_dir)

            # save the label base for lora (same head/label maps as training)
            base.save_pretrained(base_dir)
            tokenizer.save_pretrained(base_dir)
            save_labels_json(base_dir, encoder.classes_)

            # merge and save a standalone model
            peft_model: PeftModel = trainer.model  # already on best weights from load_best_model_at_end=True
            merged = peft_model.merge_and_unload()  # bake LoRA into base weights
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            save_labels_json(merged_dir, encoder.classes_)

            # logs artifacts to mlflow
            mlflow.log_artifacts(adapter_dir, artifact_path="model/adapter")
            mlflow.log_artifacts(base_dir,    artifact_path="model/base_77")
            mlflow.log_artifacts(merged_dir,  artifact_path="model/merged")
            mlflow.log_artifact(str(labels_file), artifact_path="model")

            # also log which path is the “primary” model to use for eval/deploy for posterity
            mlflow.set_tags({"export.primary_model": "merged", "export.contains": "adapter,base_77,merged"})

        else:
            # head-only or full-ft are both already standalone HF models
            model_dir = out_root / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            save_labels_json(model_dir, encoder.classes_)

            mlflow.log_artifacts(model_dir, artifact_path="model")
            mlflow.log_artifact(str(labels_file), artifact_path="model")

        # plaintext eval summary
        mlflow.log_text(str(final), "metrics.txt")


if __name__ == "__main__":
    app()
