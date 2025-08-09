
import logging
from pathlib import Path
from typing import Dict

import mlflow
import pandas as pd
import torch
import typer
from datasets import Dataset
from peft import LoraConfig, get_peft_model
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


@app.command()
def main(
    train_config: str = "configs/train.yaml",
    sample_n: int = 0,
    regime: str = typer.Option(
        "lora", "--regime", "-r", help="one of: head-only | lora | full-ft"
    ),
) -> None:
    cfg: TrainCfg = load_config(train_config, TrainCfg)

    mlflow.set_experiment("banking77")
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
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        mlflow.log_artifacts(out_dir, artifact_path="model")
        mlflow.log_text(str(final), "metrics.txt")

        logger.info(
            "final: %s — Acc: %.3f | Macro-F1: %.3f",
            regime,
            final.get("eval_accuracy", -1.0),
            final.get("eval_macro_f1", -1.0),
        )


if __name__ == "__main__":
    app()
