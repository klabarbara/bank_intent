
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
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

app = typer.Typer(add_completion=False)

def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = torch.from_numpy(logits).argmax(dim=-1).cpu().numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


@app.command()
def main(
    train_config: str = "configs/train.yaml",
    sample_n: int = 0,
) -> None:
    cfg: TrainCfg = load_config(train_config, TrainCfg)  
    mlflow.set_experiment("lora_finetune")

    with mlflow.start_run():
        mlflow.log_params(vars(cfg))

        full_train_df = pd.read_parquet("data/processed/train.parquet")
        val_df   = pd.read_parquet("data/processed/val.parquet")


        # dev sample truncation
        if sample_n > 0:
            train_df = full_train_df.iloc[:sample_n].reset_index(drop=True)
            val_df   = val_df.iloc[: max(sample_n // 4, 1)].reset_index(drop=True)
            logger.info("dev run: train=%d  val=%d rows", len(train_df), len(val_df))
        else:
            train_df = full_train_df

        # encoding full set of labels regardlesss of training subset len
        encoder = LabelEncoder().fit(full_train_df["label_text"])
        train_df["labels"] = encoder.transform(train_df["label_text"])
        val_df["labels"] = encoder.transform(val_df["label_text"])
        num_labels = len(encoder.classes_)
        mlflow.log_param("num_labels", num_labels)


        
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        # returns dict  { "input_ids": Tensor, "attention_mask": Tensor } 
        # TODO: figure out type hint for ^
        def tok(batch):
            return tokenizer(
                batch["text"], truncation=True, padding="max_length", max_length=64
            )

        # batched tokenization
        train_ds = Dataset.from_pandas(train_df[["text", "labels"]]).map(tok, batched=True)
        val_ds = Dataset.from_pandas(val_df[["text", "labels"]]).map(tok, batched=True)
        
        cols = ["input_ids", "attention_mask", "labels"]
        train_ds.set_format("torch", columns=cols)
        val_ds.set_format("torch", columns=cols)
        """
        every item in train_ds and val_ds formatted like: 
        {
            "input_ids": torch.LongTensor([…64 ints…]),
            "attention_mask": torch.LongTensor([…64 ints…]),
            "labels": torch.LongTensor([label_id])
        }
        """

        base_model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=num_labels,
            id2label={i: l for i, l in enumerate(encoder.classes_)},
            label2id={l: i for i, l in enumerate(encoder.classes_)},
        )

        lora_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            bias="none",
            task_type="SEQ_CLS",
        )

        model = get_peft_model(base_model, lora_cfg)
        model.print_trainable_parameters()  

        args = TrainingArguments(
            disable_tqdm=True,          
            logging_strategy="epoch",   
            fp16=torch.cuda.is_available(),  
            output_dir=cfg.output_dir,
            learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay,
            eval_strategy="epoch",
            num_train_epochs=cfg.epochs,
            logging_steps=50,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            load_best_model_at_end=True,
            save_strategy="epoch",
            metric_for_best_model="macro_f1",
            report_to="none",
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
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        mlflow.log_artifacts(out_dir, artifact_path="model")
        mlflow.log_text(str(metrics), "metrics.txt")

        logger.info(
            "lora fine-tune complete — acc: %.3f | macro-F1: %.3f",
            metrics.get("accuracy", -1),
            metrics.get("macro_f1", -1),
        )


if __name__ == "__main__":
    app()
