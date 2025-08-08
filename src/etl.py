"""
ETL for Banking 77 dataset. EDA shows Banking77 ships as a single `train` split on hf.  

1. downloads the corpus
2. performs a stratified train/val/test split (80/10/10 by default),
3. stores each partition as Apache Parquet

run from the repo root:

    python -m etl

"""

from pathlib import Path

import pandas as pd
import typer
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .config import DataCfg, load_config

app = typer.Typer()

@app.command()
def run(cfg_path: str = "configs/data.yaml"):
    cfg = load_config(cfg_path, DataCfg)
    ds = load_dataset(cfg.dataset_name)
    label_map = ds["train"].features["label"].names # mapping to prevent labels being parsed as ints in train/testing
    df = pd.DataFrame(ds["train"])  # banking77 dataset only has 'train' split (no val/test)
    df["label_text"] = df["label"].apply(lambda i: label_map[i])
    train_df, temp = train_test_split(df, test_size=cfg.test_split+cfg.val_split,
                                      random_state=cfg.seed, stratify=df["label"])
    rel = cfg.val_split/(cfg.test_split+cfg.val_split)
    val_df, test_df = train_test_split(temp, test_size=1-rel,
                                       random_state=cfg.seed, stratify=temp["label"])
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir/"train.parquet")
    val_df.to_parquet(out_dir/"val.parquet")
    test_df.to_parquet(out_dir/"test.parquet")
    typer.echo("ETL complete.")

if __name__ == "__main__":
    app()
