import typer, pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from .config import load_config, DataCfg


def run(cfg_path: str = "configs/data.yaml"):
    cfg = load_config(cfg_path, DataCfg)
    ds = load_dataset(cfg.dataset_name)
    df = pd.DataFrame(ds["train"])  # banking77 dataset only has 'train' split (no val/test)
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
