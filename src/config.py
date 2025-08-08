from typing import List
from pydantic import BaseModel
import yaml

class DataCfg(BaseModel):
    dataset_name: str
    test_split: float
    val_split: float
    seed: int
    output_dir: str

def load_config(path: str, model):
    with open(path) as f:
        data = yaml.safe_load(f)
    return model(**data)


class LoraSettings(BaseModel):
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]

class TrainCfg(BaseModel):
    model_name: str
    batch_size: int
    lr: float
    epochs: int
    lora: LoraSettings
    output_dir: str
