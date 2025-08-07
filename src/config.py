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
