# Banking77 Intent Classification
## Why I Chose LoRA (and How I Built It)

## TL;DR

Demo available at: https://huggingface.co/spaces/KayLaB/bankintent

- Task: Single sentence user query classification on the Banking77 dataset (with 77 classes)
- Approach: Two Training regimes in one script. LoRA, full fine-tune
- Why LoRA: Near-parity accuracy to full fine-tuning with small fraction of trainable parameters, lower compute time and resources, and faster iteration. 
- Result snapshot: (current runs, may not be reflected in demo): LoRA achieved 92.2% accuracy, full fine-tune 93.5%, LoRA trains only ~0.03% of pararmeters.
- Dev and MLOps tooling: Config driven Typer CLIs, Parquet ETL, HF trainer and jupyter notebook Colab walkthrough, MLflow tracking, artifact layout designed for adapter and merged models, small FastAPI backend and Gradio frontend for local serving (live demo on spaces uses slight variation)

---

## Motivation

Banking apps need to instantly interpret what users want (eg: "check my balance", "open a new card", or "dispute a transaction") with high accuracy and low latency. There are dozens of fine grained "intents" in production banking bots, and those categories change as banks evolve.

Classic large language models can solve this, but full fine-tuning of modern transformers (BERT, RoBERTa, DeBERTa, etc) is costly, especially for teams without dedicated ML infra or large budgets for GPUs.  
---

## Why Won’t a Pretrained get the job done? Zero-shot on Banking77 

- Label–verbalizer mismatch: NLI style zero-shot treats labels as hypotheses. Many Banking77 names aren’t natural paraphrases of user text(eg: people don't complain about their "lost or stolen card"), so entailment signals are weak/misaligned.

- Fine-grained overlap: Banking77 has 77 intents classese with semantically near neighbors (eg: delivery vs arrival vs replacement). This pushes zero-shot inference toward superficial cues and frequent, generic labels.

- Short, pragmatic utterances: Tickets are brief ("why still no card"  or "limits?") and are context light. Disambiguation requires domain knowledge, not just pretraining priors.

- Domain drift: Fintech terms, BINs, policy lingo, and merchant patterns are updated frequently but also underrepresented in NLI style pretraining.  Embeddings can't separate banking's (or any similar domain's) specific classes if their pretraining has never encountered them.

- Prompt sensitivity, no adaptation: Tiny prompt edits swing predictions and there’s no gradient to align the encoder to your 77-way ontology. In practice I saw ~14–35% zero-shot accuracy using an NLI model (facebook/bart-large-mnli). Using the smaller, general purpose deberta-v3-base, zero-shot inference was unusable at ~6%. However, the same small model, after having a minority of its parameters trained through LoRA, yielded >92% once adapted far surpassing even the NLI-task specialized large model.

---

## Enter the LoRA

### What is LoRA?
**LoRA (Low-Rank Adaptation)** of Large Language Models is a method to make large model adaptation cheap and modular. Instead of updating all weights, it injects small trainable adapters into key layers (like attention projections). These adapters learn task-specific tweaks while the base model stays frozen.

#### One Minute of Math

LoRA freezes the pretrained weights W and learns a low-rank<sup>*</sup> update ∆W = BA where A where A ∈ ℝ^{r x k}, B ∈ ℝ^{d x r}, and r << min(d,k). During inference, the effective weight is:

W' = W + BA

<sup>*</sup> Low ranking refers to the rank of the given matrix; essentially the entire adaptation lives in a small r-dimensional subspace (think low dim plane) instead of a full d x k parameter space.  

### Benefits of LoRA ###
- Freezes big weights, only training the small A and B matrices (these are the adapters)

- Target specific layers by attaching adapters to specific attention projections as well as the output projection via config.

- Easily swap in/out adapters for different tasks on the same model.

- After training you can merge the adapter into the base via update W <- W + (alpha/rank) * BA. Producs a standard HF model with no PEFT dependency and therefore 

### Why Not Just Full or Head-Only Fine-Tuning?
- Full fine-tuning: Highest accuracy, but compute/storage corresponding to full model size.
- Head-only: Fastest and least parameters, but often underfits, producing lower accuracy.
- LoRA: Almost all the quality, with far less of the cost. The frozen backbone and inductiv3e bias therein encode most of the langauge knowledge of the pretrained LLM. The adapter simply nudges the weights toward the banking intents. The intents are relatively low in complexity with short expected inputs, so this produces enough flow through the model to successfully adapt it to the specific task. 

---

## Design Overview

### Three training regimes in one file (`multiregime.py`)

* **head‑only:** Train just the classifier and pooler. Underfits.
* **lora:** Wrap the base model with `get_peft_model()` using a `LoraConfig` and keep the classifier trainable via `modules_to_save=["classifier"]`.
* **full‑ft:** Unfreeze everything, fine tune entire model on training set.

Each regime shares the same tokenization, batching, Trainer, metrics, and MLflow logging. That makes comparisons fair.

Note: `baseline.py` evaluates the pretrained model's zero shot benchmark on the dataset. Included separate from multiregime as results are not much better than chance ie: ~1/77.  

### Consistent and simple ETL (`etl.py`)

* Pulls `banking77` from HF Datasets.
* Stratified split into train/val/test (default 80/10/10) to keep label priors consistent. Seedable for reproducability in experiements.
* Saves to Parquet and adds a stable `label_text` column so labels don’t get silently re‑indexed later.

### Evaluation that matches training (`eval.py`)

* Loads the merged model if present (simplest case, least configuring), else base_77 + adapter, else plain checkpoint. (See experiment tracking below for more details)
* Computes Accuracy** and Macro‑F1, prints a clean summary, logs a normalized confusion matrix to MLflow.

### Simple, but Production-tooled Serving solution (`backend.py`)

* Small FastAPI app exposing `/predict/` that returns the top label.
* Uses `MODEL_DIR` env var; expects either a merged LoRA model or any HF seq‑cls model with `labels.json`.

### Experiment tracking

* MLflow tags: `dataset`, `backbone`, `regime`.
* Logs flattened LoRA params (e.g., `lora.r`, `lora.alpha`, etc), trainable-vs-total params, metrics, and artifacts.
* LoRA exports three artifacts for flexible downstream use:

  * `adapter/` — PEFT adapter only
  * `base_77/` — base model and tokenizer aligned to the 77 labels
  * `merged/` — standalone model with LoRA baked in (recommended for eval/serve)

---

## Engineering Decisions

* Single, multiple regime script. Easier to keep data, metrics, and Trainer behavior aligned for fair comparisons.
* Stratified splits  Parquet. Avoids label skew and makes downstream loading fast and schema‑stable.
* Label text baked into artifacts. `labels.json` freezes `id2label` so serving/eval can’t silently drift.
* LoRA export strategy. I log adapter, base_77, and merged so you can either reuse the adapter elsewhere or deploy a single merged model with zero PEFT runtime.
* MLflow for resilient truth. Every run has tags, params, metrics, plots, and the exact artifacts you need for eval or rollback.
* Small, straightforward API. Easy to smoke‑test models and wire into a UI.

---

## Quickstart

### **Please see notebooks/walkthrough.ipynb for colab training, tracking, and eval.**

```bash
# 1) Create env & install deps 
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt  

# 2) Download and  the data
python -m etl --cfg-path configs/data.yaml

# 3) Train (choose a regime)
python -m multiregime --train-config configs/train.yaml --regime lora
# or
python -m multiregime --train-config configs/train.yaml --regime head-only
# or
python -m multiregime --train-config configs/train.yaml --regime full-ft

# 4) Evaluate (prefers merged model if available)
python -m eval --checkpoint_path artifacts/models/your_run

# 5) Serve
export MODEL_DIR=artifacts/models/your_run/merged
uvicorn backend:app --reload
```

Note: The Trainer auto‑enables `fp16` if CUDA is available. Set `epochs`, `lr`, and `batch_size` in `configs/train.yaml`.

---

## Application: Where/how to LoRA

In `multiregime.py` the LoRA config is passed like so:

```python
lora_cfg = LoraConfig(
    r=cfg.lora.r,
    lora_alpha=cfg.lora.alpha,
    lora_dropout=cfg.lora.dropout,
    target_modules=cfg.lora.target_modules,
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["classifier"],`
)
model = get_peft_model(base, lora_cfg)
```

* `target_modules` is architecture‑specific. For many HF backbones, names like `q_proj`, `v_proj`, `query`, `value`, or `dense` are common. Keep it in config so you can switch backbones without touching code.
* `modules_to_save` ensures non‑LoRA layers you still want to train (here the classifier head) are persisted alongside the adapter.
* After training `peft_model.merge_and_unload()` writes `merged/` as a vanilla HF model for ease of portability.

---

## Metrics

* **Accuracy** is intuitive but can hide per‑class failures when classes are imbalanced or fine‑grained.
* **Macro‑F1** gives each class equal weight. Balances precision and recall -per class- so rare classes are not oversahdowed by frequent classes.

`compute_metrics()` returns both, and MLflow tracks them over epochs.

---

## Results (personal run results)

From comparable runs with the same backbone and training budget:

| Regime    | Trainable Params | Accuracy    | Macro‑F1            | Notes                                 |
| --------- | ---------------- | ----------- | ------------------- | ------------------------------------- |
| Head‑only | ~0.001%             | ~15.5%  | ~12.5%            | Fastest, underfits good as a smoke test         |
| **LoRA**  | **~0.003%**,         | **~92.2%** |     ~91%   | Near‑parity with a fraction of params of full fine tune |
| Full FT   | 100%             | ~93.8%     | ~93.1% | Most compute‑intensive                |


Exact metrics depend on backbone, `r/alpha`, epochs, sequence length, and other hyperparameters. However, the efficiency‑to‑accuracy trade‑off holds.

<sub>Note: Baseline zero-shot inference also included (`baseline.py`) for reference. Using the frozen NLI model facebook/bart-large-mnli, achieved peak of 35% acc.</sub>

---

## Config Examples

`data.yaml`

```yaml
dataset_name: "banking77"
test_split: 0.10
val_split: 0.10
seed: 6
output_dir: "data/processed"
```

`train.yaml` (~92% acc hyperparameters)

```yaml
model_name: "microsoft/deberta-v3-base"
batch_size: 64
lr: 1e-3
epochs: 10
weight_decay: 0.01
lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["query_proj","key_proj","value_proj","output_proj"] 
output_dir: "artifacts/models/deberta_lora"
```

Note:  Tweak `r/alpha/dropout` for your GPU and dataset. Consider using scheduling or other hyperparameters via the TrainingArguments class in `multiregime.py`. Increase `max_length` in `multiregime.py` if your inputs are getting truncated.

---

## Troubleshooting

* **"Found adapter but no base_77 directory."** Use the **merged** model for eval/serve, or re‑attach the adapter to the exact `base_77/` it was trained with.
* **Classifier missing after LoRA load.** Ensure `modules_to_save=["classifier"]` is in the LoRA config.
* **Macro‑F1 is low but accuracy is fine.** Check class balance and confusion matrix; you may be under‑serving rare intents.
* **GPU detected but slow.** Ensure batch size is reasonable (eg: in colab, with a T4 on default memory, batch size 64 is fine); `fp16` auto‑enabled if CUDA available.
* **Label drift between scripts.** This repo writes and reads the same `labels.json`. If you hand‑edit artifacts, keep `id2label/label2id` consistent.

---

## Extending This Work

* **Swap backbones** by editing `model_name` in `train.yaml` and updating `target_modules`.
* **Hard Negative Contrast Training** Mine contrast set for hard negative pairs (eg: "I lost my debit card" vs. "I need a new debit PIN"), fine-tune prior to LoRA.
* **UI**: Swap out Gradio front end for more full-featured React htmx for a more webdev-ready, 'python-free' UI. 

---

## Citation & Thanks

* LoRA: Hu et al., *LoRA: Low‑Rank Adaptation of Large Language Models*, 2021.
* Hugging Face Transformers, Datasets, PEFT, scikit‑learn, MLflow, FastAPI — thanks for all the shoulders, giants. 

---

## License

MIT
