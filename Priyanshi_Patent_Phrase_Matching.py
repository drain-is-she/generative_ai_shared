import os
import glob
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tokenizers.processors import TemplateProcessing
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

SEED = 42
MODEL_ID = "microsoft/deberta-v3-small"
MAX_LEN = 64
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 32
VAL_FRACTION = 0.2
OUT_DIR = "/kaggle/working"
RUN_EXPERIMENTS = True
RUN_BI_ENCODER = True
BI_ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"

transformers.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CPC_SECTIONS = {
    "A": "human necessities",
    "B": "performing operations; transporting",
    "C": "chemistry; metallurgy",
    "D": "textiles; paper",
    "E": "fixed constructions",
    "F": "mechanical engineering; lighting; heating; weapons; blasting",
    "G": "physics",
    "H": "electricity",
}


def find_one(pattern, root="/kaggle/input"):
    hits = glob.glob(f"{root}/**/{pattern}", recursive=True)
    return os.path.dirname(hits[0]) if hits else None


def load_data():
    data_dir = find_one("us-patent-phrase-to-phrase-matching/train.csv") or find_one("train.csv")
    if not data_dir:
        raise FileNotFoundError("Attach the U.S. Patent Phrase to Phrase Matching competition data in Kaggle.")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test


def add_inputs(df, context_mode="section"):
    df = df.copy()
    df["text_a"] = df["anchor"].fillna("").astype(str)
    if context_mode == "section":
        context = df["context"].fillna("").astype(str).str[0].map(CPC_SECTIONS).fillna("unknown technical field")
        df["text_b"] = df["target"].fillna("").astype(str) + "; " + context
    elif context_mode == "raw":
        df["text_b"] = df["target"].fillna("").astype(str) + "; CPC " + df["context"].fillna("").astype(str)
    elif context_mode == "none":
        df["text_b"] = df["target"].fillna("").astype(str)
    else:
        raise ValueError(context_mode)
    return df


def make_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if getattr(tokenizer, "backend_tokenizer", None) is not None:
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        if cls_id is not None and sep_id is not None:
            tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
            )
    return tokenizer


def to_dataset(df, tokenizer, with_labels=True):
    cols = ["text_a", "text_b"] + (["score"] if with_labels else [])
    frame = df[cols].copy()
    if with_labels:
        frame["score"] = frame["score"].astype("float32")
    ds = Dataset.from_pandas(frame, preserve_index=False)
    ds = ds.map(
        lambda batch: tokenizer(batch["text_a"], batch["text_b"], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=["text_a", "text_b"],
    )
    if with_labels:
        ds = ds.rename_column("score", "labels")
    return ds


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if np.std(predictions) == 0 or np.std(labels) == 0:
        return {"pearson": 0.0}
    return {"pearson": float(np.corrcoef(predictions, labels)[0, 1])}


def train_cross_encoder(train_frame, val_frame, test_frame, model_path, context_mode="section", learning_rate=LR, run_name="baseline"):
    start = time.time()
    trn = add_inputs(train_frame, context_mode)
    val = add_inputs(val_frame, context_mode)
    tst = add_inputs(test_frame, context_mode)
    tokenizer = make_tokenizer(model_path)
    ds_train = to_dataset(trn, tokenizer, True)
    ds_val = to_dataset(val, tokenizer, True)
    ds_test = to_dataset(tst, tokenizer, False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1, problem_type="regression", dtype=torch.float32
    )
    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, f"trainer_{run_name}"),
        num_train_epochs=EPOCHS,
        learning_rate=learning_rate,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=2 * BATCH_SIZE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    val_pred = trainer.predict(ds_val).predictions.reshape(-1)
    test_pred = trainer.predict(ds_test).predictions.reshape(-1)
    elapsed = time.time() - start
    score = float(np.corrcoef(val_pred, val["score"].to_numpy())[0, 1])
    val_result = val_frame.copy()
    val_result["pred"] = val_pred
    return {
        "model": model,
        "tokenizer": tokenizer,
        "trainer": trainer,
        "val_predictions": val_result,
        "test_predictions": np.clip(test_pred, 0, 1),
        "pearson": score,
        "seconds": elapsed,
    }


def run_biencoder(train_frame, val_frame, model_id=BI_ENCODER_ID):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, device=DEVICE)
    anchors = val_frame["anchor"].fillna("").astype(str).tolist()
    targets = val_frame["target"].fillna("").astype(str).tolist()
    anchor_embeddings = model.encode(anchors, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    target_embeddings = model.encode(targets, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    predictions = np.sum(anchor_embeddings * target_embeddings, axis=1)
    labels = val_frame["score"].to_numpy()
    return float(np.corrcoef(predictions, labels)[0, 1]), predictions


def run_error_analysis(val_predictions):
    df = val_predictions.copy()
    df["abs_error"] = np.abs(df["score"] - df["pred"])
    df["section"] = df["context"].fillna("").astype(str).str[0]
    section_rows = []
    for section, group in df.groupby("section"):
        corr = group["score"].corr(group["pred"]) if len(group) > 1 else np.nan
        section_rows.append({"section": section, "rows": len(group), "pearson": corr})
    section_metrics = pd.DataFrame(section_rows).sort_values("pearson", na_position="first")
    score_means = df.groupby("score", as_index=False)["pred"].mean().rename(columns={"pred": "mean_prediction"})
    worst = df.nlargest(10, "abs_error")[["anchor", "target", "context", "score", "pred", "abs_error"]]
    section_metrics.to_csv(os.path.join(OUT_DIR, "section_metrics.csv"), index=False)
    score_means.to_csv(os.path.join(OUT_DIR, "score_calibration.csv"), index=False)
    worst.to_csv(os.path.join(OUT_DIR, "worst_validation_errors.csv"), index=False)
    return section_metrics, score_means, worst


def main():
    print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEVICE)
    train, test = load_data()
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    train_idx, val_idx = next(splitter.split(train, groups=train["anchor"]))
    trn = train.iloc[train_idx].reset_index(drop=True)
    val = train.iloc[val_idx].reset_index(drop=True)
    if not set(trn["anchor"]).isdisjoint(set(val["anchor"])):
        raise RuntimeError("Anchor leakage detected in validation split.")
    print("train rows:", len(trn), "| validation rows:", len(val), "| validation anchors:", val["anchor"].nunique())

    model_path = find_one(f"{MODEL_ID.split('/')[-1]}/config.json") or MODEL_ID
    if model_path == MODEL_ID:
        print("Checkpoint not found in Kaggle inputs; this run requires internet or an attached checkpoint.")

    results = []
    baseline = train_cross_encoder(trn, val, test, model_path, "section", LR, "baseline")
    results.append({
        "setting": "fine-tuned cross-encoder, section description, lr=2e-5",
        "validation_pearson": baseline["pearson"],
        "training_seconds": baseline["seconds"],
    })
    baseline["val_predictions"].to_csv(os.path.join(OUT_DIR, "val_predictions.csv"), index=False)

    if RUN_BI_ENCODER:
        try:
            biencoder_score, _ = run_biencoder(trn, val)
            results.append({
                "setting": "pretrained bi-encoder, no fine-tuning",
                "validation_pearson": biencoder_score,
                "training_seconds": 0.0,
            })
            print("Bi-encoder validation Pearson:", round(biencoder_score, 4))
        except Exception as exc:
            print("Bi-encoder baseline skipped:", str(exc))

    if RUN_EXPERIMENTS:
        no_context = train_cross_encoder(trn, val, test, model_path, "none", LR, "no_context")
        results.append({
            "setting": "no context",
            "validation_pearson": no_context["pearson"],
            "training_seconds": no_context["seconds"],
        })
        lr_variant = train_cross_encoder(trn, val, test, model_path, "section", 1e-5, "lr_1e5")
        results.append({
            "setting": "learning rate=1e-5",
            "validation_pearson": lr_variant["pearson"],
            "training_seconds": lr_variant["seconds"],
        })

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(os.path.join(OUT_DIR, "experiment_results.csv"), index=False)
    print(results_frame.to_string(index=False))

    sections, score_means, worst = run_error_analysis(baseline["val_predictions"])
    print("Section metrics")
    print(sections.to_string(index=False))
    print("Mean prediction by true score")
    print(score_means.to_string(index=False))
    print("Largest validation errors")
    print(worst.to_string(index=False))

    submission = pd.DataFrame({
        "id": test["id"],
        "score": baseline["test_predictions"],
    })
    if submission["score"].isna().any() or len(submission) != len(test):
        raise RuntimeError("Submission contains missing predictions or the wrong number of rows.")
    submission.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
    print("Wrote", os.path.join(OUT_DIR, "submission.csv"))


if __name__ == "__main__":
    main()
