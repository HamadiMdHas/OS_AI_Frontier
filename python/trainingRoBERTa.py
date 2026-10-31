import os
import re
import random
import numpy as np
import pandas as pd
import torch

from langdetect import detect_langs, DetectorFactory, LangDetectException

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
)
from scipy.stats import spearmanr

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


DATA_PATH = os.path.join("data", "dataor.xlsx")

OUTPUT_DIR = os.path.join("outputs", "roberta_out_version3")

CLEANED_PATH = os.path.join("outputs", "cleaned_version3.xlsx")

TEXT_COL = "content_cleaned"
RATING_COL = "score"

MODEL_NAME = "roberta-base"

SEED = 42
MAX_LEN = 128
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 1

NON_ENGLISH_CONFIDENCE = 0.90

MIN_WORDS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

DetectorFactory.seed = SEED
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    DEVICE_DESCRIPTION = "CUDA GPU"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE_DESCRIPTION = "Apple Silicon MPS"
else:
    DEVICE_DESCRIPTION = "CPU"

print(f"Device available: {DEVICE_DESCRIPTION}")


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Run this script from the project "
        "root folder."
    )

df = pd.read_excel(DATA_PATH, sheet_name=0)

df["review_id"] = df.index

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)
print(f"Rows read from Excel: {len(df):,}")

df["text"] = df[TEXT_COL].fillna("").astype(str).str.strip()
df["score"] = pd.to_numeric(df[RATING_COL], errors="coerce")

before = len(df)
df = df[df["text"].ne("")].copy()
print(f"Rows removed for missing/empty text: {before - len(df):,}")

before = len(df)
valid_rating = (
    df["score"].notna()
    & df["score"].between(1, 5)
    & np.isclose(df["score"], np.round(df["score"]))
)
df = df[valid_rating].copy()
df["score"] = df["score"].astype(int)

print(f"Rows removed for invalid/missing ratings: {before - len(df):,}")

before = len(df)
df["word_count"] = df["text"].str.split().str.len()
df = df[df["word_count"] >= MIN_WORDS].copy()

print(f"Rows removed for fewer than {MIN_WORDS} words: {before - len(df):,}")


def normalize_review(text):
    text = str(text).casefold().strip()
    return re.sub(r"\s+", " ", text)

before = len(df)
df["normalized_text"] = df["text"].apply(normalize_review)
df = df.drop_duplicates(subset="normalized_text", keep="first").copy()

print(f"Duplicate reviews removed: {before - len(df):,}")


def classify_language(text):
    try:
        candidates = detect_langs(str(text))

        if not candidates:
            return "unknown", "uncertain"

        best = candidates[0]
        language_code = best.lang
        confidence = float(best.prob)

        if language_code == "en":
            return language_code, "english"

        if confidence >= NON_ENGLISH_CONFIDENCE:
            return language_code, "non_english"

        return language_code, "uncertain"

    except (LangDetectException, Exception):
        return "unknown", "uncertain"


print("\nDetecting review languages...")

language_results = df["text"].apply(classify_language)

df["detected_language"] = language_results.apply(lambda x: x[0])
df["language_status"] = language_results.apply(lambda x: x[1])

print("\nLanguage detection results:")
print(
    df["language_status"]
    .value_counts()
    .rename_axis("language_status")
    .to_string()
)

df.to_csv(
    os.path.join(OUTPUT_DIR, "language_audit_before_filtering.csv"),
    index=False,
)

before = len(df)
df = df[df["language_status"] != "non_english"].copy()

print(f"\nConfidently non-English reviews removed: {before - len(df):,}")
print(f"Final observations retained: {len(df):,}")

if len(df) == 0:
    raise ValueError("No observations remain after cleaning.")

df.to_excel(CLEANED_PATH, index=False)

print(f"Cleaned workbook saved to: {CLEANED_PATH}")

print("\nFinal rating distribution:")
print(df["score"].value_counts().sort_index().to_string())

print("\nFinal rating percentages:")
print(
    (df["score"].value_counts(normalize=True).sort_index() * 100)
    .round(2)
    .astype(str)
    .add("%")
    .to_string()
)


train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["score"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["score"],
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

print("\n" + "=" * 70)
print("SHARED DATA SPLIT")
print("=" * 70)

print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(val_df):,}")
print(f"Test:       {len(test_df):,}")

split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

split_df[
    ["review_id", "text", "score", "split", "detected_language",
     "language_status"]
].to_csv(
    os.path.join(OUTPUT_DIR, "shared_split_membership.csv"),
    index=False,
)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN,
    )


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


class WeightedTrainer(Trainer):

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float)
            if class_weights is not None
            else None
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.logits

        weights = None
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)

        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def calculate_metrics(eval_prediction, class_ratings):
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    labels = eval_prediction.label_ids

    logits = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum(
        axis=1, keepdims=True
    )

    ratings = np.asarray(class_ratings, dtype=float)
    expected_ratings = probabilities @ ratings
    predicted_ratings = ratings[np.argmax(probabilities, axis=1)]
    actual_ratings = ratings[labels]

    return {
        "mae": float(mean_absolute_error(actual_ratings, expected_ratings)),
        "rmse": float(
            np.sqrt(mean_squared_error(actual_ratings, expected_ratings))
        ),
        "exact_accuracy": float(
            accuracy_score(actual_ratings, predicted_ratings)
        ),
    }


def make_dataset(frame, rating_to_class):
    frame = frame.copy()
    frame["label"] = frame["score"].map(rating_to_class)

    frame = frame[frame["label"].notna()].copy()
    frame["label"] = frame["label"].astype(int)

    if frame.empty:
        return None

    dataset = Dataset.from_pandas(
        frame[["text", "label"]],
        preserve_index=False,
    )

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
    )

    return dataset


def run_model(
    model_key,
    model_title,
    class_ratings,
    rating_to_class,
    train_source,
    val_source,
    test_source,
):
    print("\n\n" + "#" * 78)
    print(f"TRAINING: {model_title}")
    print("#" * 78)

    model_dir = os.path.join(OUTPUT_DIR, model_key)
    os.makedirs(model_dir, exist_ok=True)

    train_part = train_source[
        train_source["score"].isin(rating_to_class.keys())
    ].copy()

    val_part = val_source[
        val_source["score"].isin(rating_to_class.keys())
    ].copy()

    test_part = test_source[
        test_source["score"].isin(rating_to_class.keys())
    ].copy()

    print(f"Training observations:   {len(train_part):,}")
    print(f"Validation observations: {len(val_part):,}")
    print(f"Test observations:       {len(test_part):,}")

    if train_part.empty or val_part.empty or test_part.empty:
        raise ValueError(
            f"{model_title}: one of the train/validation/test sets is empty."
        )

    n_classes = len(class_ratings)

    observed_classes = set(train_part["score"].map(rating_to_class))
    expected_classes = set(range(n_classes))

    if observed_classes != expected_classes:
        raise ValueError(
            f"{model_title}: missing training classes. "
            f"Expected {expected_classes}, observed {observed_classes}"
        )

    train_counts = (
        train_part["score"]
        .map(rating_to_class)
        .value_counts()
        .reindex(range(n_classes), fill_value=0)
    )

    class_weights = (
        len(train_part) / (n_classes * train_counts.to_numpy())
    )
    class_weights = class_weights / class_weights.mean()

    train_ds = make_dataset(train_part, rating_to_class)
    val_ds = make_dataset(val_part, rating_to_class)
    test_ds = make_dataset(test_part, rating_to_class)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
    )

    training_args = TrainingArguments(
        output_dir=model_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_mae",
        greater_is_better=False,
        save_total_limit=1,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        fp16=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        class_weights=class_weights,
        compute_metrics=lambda p: calculate_metrics(
            p, class_ratings
        ),
        callbacks=[],
    )

    from transformers import EarlyStoppingCallback

    trainer.add_callback(
        EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )
    )

    trainer.train()

    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    test_metrics = trainer.evaluate(
        eval_dataset=test_ds,
        metric_key_prefix="test",
    )

    print("\nTest metrics:")
    for key, value in test_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")

    prediction_parts = []

    for split_name, frame, dataset in [
        ("validation", val_part, val_ds),
        ("test", test_part, test_ds),
    ]:
        output = trainer.predict(dataset)
        logits = output.predictions

        if isinstance(logits, tuple):
            logits = logits[0]

        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)

        ratings = np.asarray(class_ratings, dtype=float)
        predicted_class = np.argmax(probs, axis=1)
        predicted_rating = ratings[predicted_class]
        expected_rating = probs @ ratings

        result = frame[
            ["review_id", "text", "score"]
        ].copy().reset_index(drop=True)

        result["split"] = split_name
        result["predicted_rating"] = predicted_rating
        result["expected_rating"] = expected_rating

        for class_index, rating in enumerate(class_ratings):
            result[f"p_{rating}star"] = probs[:, class_index]

        prediction_parts.append(result)

        if split_name == "test":
            actual = frame["score"].to_numpy()
            predicted = predicted_rating

            cm = confusion_matrix(
                actual,
                predicted,
                labels=list(class_ratings),
            )

            cm_df = pd.DataFrame(
                cm,
                index=[f"actual_{r}star" for r in class_ratings],
                columns=[f"predicted_{r}star" for r in class_ratings],
            )

            cm_df.to_csv(
                os.path.join(model_dir, "test_confusion_matrix.csv")
            )

            rho = spearmanr(actual, expected_rating).statistic

            metrics = {
                "model": model_title,
                "n_train": len(train_part),
                "n_validation": len(val_part),
                "n_test": len(test_part),
                "mae_expected_rating": mean_absolute_error(
                    actual, expected_rating
                ),
                "rmse_expected_rating": np.sqrt(
                    mean_squared_error(actual, expected_rating)
                ),
                "spearman": rho,
                "exact_rating_accuracy": accuracy_score(
                    actual, predicted
                ),
            }

            pd.DataFrame([metrics]).to_csv(
                os.path.join(model_dir, "test_metrics.csv"),
                index=False,
            )

            print("\nTest-set confusion matrix:")
            print(cm_df.to_string())
            print("\nTest-set summary:")
            for key, value in metrics.items():
                print(
                    f"{key}: {value:.4f}"
                    if isinstance(value, (float, np.floating))
                    else f"{key}: {value}"
                )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_csv(
        os.path.join(model_dir, "validation_test_predictions.csv"),
        index=False,
    )

    print(f"\nSaved model and outputs to: {model_dir}")

    del trainer, model, train_ds, val_ds, test_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    return metrics


main_metrics = run_model(
    model_key="model_1_main_5class",
    model_title="Main model: 1-5 stars",
    class_ratings=[1, 2, 3, 4, 5],
    rating_to_class={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    train_source=train_df,
    val_source=val_df,
    test_source=test_df,
)

main_extreme_metrics = run_model(
    model_key="model_2_extremes_1_vs_5",
    model_title="Specialized extremes model: 1 vs 5 stars",
    class_ratings=[1, 5],
    rating_to_class={1: 0, 5: 1},
    train_source=train_df,
    val_source=val_df,
    test_source=test_df,
)

middle_metrics = run_model(
    model_key="model_3_middle_2_to_4",
    model_title="Specialized middle model: 2-4 stars",
    class_ratings=[2, 3, 4],
    rating_to_class={2: 0, 3: 1, 4: 2},
    train_source=train_df,
    val_source=val_df,
    test_source=test_df,
)


summary = pd.DataFrame(
    [main_metrics, main_extreme_metrics, middle_metrics]
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "three_model_test_metrics_summary.csv",
)

summary.to_csv(summary_path, index=False)

print("\n" + "=" * 78)
print("ALL THREE MODELS COMPLETED")
print("=" * 78)
print(summary.to_string(index=False))
print(f"\nSummary saved to: {summary_path}")
print(f"All outputs are in: {OUTPUT_DIR}")