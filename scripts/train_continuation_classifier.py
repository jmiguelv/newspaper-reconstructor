"""Train or evaluate a binary BERT continuation classifier."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

SPLIT_NAMES = ("train", "validation", "test")


class PairDataset:
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_model(args)
    else:
        evaluate_model(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Fine-tune the classifier")
    train_parser.add_argument("--data-dir", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument(
        "--model", default="distilbert/distilbert-base-multilingual-cased"
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--max-length", type=int, default=512)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--train-batch-size", type=int, default=16)
    train_parser.add_argument("--eval-batch-size", type=int, default=16)
    train_parser.add_argument("--epochs", type=float, default=3)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)

    evaluate_parser = commands.add_parser("evaluate", help="Evaluate a saved model")
    evaluate_parser.add_argument("--data-dir", required=True)
    evaluate_parser.add_argument("--model-dir", required=True)
    evaluate_parser.add_argument(
        "--split", choices=("validation", "test"), required=True
    )
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def train_model(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    records = load_records(Path(args.data_dir), split_names=("train", "validation"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    datasets = {
        name: tokenize_records(split_records, tokenizer, args.max_length)
        for name, split_records in records.items()
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "NOT_CONTINUATION", 1: "CONTINUATION"},
        label2id={"NOT_CONTINUATION": 0, "CONTINUATION": 1},
        dtype=torch.float32,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        dataloader_pin_memory=not torch.backends.mps.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=trainer_metrics,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)

    evals = epoch_evals(trainer.state.log_history)
    best = max(evals, key=lambda entry: entry["eval_f1"])
    training = {
        "model": args.model,
        "transformers_version": transformers.__version__,
        "seed": args.seed,
        "hyperparameters": {
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
        },
        "examples": {name: len(split) for name, split in datasets.items()},
        "best_checkpoint": {
            "epoch": best["epoch"],
            "validation_f1": trainer.state.best_metric,
        },
        "epoch_history": evals,
        "best_validation_f1": trainer.state.best_metric,
        "train": train_result.metrics,
    }
    (output_dir / "training.json").write_text(
        json.dumps(training, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Best checkpoint: epoch {best['epoch']:g} of {args.epochs:g} "
        f"(validation F1 {trainer.state.best_metric:.4f})"
    )
    print(f"Saved model and training metadata to {output_dir}")


def evaluate_model(args: argparse.Namespace) -> None:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    model_dir = Path(args.model_dir)
    records = load_records(Path(args.data_dir), split_names=(args.split,))[args.split]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    dataset = tokenize_records(records, tokenizer, args.max_length)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_dir / "evaluation"),
            per_device_eval_batch_size=args.batch_size,
            report_to="none",
            dataloader_pin_memory=not torch.backends.mps.is_available(),
        ),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    metrics = {
        "split": args.split,
        "examples": len(dataset),
        **evaluate_dataset(trainer, dataset),
    }
    output_path = model_dir / f"{args.split}-metrics.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved metrics to {output_path}")


def load_records(
    data_dir: Path, split_names: Sequence[str] = SPLIT_NAMES
) -> dict[str, list[dict]]:
    records = {}
    required = {"page_id", "a_id", "b_id", "text_a", "text_b", "label"}
    for name in split_names:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}")
        path = data_dir / f"{name}.jsonl"
        with path.open(encoding="utf-8") as file:
            split_records = [json.loads(line) for line in file if line.strip()]
        if not split_records:
            raise ValueError(f"{name}.jsonl must not be empty")
        for record in split_records:
            if not required <= record.keys():
                raise ValueError(f"{name}.jsonl contains an incomplete record")
            if record["label"] not in (0, 1):
                raise ValueError(f"{name}.jsonl labels must be zero or one")
        records[name] = split_records
    return records


def tokenize_records(records: list[dict], tokenizer, max_length: int) -> PairDataset:
    encoded = tokenizer(
        [record["text_a"] for record in records],
        [record["text_b"] for record in records],
        truncation=True,
        max_length=max_length,
    )
    items = [
        {
            **{name: values[index] for name, values in encoded.items()},
            "labels": record["label"],
        }
        for index, record in enumerate(records)
    ]
    return PairDataset(items)


def epoch_evals(log_history: list[dict]) -> list[dict]:
    evals = [
        {
            "epoch": entry["epoch"],
            "eval_loss": entry["eval_loss"],
            "eval_f1": entry["eval_f1"],
            "eval_accuracy": entry["eval_accuracy"],
        }
        for entry in log_history
        if "eval_f1" in entry
    ]
    if not evals:
        raise ValueError("log history contains no evaluation entries")
    return evals


def positive_probabilities(logits) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def classification_metrics(
    probabilities: list[float] | np.ndarray, labels: list[int] | np.ndarray
) -> dict:
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def trainer_metrics(eval_prediction) -> dict:
    metrics = classification_metrics(
        positive_probabilities(eval_prediction.predictions), eval_prediction.label_ids
    )
    confusion = metrics.pop("confusion")
    return {**metrics, **confusion}


def evaluate_dataset(trainer, dataset: PairDataset) -> dict:
    prediction = trainer.predict(dataset)
    return classification_metrics(
        positive_probabilities(prediction.predictions), prediction.label_ids
    )


if __name__ == "__main__":
    main()
