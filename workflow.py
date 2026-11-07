"""Kleiner KI-Workflow: Supporttexte klassifizieren und Vorhersagen protokollieren."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
LABELS = ["Abrechnung", "Technik", "Vertrag"]
SEED = 42


def utc_now():
    """Zeitstempel einschließlich Zeitzone in UTC erstellen."""
    return datetime.now(timezone.utc).isoformat()


def load_data(path):
    """CSV einlesen; leere Texte, unbekannte Klassen und Duplikate zurückweisen."""
    texts, labels, seen = [], [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"text", "label"}:
            raise ValueError("Die CSV benötigt genau die Spalten text und label.")
        for line, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if None in row or not text or label not in LABELS:
                raise ValueError(f"Ungültiger Text oder ungültige Kategorie in Zeile {line}.")
            normalized = " ".join(text.casefold().split())
            if normalized in seen:
                raise ValueError(f"Doppelter Text in Zeile {line}.")
            seen.add(normalized)
            texts.append(text)
            labels.append(label)
    counts = Counter(labels)
    if any(counts[label] < 5 for label in LABELS):
        raise ValueError("Jede der drei Kategorien benötigt mindestens fünf Beispiele.")
    return texts, labels


def train(data_path=ROOT / "data/tickets.csv", output_dir=ROOT / "artifacts"):
    """Modell nur auf Trainingsdaten fitten; Bewertung auf separaten Testdaten."""
    data_path, output_dir = Path(data_path), Path(output_dir)
    texts, labels = load_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    # Die Pipeline lernt auch das Textvokabular ausschließlich aus x_train.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(
        y_test, predictions, labels=LABELS, output_dict=True, zero_division=0
    )

    # Vergleich mit einem Modell, das immer die häufigste Trainingsklasse wählt.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    model_version = f"v1-{data_hash[:12]}-sklearn{sklearn.__version__}"
    trained_at = utc_now()
    metrics = {
        "model_version": model_version,
        "trained_at_utc": trained_at,
        "data_sha256": data_hash,
        "seed": SEED,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "baseline_accuracy": float(accuracy_score(y_test, baseline.predict(x_test))),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=LABELS).tolist(),
        "classification_report": report,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "model_version": model_version}, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def predict(text, model_path=ROOT / "artifacts/model.joblib", log_path=ROOT / "logs/predictions.jsonl"):
    """Text klassifizieren und technische Metadaten ohne den Rohtext protokollieren."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Bitte einen nicht leeren Text angeben.")
    text = text.strip()
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError("Kein Modell vorhanden. Zuerst python workflow.py train ausführen.")

    started = perf_counter()
    # Nur selbst erzeugte oder vertrauenswürdige joblib-Dateien laden.
    bundle = joblib.load(model_path)
    category = str(bundle["model"].predict([text])[0])
    result = {
        "category": category,
        "model_version": bundle["model_version"],
        "timestamp_utc": utc_now(),
        "runtime_ms": round((perf_counter() - started) * 1000, 3),
        "text_length": len(text),
    }
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Modell trainieren und evaluieren")
    training.add_argument("--data", type=Path, default=ROOT / "data/tickets.csv")
    training.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    prediction = commands.add_parser("predict", help="Einen neuen Text klassifizieren")
    prediction.add_argument("--text", required=True)
    prediction.add_argument("--model", type=Path, default=ROOT / "artifacts/model.joblib")
    prediction.add_argument("--log-file", type=Path, default=ROOT / "logs/predictions.jsonl")
    args = parser.parse_args()
    try:
        if args.command == "train":
            result = train(args.data, args.output_dir)
        else:
            result = predict(args.text, args.model, args.log_file)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(2, f"Fehler: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
