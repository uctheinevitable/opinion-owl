"""Run the three trained models on (text, aspect) rows.

Used by notebook 04 (labeled test rows), notebook 06 (real comments) and the app, so all of
them predict in exactly the same way. Every predict_* function takes a DataFrame with the
columns "text" and "aspect_term" and returns a DataFrame with the same index and the
columns predicted_polarity, confidence, all_class_scores and model.

Each function also accepts an optional `progress` object (anything with a .progress(fraction)
method, such as a Streamlit progress bar) and updates it as batches finish.
"""
import json
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from opinion_owl.atae import AtaePredictor
from opinion_owl.config import ATAE_DIR, DEBERTA_DIR, DEBERTA_MODEL_ID, TFIDF_DIR

TFIDF_VECTORIZER_PATH = TFIDF_DIR / "tfidf_vectorizer.joblib"
TFIDF_CLASSIFIER_PATH = TFIDF_DIR / "tfidf_classifier.joblib"
TFIDF_INFO_PATH = TFIDF_DIR / "tfidf_model_info.joblib"
ATAE_MODEL_PATH = ATAE_DIR / "atae_lstm_best.pt"
DEBERTA_CONFIG_PATH = DEBERTA_DIR / "config.json"
DEBERTA_WEIGHTS_PATH = DEBERTA_DIR / "model.safetensors"

ATAE_BATCH_SIZE = 64
DEBERTA_MAX_LENGTH = 128   # same as training
DEBERTA_BATCH_SIZE = 8  # lower peak memory on free CPU hosting


def _clean_inputs(frame):
    texts = frame["text"].astype(str).str.strip().tolist()
    # Lowercase to match how aspect names looked during training.
    aspects = frame["aspect_term"].astype(str).str.strip().str.lower().tolist()
    return texts, aspects


def _result_frame(frame, labels, confidences, score_dicts, model_name):
    return pd.DataFrame(
        {
            "predicted_polarity": labels,
            "confidence": np.round(np.asarray(confidences, dtype=float), 4),
            "all_class_scores": score_dicts,
            "model": model_name,
        },
        index=frame.index,
    )


def predict_tfidf(frame, progress=None):
    """TF-IDF + linear classifier. Input is "aspect [SEP] sentence", as in training (notebook 01)."""
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    classifier = joblib.load(TFIDF_CLASSIFIER_PATH)
    model_name = "tfidf_classifier"
    if TFIDF_INFO_PATH.exists():
        model_name = joblib.load(TFIDF_INFO_PATH).get("model_name", model_name)

    texts, aspects = _clean_inputs(frame)
    features = vectorizer.transform([a + " [SEP] " + t for a, t in zip(aspects, texts)])
    labels = classifier.predict(features)

    # LogisticRegression gives real probabilities. LinearSVC only gives decision scores, which are
    # turned into 0-1 values with a softmax: fine for ranking rows by how sure the model is,
    # but not calibrated probabilities.
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(features)
    else:
        raw = classifier.decision_function(features)
        if raw.ndim == 1:   # two-class LinearSVC returns one score per row
            raw = np.column_stack([-raw, raw])
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        scores = exp / exp.sum(axis=1, keepdims=True)

    score_dicts = [
        {label: round(float(p), 4) for label, p in zip(classifier.classes_, row)} for row in scores
    ]
    if progress:
        progress.progress(1.0)
    return _result_frame(frame, labels, scores.max(axis=1), score_dicts, model_name)


def predict_atae(frame, progress=None):
    """Aspect-aware attention BiLSTM. Model and checkpoint loading live in atae.py."""
    predictor = AtaePredictor(ATAE_MODEL_PATH)
    texts, aspects = _clean_inputs(frame)

    results = []
    for start in range(0, len(texts), ATAE_BATCH_SIZE):
        end = start + ATAE_BATCH_SIZE
        results.extend(predictor.predict_batch(texts[start:end], aspects[start:end]))
        if progress:
            progress.progress(min(1.0, end / len(texts)))
    if progress:
        progress.progress(1.0)

    return _result_frame(
        frame,
        [r["predicted_polarity"] for r in results],
        [r["confidence"] for r in results],
        [r["all_class_scores"] for r in results],
        "atae_lstm",
    )


@lru_cache(maxsize=1)
def _load_deberta():
    """Load DeBERTa once, using local weights if present or the Hugging Face Hub otherwise.

    The published Hub repository has tokenizer.json but not the slow SentencePiece
    vocabulary file, so `use_fast=True` is required for the hosted model.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local_model_ready = DEBERTA_CONFIG_PATH.is_file() and DEBERTA_WEIGHTS_PATH.is_file()
    model_source = str(DEBERTA_DIR) if local_model_ready else DEBERTA_MODEL_ID
    if not model_source:
        raise RuntimeError(
            "No DeBERTa model source configured. Set OPINION_OWL_DEBERTA_MODEL_ID "
            "or place a complete model under models/deberta_v3_absa."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_source)
        model.to(device)
        model.eval()

        if local_model_ready and (DEBERTA_DIR / "label_map.json").is_file():
            label_path = DEBERTA_DIR / "label_map.json"
        else:
            from huggingface_hub import hf_hub_download
            label_path = Path(hf_hub_download(
                repo_id=model_source,
                filename="label_map.json",
            ))

        label_data = json.loads(Path(label_path).read_text(encoding="utf-8"))
        id_to_label = {
            int(k): str(v).lower()
            for k, v in label_data["id_to_label"].items()
        }

        expected_ids = set(range(int(model.config.num_labels)))
        if set(id_to_label) != expected_ids:
            raise ValueError(
                f"Label map IDs {sorted(id_to_label)} do not match model output IDs "
                f"{sorted(expected_ids)}."
            )
        return tokenizer, model, id_to_label, device

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DeBERTa from {model_source!r}. Check that the model repository "
            "is public, its files are complete, and torch/transformers/huggingface_hub "
            f"are installed. Original error: {exc}"
        ) from exc


def predict_deberta(frame, progress=None):
    """Fine-tuned DeBERTa-v3 reading the pair "aspect [SEP] sentence"."""
    import torch

    tokenizer, model, id_to_label, device = _load_deberta()
    texts, aspects = _clean_inputs(frame)

    # Sorting by length keeps each batch's padding small, which is much faster on CPU.
    order = np.argsort([len(t) for t in texts], kind="stable")
    probabilities = np.zeros((len(texts), len(id_to_label)), dtype=np.float32)

    for start in range(0, len(order), DEBERTA_BATCH_SIZE):
        idx = order[start:start + DEBERTA_BATCH_SIZE]
        encoded = tokenizer(
            [aspects[i] for i in idx],
            [texts[i] for i in idx],
            truncation="only_second",
            max_length=DEBERTA_MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            probabilities[idx] = torch.softmax(model(**encoded).logits, dim=1).cpu().numpy()
        if progress:
            progress.progress(min(1.0, (start + DEBERTA_BATCH_SIZE) / len(order)))

    score_dicts = [
        {id_to_label[i]: round(float(p), 4) for i, p in enumerate(row)} for row in probabilities
    ]
    labels = [id_to_label[i] for i in probabilities.argmax(axis=1)]
    return _result_frame(frame, labels, probabilities.max(axis=1), score_dicts, "opinion_owl_deberta")


# name -> (function, files that must exist, notebook that makes them, output file stem)
MODELS = {
    "TF-IDF": (predict_tfidf, [TFIDF_VECTORIZER_PATH, TFIDF_CLASSIFIER_PATH], "notebook 01", "tfidf"),
    "ATAE-LSTM": (predict_atae, [ATAE_MODEL_PATH], "notebook 02", "atae"),
    "DeBERTa-v3": (predict_deberta, [DEBERTA_CONFIG_PATH, DEBERTA_WEIGHTS_PATH], "notebook 03", "deberta"),
}


def model_is_trained(name):
    """Whether inference assets are available locally or through the configured Hub repo."""
    if name == "DeBERTa-v3" and DEBERTA_MODEL_ID:
        return (
            DEBERTA_CONFIG_PATH.is_file() and DEBERTA_WEIGHTS_PATH.is_file()
        ) or bool(DEBERTA_MODEL_ID)
    return all(path.exists() for path in MODELS[name][1])
