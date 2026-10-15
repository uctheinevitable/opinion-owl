import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"          # original sources, never edited
PROCESSED_DIR = DATA_DIR / "processed"    # files built from raw by notebook 00
SPLIT_DIR     = DATA_DIR / "splits"       # the one train/validation/test split every model uses
EMBED_DIR     = DATA_DIR / "embeddings"
COMMENTS_DIR  = DATA_DIR / "comments"

MODELS_DIR       = ROOT / "models"
TFIDF_DIR        = MODELS_DIR / "tfidf"
ATAE_DIR         = MODELS_DIR / "atae_lstm"
DEBERTA_DIR      = MODELS_DIR / "deberta_v3_absa"
DEBERTA_BASE_DIR = MODELS_DIR / "deberta_v3_base_pretrained"
DEBERTA_RUNS_DIR = MODELS_DIR / "deberta_v3_runs"   # Trainer scratch, gitignored

# Public Hugging Face model used when local fine-tuned weights are absent.
# Override with OPINION_OWL_DEBERTA_MODEL_ID if you publish a new model version.
DEBERTA_MODEL_ID = os.getenv(
    "OPINION_OWL_DEBERTA_MODEL_ID", "H1manshu-Gupta/opinion-owl"
).strip() or "H1manshu-Gupta/opinion-owl"

RESULTS_DIR     = ROOT / "results"
BEST_MODEL_JSON = RESULTS_DIR / "best_model.json"

SEMEVAL_TRAIN  = RAW_DIR / "semeval_train.csv"
REDDIT_LABELED = RAW_DIR / "reddit_labeled.csv"
COMBINED_TRAIN = PROCESSED_DIR / "combined_train.csv"

TRAIN_SPLIT      = SPLIT_DIR / "train_split.csv"
VALIDATION_SPLIT = SPLIT_DIR / "validation_split.csv"
TEST_SPLIT       = SPLIT_DIR / "test_split.csv"
SPLIT_REPORT     = SPLIT_DIR / "split_report.json"

GLOVE_PATH = EMBED_DIR / "glove.6B.100d.txt"

for _d in (PROCESSED_DIR, SPLIT_DIR, COMMENTS_DIR, RESULTS_DIR,
           TFIDF_DIR, ATAE_DIR, DEBERTA_DIR, DEBERTA_RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
