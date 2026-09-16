"""
Aspect Extraction via PyABSA -- for BOTH your scraped Reddit data AND GoEmotions
====================================================================================
Runs PyABSA's pretrained multilingual ATEPC extractor over raw text to
produce candidate aspect terms, for two different downstream uses:

  MODE "reddit"    -> your own scraped comments (reddit_for_labeling.csv).
                      PyABSA fills in 'aspect_term'. Its own predicted
                      sentiment is IGNORED -- you fill 'polarity' yourself
                      afterward, since your judgment on sarcasm/slang/tone
                      matters more than a pretrained guess here.

  MODE "goemotion" -> a GoEmotions CSV (already has human-labeled emotions
                      in separate 0/1 columns). PyABSA extracts the aspect
                      term; sentiment comes from mapping GoEmotions' own
                      27 emotion columns to positive/negative/neutral (NOT
                      from PyABSA's guess, and NOT from you relabeling by
                      hand) -- see map_goemotion_sentiment() below.

Both modes produce output in the same schema so they merge cleanly with
your SemEval CSV later: text, aspect_term, polarity, domain, tier

INSTALL:
    pip install pyabsa pandas

USAGE:
    # Reddit: extract aspects only, leave polarity blank for you to fill in
    python extract_aspects_pyabsa.py --mode reddit \\
        --input reddit_for_labeling.csv --output reddit_aspects_extracted.csv

    # GoEmotions: extract aspects AND map existing emotion labels to sentiment
    python extract_aspects_pyabsa.py --mode goemotion \\
        --input goemotions_1.csv --output goemotion_aspects_extracted.csv
"""

import argparse
import pandas as pd
from pyabsa import AspectTermExtraction as ATEPC

# GoEmotions' 27 emotion columns + neutral, mapped to 3-class sentiment.
# "Ambiguous" emotions (confusion, curiosity, realization, surprise) are
# deliberately excluded from all three buckets, rows where these are the
# ONLY tag get dropped rather than forced into "neutral", since they aren't
# true sentiment-neutral statements, just emotionally ambiguous ones.
POSITIVE_EMOTIONS = {
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief",
}
NEGATIVE_EMOTIONS = {
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
}
NEUTRAL_EMOTIONS = {"neutral"}
AMBIGUOUS_EMOTIONS = {"confusion", "curiosity", "realization", "surprise"}


def map_goemotion_sentiment(row: pd.Series) -> str:
    """
    GoEmotions allows multiple emotion tags per row (one-hot columns).
    Collapse whichever tags are set to exactly one of positive/negative/
    neutral/discard, based on which bucket has the most tags active.
    Ties or rows with tags split across positive AND negative are
    discarded rather than guessed at.
    """
    pos = sum(row.get(e, 0) for e in POSITIVE_EMOTIONS)
    neg = sum(row.get(e, 0) for e in NEGATIVE_EMOTIONS)
    neu = sum(row.get(e, 0) for e in NEUTRAL_EMOTIONS)
    amb = sum(row.get(e, 0) for e in AMBIGUOUS_EMOTIONS)

    if pos == 0 and neg == 0 and neu == 0 and amb == 0:
        return "discard"
    if pos > 0 and neg > 0:
        return "discard"  # mixed signal, don't guess
    if pos > neg and pos > 0:
        return "positive"
    if neg > pos and neg > 0:
        return "negative"
    if neu > 0:
        return "neutral"
    return "discard"  # only ambiguous tags were set


def load_extractor():
    print("Loading PyABSA multilingual aspect extractor (downloads checkpoint on first run)...")
    return ATEPC.AspectExtractor("multilingual", auto_device=True)


def extract_aspects_batch(extractor, texts: list, batch_size: int = 32) -> list:
    """
    Returns a list of lists: one list of extracted aspect terms per input
    text (a single comment/sentence can yield zero, one, or several
    aspects). PyABSA's own sentiment prediction is intentionally not read
    here -- see module docstring for why.
    """
    all_aspects = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i + batch_size]]
        try:
            results = extractor.predict(batch, print_result=False, save_result=False, ignore_error=True)
        except Exception as e:
            print(f"  Batch {i}-{i+batch_size} failed: {e}, skipping batch.")
            results = [{"aspect": []} for _ in batch]

        if isinstance(results, dict):
            results = [results]

        for r in results:
            aspects = r.get("aspect", []) if isinstance(r, dict) else []
            all_aspects.append(aspects)

        if i % (batch_size * 5) == 0:
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts...")

    return all_aspects


def run_reddit_mode(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError("Input CSV must have a 'text' column.")

    extractor = load_extractor()
    aspect_lists = extract_aspects_batch(extractor, df["text"].tolist())

    rows = []
    for (_, row), aspects in zip(df.iterrows(), aspect_lists):
        if not aspects:
            # Keep a row even with no detected aspect so you can review
            # and manually add one if PyABSA missed it (common on
            # Hinglish slang), rather than silently losing the comment.
            rows.append({
                "text": row["text"],
                "aspect_term": "",
                "polarity": "",
                "domain": row.get("domain_tag", "unspecified"),
                "tier": "tier2_own_labeled",
                "subreddit": row.get("subreddit", ""),
            })
        else:
            for aspect in aspects:
                rows.append({
                    "text": row["text"],
                    "aspect_term": aspect,
                    "polarity": "",  # fill in yourself
                    "domain": row.get("domain_tag", "unspecified"),
                    "tier": "tier2_own_labeled",
                    "subreddit": row.get("subreddit", ""),
                })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(out_df)} (text, aspect_term) rows to {output_path}")
    print(f"{(out_df['aspect_term'] == '').sum()} rows have no detected aspect -- review these manually.")
    print("Next: open this CSV and fill in 'polarity' (positive/negative/neutral) for each row yourself.")


def run_goemotion_mode(input_path: str, output_path: str, max_rows: int = None):
    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError("Input CSV must have a 'text' column.")

    if max_rows:
        df = df.head(max_rows)

    df["mapped_sentiment"] = df.apply(map_goemotion_sentiment, axis=1)
    before = len(df)
    df = df[df["mapped_sentiment"] != "discard"].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with no clear/mixed/ambiguous-only sentiment "
          f"({len(df)} rows remain).")

    extractor = load_extractor()
    aspect_lists = extract_aspects_batch(extractor, df["text"].tolist())

    rows = []
    for (_, row), aspects in zip(df.iterrows(), aspect_lists):
        if not aspects:
            continue  # no aspect found -> nothing to pair with the sentiment, skip
        for aspect in aspects:
            rows.append({
                "text": row["text"],
                "aspect_term": aspect,
                "polarity": row["mapped_sentiment"],  # from GoEmotions labels, not re-labeled
                "domain": row.get("subreddit", "goemotion"),
                "tier": "tier3_distant_supervision",
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(out_df)} (text, aspect_term, polarity) rows to {output_path}")
    print("Sentiment came from GoEmotions' own human labels (mapped), NOT re-labeled by you or by PyABSA.")
    print("This is Tier 3 (noisy/distant supervision) -- cap its weight when merging with prepare_multi_domain_dataset.py.")


def main():
    parser = argparse.ArgumentParser(description="Extract aspect terms with PyABSA for Reddit or GoEmotions data.")
    parser.add_argument("--mode", required=True, choices=["reddit", "goemotion"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap, useful for a quick test run first")
    args = parser.parse_args()

    if args.mode == "reddit":
        run_reddit_mode(args.input, args.output)
    else:
        run_goemotion_mode(args.input, args.output, args.max_rows)


if __name__ == "__main__":
    main()
