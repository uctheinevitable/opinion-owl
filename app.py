"""Opinion Owl - aspect-based summary of laptop reviews (Streamlit).

Run from the project root:   streamlit run app.py

Pipeline:
  CSV exported by the browser extension -> clean text -> split into sentences
  -> regex finds aspects -> DeBERTa-v3 predicts polarity per (aspect, sentence)
  -> aggregate per aspect -> summary, chips, evidence, chart and phrases.

The model code lives in src/opinion_owl/inference.py, shared with the notebooks.
"""
import html
import json
import re

import numpy as np
import pandas as pd
import streamlit as st

from opinion_owl.aspect_patterns import COMPILED
from opinion_owl.config import BEST_MODEL_JSON, DEBERTA_DIR
from opinion_owl.inference import model_is_trained, predict_deberta

# The extension's CSV uses "text"; the other names let a hand-made CSV work too.
TEXT_COLUMNS = ["text", "comment", "body", "review", "content"]
MIN_COMMENT_CHARS = 15  # drops "Sameee", "[deleted]" and other too-short comments

# Warm beige palette. The same colours are used in the CSS, the chart and the tags.
PALETTE = {
    "page": "#F4EDE1",
    "card": "#FBF7F0",
    "sand": "#EADFCC",
    "line": "#E0D3BD",
    "ink": "#3A2E25",
    "muted": "#7C6B5A",
    "accent": "#A0643C",
    "positive": "#5E8C4F",
    "neutral": "#A2988A",
    "negative": "#B4513A",
}
COLORS = {k: PALETTE[k] for k in ("positive", "neutral", "negative")}
SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

POSITIVE_VERDICTS = ["Mostly positive", "Positive-leaning"]
NEGATIVE_VERDICTS = ["Mostly negative", "Negative-leaning"]


# ------------------------------------------------------------- reading the CSV
def clean_text(text):
    """Strip Reddit/markdown noise so the model sees plain sentences."""
    text = str(text)
    text = re.sub(r"\[([^\]]+)\]\((?:https?://)[^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"^\s*(?:&gt;|>).*$", " ", text, flags=re.MULTILINE)  # quoted replies
    text = re.sub(r"[*_~`#]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_comments_from_csv(source):
    """Read the extension's CSV (path or uploaded file) into a list of clean, unique comments."""
    frame = pd.read_csv(source, encoding="utf-8-sig")
    column = next((c for c in TEXT_COLUMNS if c in frame.columns), None)
    if column is None:
        raise ValueError(
            f"No text column found. Expected one of {TEXT_COLUMNS}. Columns in file: {list(frame.columns)}"
        )

    seen, comments = set(), []
    for value in frame[column].dropna().astype(str):
        text = clean_text(value)
        # The same comment can appear twice when the extension expands threads.
        if len(text) >= MIN_COMMENT_CHARS and text.lower() not in seen:
            seen.add(text.lower())
            comments.append(text)
    return comments


# ------------------------------------------------------------ finding aspects
def split_units(text, granularity):
    """Cut a comment into the pieces that get scored separately."""
    if granularity == "Whole comment":
        return [text]

    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if granularity == "Clause (also split on but/however)":
        # "great screen but bad battery" would otherwise give both aspects the same label.
        clauses = []
        for part in parts:
            clauses += [
                c.strip(" ,;")
                for c in re.split(r"\b(?:but|however|although|though|yet)\b", part, flags=re.IGNORECASE)
            ]
        parts = [c for c in clauses if len(c) >= 3]
    return parts


def find_aspects(text):
    return [aspect for aspect, regexes in COMPILED.items() if any(r.search(text) for r in regexes)]


def build_rows(comments, granularity):
    """One row per (piece of text, aspect) pair - the unit the model scores."""
    rows = []
    for comment_id, comment in enumerate(comments):
        for unit in split_units(comment, granularity):
            for aspect in find_aspects(unit):
                rows.append({"comment_id": comment_id, "text": unit, "aspect_term": aspect})
    return pd.DataFrame(rows, columns=["comment_id", "text", "aspect_term"])


# ----------------------------------------------------------------- the model
@st.cache_resource(show_spinner="Loading DeBERTa model...")
def warm_up_model():
    """Load DeBERTa once per app session. inference.py keeps the loaded copy for later calls."""
    from opinion_owl.inference import _load_deberta
    _load_deberta()
    return True


def analyse(comments, granularity, predictor=predict_deberta, progress=None):
    """Full pass: find aspects, score them, and return one row per aspect mention.

    `predictor` takes a table with text and aspect_term columns and returns a table with
    predicted_polarity and confidence columns. It can be swapped for a fake in tests, so the
    logic runs without the model.
    """
    rows = build_rows(comments, granularity)
    if rows.empty:
        return rows

    if predictor is predict_deberta:
        if not model_is_trained("DeBERTa-v3"):
            raise FileNotFoundError(f"Model not found: {DEBERTA_DIR}. Run notebook 03 (train DeBERTa) first.")
        warm_up_model()

    result = predictor(rows[["text", "aspect_term"]], progress=progress)
    rows["polarity"] = result["predicted_polarity"].to_numpy()
    rows["confidence"] = result["confidence"].to_numpy()
    return rows


# --------------------------------------------------------------- aggregation
def verdict(pos, neu, neg):
    """Turn the positive/neutral/negative shares into a short label."""
    if pos >= 0.6:
        return "Mostly positive"
    if neg >= 0.6:
        return "Mostly negative"
    if pos >= 0.25 and neg >= 0.25:
        return "Mixed"
    return "Positive-leaning" if pos > neg else "Negative-leaning" if neg > pos else "Neutral"


def summarise_aspects(rows, min_confidence, min_mentions):
    """Count polarities per aspect after dropping shaky predictions and rarely-seen aspects."""
    kept = rows[rows["confidence"] >= min_confidence]
    table = (
        kept.groupby("aspect_term")["polarity"]
        .value_counts().unstack(fill_value=0)
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    )
    table["mentions"] = table.sum(axis=1)
    table = table[table["mentions"] >= min_mentions].copy()
    for label in ["positive", "neutral", "negative"]:
        table[f"{label}_share"] = table[label] / table["mentions"]
    table["verdict"] = [
        verdict(r.positive_share, r.neutral_share, r.negative_share) for r in table.itertuples()
    ]
    table["score"] = table["positive_share"] - table["negative_share"]
    return table.sort_values("mentions", ascending=False)


def join_names(names):
    """['a', 'b', 'c'] -> 'a, b and c'."""
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def customers_say(table):
    """One short paragraph (Amazon-style 'Customers say') built from the verdicts.

    This is a template over the predicted verdicts, not generated by a language model.
    """
    liked = table[table["verdict"].isin(POSITIVE_VERDICTS)].head(4).index
    disliked = table[table["verdict"].isin(NEGATIVE_VERDICTS)].head(4).index
    mixed = table[table["verdict"] == "Mixed"].head(4).index

    sentences = []
    if len(liked):
        sentences.append(f"Commenters praise {join_names(liked)}.")
    if len(mixed):
        sentences.append(f"Opinions are mixed on {join_names(mixed)}.")
    if len(disliked):
        start = "However, commenters" if sentences else "Commenters"
        sentences.append(f"{start} criticise {join_names(disliked)}.")
    return " ".join(sentences) if sentences else "Not enough aspect mentions to summarise."


def aspect_blurb(aspect, row):
    """One sentence for the detail card of a single aspect."""
    pos, neg = row["positive_share"], row["negative_share"]
    label = row["verdict"]
    if label == "Mostly positive":
        text = f"Commenters speak well of {aspect}: {pos:.0%} of mentions are positive."
    elif label == "Mostly negative":
        text = f"Commenters are unhappy with {aspect}: {neg:.0%} of mentions are negative."
    elif label == "Mixed":
        text = f"Opinions on {aspect} are split: {pos:.0%} positive and {neg:.0%} negative."
    elif label == "Positive-leaning":
        text = f"Opinions on {aspect} lean positive ({pos:.0%} positive, {neg:.0%} negative)."
    elif label == "Negative-leaning":
        text = f"Opinions on {aspect} lean negative ({neg:.0%} negative, {pos:.0%} positive)."
    else:
        text = f"{aspect.capitalize()} is mostly mentioned without a strong opinion."
    return text[0].upper() + text[1:]


def frequent_phrases(comments, rows, top_n=15):
    """Most repeated 2-3 word phrases, each with the average polarity of comments using it."""
    from sklearn.feature_extraction.text import CountVectorizer

    empty = pd.DataFrame(columns=["phrase", "comments", "polarity", "score"])
    if len(comments) < 3:
        return empty

    # binary=True counts each phrase once per comment, so one long rant cannot dominate.
    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words="english", min_df=2, binary=True)
    try:
        matrix = vectorizer.fit_transform(comments)
    except ValueError:  # no phrase appears in two comments
        return empty

    rows = rows.copy()
    rows["score"] = rows["polarity"].map(SCORE)
    comment_score = rows.groupby("comment_id")["score"].mean()
    scores = np.array([comment_score.get(i, np.nan) for i in range(len(comments))])

    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    result = []
    # Look at extra candidates (top_n * 3) because phrases in comments with no aspect get dropped.
    for j in np.argsort(-counts)[: top_n * 3]:
        docs = matrix[:, j].nonzero()[0]
        value = np.nanmean(scores[docs]) if np.isfinite(scores[docs]).any() else np.nan
        result.append({"phrase": terms[j], "comments": int(counts[j]), "score": value})

    out = pd.DataFrame(result).dropna(subset=["score"]).head(top_n)
    out["polarity"] = np.where(out["score"] > 0.2, "positive", np.where(out["score"] < -0.2, "negative", "neutral"))
    return out[["phrase", "comments", "polarity", "score"]].round(2)


# --------------------------------------------------------------------- style
CSS = """
<style>
:root {
  --page: %(page)s; --card: %(card)s; --sand: %(sand)s; --line: %(line)s;
  --ink: %(ink)s; --muted: %(muted)s; --accent: %(accent)s;
  --pos: %(positive)s; --neu: %(neutral)s; --neg: %(negative)s;
}
.stApp { background: var(--page); color: var(--ink); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1080px; padding-top: 1.2rem; padding-bottom: 4rem; }
h1, h2, h3, .oo-serif { font-family: Georgia, "Times New Roman", serif !important; color: var(--ink); letter-spacing: -0.01em; }
section[data-testid="stSidebar"] { background: var(--sand); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] * { color: var(--ink); }

.oo-nav { display:flex; align-items:center; justify-content:space-between; padding: 0.4rem 0 1.2rem 0; border-bottom: 1px solid var(--line); margin-bottom: 1.6rem; }
.oo-brand { font-family: Georgia, serif; font-size: 1.35rem; font-weight: 700; }
.oo-brand span { color: var(--accent); }
.oo-nav-note { color: var(--muted); font-size: 0.85rem; }

.oo-hero { padding: 1.2rem 0 1.4rem 0; }
.oo-hero h1 { font-size: 2.6rem; line-height: 1.12; margin: 0 0 0.6rem 0; }
.oo-hero p { color: var(--muted); font-size: 1.08rem; max-width: 40rem; margin: 0; }

.oo-steps { display:grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 1.2rem 0 1.6rem 0; }
.oo-step { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; }
.oo-step b { display:inline-flex; width:26px; height:26px; align-items:center; justify-content:center; border-radius:50%%; background: var(--accent); color: #fff; font-size: 0.85rem; margin-right: 8px; }
.oo-step div { color: var(--muted); font-size: 0.92rem; margin-top: 6px; }

.oo-kpis { display:grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 0.4rem 0 1.4rem 0; }
.oo-kpi { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px 18px; }
.oo-kpi .v { font-family: Georgia, serif; font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.oo-kpi .l { color: var(--muted); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; }

.oo-card { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 22px 26px; margin-bottom: 1.2rem; }
.oo-card h2 { font-size: 1.5rem; margin: 0 0 0.5rem 0; }
.oo-say { font-size: 1.05rem; line-height: 1.6; margin: 0 0 0.4rem 0; }
.oo-src { color: var(--muted); font-size: 0.85rem; }
.oo-sub { font-weight: 600; margin: 1.1rem 0 0.4rem 0; }

.oo-detail-head { display:flex; align-items:baseline; gap: 14px; flex-wrap: wrap; margin-bottom: 0.3rem; }
.oo-detail-head .t { font-size: 1.05rem; }
.oo-pos { color: var(--pos); font-weight: 600; }
.oo-neu { color: var(--neu); font-weight: 600; }
.oo-neg { color: var(--neg); font-weight: 600; }
.oo-blurb { margin: 0.2rem 0 1rem 0; }

.oo-quote { border-left: 4px solid var(--line); padding: 4px 0 4px 14px; margin: 12px 0; }
.oo-quote.positive { border-left-color: var(--pos); }
.oo-quote.neutral { border-left-color: var(--neu); }
.oo-quote.negative { border-left-color: var(--neg); }
.oo-quote .meta { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.oo-quote b { background: rgba(160,100,60,0.16); padding: 0 3px; border-radius: 3px; font-weight: 700; }
.oo-quote details { margin-top: 4px; }
.oo-quote summary { color: var(--accent); cursor: pointer; font-size: 0.9rem; }
.oo-quote details p { color: var(--muted); font-size: 0.92rem; margin: 6px 0 0 0; }

.oo-tags { display:flex; flex-wrap:wrap; gap: 8px; }
.oo-tag { border-radius: 999px; padding: 5px 12px; font-size: 0.9rem; border: 1px solid var(--line); background: var(--card); }
.oo-tag i { font-style: normal; color: var(--muted); margin-left: 6px; font-size: 0.8rem; }
.oo-tag.positive { border-color: var(--pos); } .oo-tag.negative { border-color: var(--neg); } .oo-tag.neutral { border-color: var(--neu); }

.oo-foot { color: var(--muted); font-size: 0.82rem; text-align:center; margin-top: 2.4rem; }

/* Buttons, pills and tabs in the same palette */
.stButton > button[kind="primary"], .stDownloadButton > button { background: var(--accent); border: 1px solid var(--accent); color: #fff; border-radius: 10px; }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover { background: #87522F; border-color: #87522F; color: #fff; }
button[data-testid="stBaseButton-pills"], button[data-testid="stBaseButton-pillsActive"] {
  border-radius: 999px; border: 1px solid var(--line); background: var(--card); color: var(--ink); font-weight: 500; }
button[data-testid="stBaseButton-pillsActive"] { background: var(--accent); border-color: var(--accent); color: #fff; }
div[data-testid="stFileUploader"] section { background: var(--card); border: 1.5px dashed var(--accent); border-radius: 14px; }
.stTabs [data-baseweb="tab"] { color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--accent); }
@media (max-width: 760px) { .oo-steps, .oo-kpis { grid-template-columns: 1fr 1fr; } .oo-hero h1 { font-size: 2rem; } }
</style>
""" % PALETTE


def block(markup):
    """Flatten HTML to one line. Blank lines or 4-space indents would make Markdown treat it as code."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def show_html(markup):
    st.markdown(block(markup), unsafe_allow_html=True)


def highlight(text, aspect):
    """Escape text for HTML and bold the words the aspect patterns matched."""
    spans = sorted(
        m.span() for rx in COMPILED.get(aspect, []) for m in rx.finditer(text) if m.end() > m.start()
    )
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    out, cursor = [], 0
    for start, end in merged:
        out.append(html.escape(text[cursor:start]))
        out.append(f"<b>{html.escape(text[start:end])}</b>")
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "".join(out)


def trend_icon(verdict_label):
    if verdict_label in POSITIVE_VERDICTS:
        return "\u2197"      # up-right arrow
    if verdict_label in NEGATIVE_VERDICTS:
        return "\u2198"      # down-right arrow
    return "\u223c"          # tilde: mixed or neutral


# ------------------------------------------------------------------------ UI
def render_header(has_results):
    show_html("""
      <div class="oo-nav">
        <div class="oo-brand">&#129417; Opinion <span>Owl</span></div>
        <div class="oo-nav-note">Aspect-based summaries of laptop discussions</div>
      </div>
    """)
    if has_results:
        return
    show_html("""
      <div class="oo-hero">
        <h1>What are people really saying about this laptop?</h1>
        <p>Upload the comments you exported from a Reddit thread. Opinion Owl finds the features people
        talk about, such as battery, screen and price, and shows how they feel about each one.</p>
      </div>
      <div class="oo-steps">
        <div class="oo-step"><b>1</b>Export<div>Save a thread's comments as a CSV with the browser extension.</div></div>
        <div class="oo-step"><b>2</b>Upload<div>Drop the file below and press Analyse.</div></div>
        <div class="oo-step"><b>3</b>Read<div>Get a short summary, a sentiment chart and the quotes behind it.</div></div>
      </div>
    """)


def sidebar():
    st.sidebar.markdown("### Settings")
    granularity = st.sidebar.selectbox(
        "Analyse at the level of",
        ["Sentence", "Clause (also split on but/however)", "Whole comment"],
        help="The model was trained on sentence-level data, so Sentence or Clause usually works best. "
             "Clause stops one half of 'great screen but bad battery' leaking into the other aspect.",
    )
    min_conf = st.sidebar.slider("Ignore predictions below confidence", 0.34, 0.95, 0.50, 0.01)
    min_mentions = st.sidebar.slider("Minimum mentions per aspect", 1, 10, 2)

    st.sidebar.divider()
    if BEST_MODEL_JSON.exists():
        info = json.loads(BEST_MODEL_JSON.read_text())
        st.sidebar.caption(
            f"Model: {info['best_model']}  \n"
            f"Test macro-F1: {info['macro_f1']:.2f}, accuracy: {info['accuracy']:.2f} "
            f"on {info['rows_scored']} labeled rows."
        )
    st.sidebar.caption("Roughly 1 in 5 aspect labels can be wrong at this accuracy. Treat small counts with care.")
    return granularity, min_conf, min_mentions


def get_comments_from_upload():
    """Show the CSV uploader and return the comments once the user clicks Analyse."""
    uploaded = st.file_uploader("CSV exported by the extension (needs a `text` column)", type="csv")

    if st.button("Analyse CSV", type="primary") and uploaded is not None:
        try:
            return load_comments_from_csv(uploaded)
        except ValueError as error:
            st.error(str(error))
    return None


def render_kpis(comments, rows, table, excluded):
    tiles = [
        (len(comments), "Comments"),
        (len(rows), "Aspect mentions"),
        (len(table), "Aspects shown"),
        (excluded, "Low confidence ignored"),
    ]
    show_html(
        '<div class="oo-kpis">'
        + "".join(f'<div class="oo-kpi"><div class="v">{v:,}</div><div class="l">{label}</div></div>' for v, label in tiles)
        + "</div>"
    )


def pick_aspect(table):
    """Aspect chips (Amazon-style). Returns the selected aspect name."""
    options = {
        f"{trend_icon(row.verdict)} {aspect.title()} ({int(row.mentions)})": aspect
        for aspect, row in table.head(12).iterrows()
    }
    labels = list(options)
    if hasattr(st, "pills"):
        chosen = st.pills("Select to learn more", labels, default=labels[0], label_visibility="collapsed")
    else:  # older Streamlit versions
        chosen = st.radio("Select to learn more", labels, horizontal=True, label_visibility="collapsed")
    return options.get(chosen, options[labels[0]])


def render_quotes(kept, comments, aspect, polarity_filter, limit=6):
    subset = kept[kept["aspect_term"] == aspect]
    if polarity_filter != "All":
        subset = subset[subset["polarity"] == polarity_filter.lower()]
    # With "All", show the most confident quotes of each polarity so both sides appear.
    subset = subset.sort_values("confidence", ascending=False)
    if polarity_filter == "All":
        subset = subset.groupby("polarity", group_keys=False).head(2).sort_values("confidence", ascending=False)
    subset = subset.head(limit)

    if subset.empty:
        st.caption("No quotes for this filter.")
        return

    for r in subset.itertuples():
        full = comments[r.comment_id]
        more = ""
        if len(full) > len(r.text) + 5:
            more = f"<details><summary>Read more</summary><p>{html.escape(full)}</p></details>"
        show_html(
            f'<div class="oo-quote {r.polarity}">'
            f'<div class="meta">{r.polarity} &middot; confidence {r.confidence:.2f}</div>'
            f"&ldquo;{highlight(r.text, aspect)}&rdquo;{more}</div>"
        )


def render_say_card(comments, rows, table, min_conf):
    kept = rows[rows["confidence"] >= min_conf]

    show_html(
        f'<div class="oo-card"><h2>Commenters say</h2>'
        f'<p class="oo-say">{html.escape(customers_say(table))}</p>'
        f'<div class="oo-src">&#10024; Summarised from the predicted sentiment of each comment</div>'
        f'<div class="oo-sub">Select to learn more</div></div>'
    )
    aspect = pick_aspect(table)
    row = table.loc[aspect]

    show_html(
        f'<div class="oo-card"><div class="oo-detail-head">'
        f'<span class="t"><b>{int(row.mentions)}</b> mentions of &ldquo;{html.escape(aspect)}&rdquo;</span>'
        f'<span class="oo-pos">{int(row.positive)} positive</span>'
        f'<span class="oo-neu">{int(row.neutral)} neutral</span>'
        f'<span class="oo-neg">{int(row.negative)} negative</span></div>'
        f'<p class="oo-blurb">{html.escape(aspect_blurb(aspect, row))}</p></div>'
    )
    polarity_filter = st.radio(
        "Show quotes", ["All", "Positive", "Neutral", "Negative"], horizontal=True, label_visibility="collapsed",
    )
    render_quotes(kept, comments, aspect, polarity_filter)

    with st.expander(f"See all {int(row.mentions)} mentions of {aspect}"):
        every = kept[kept["aspect_term"] == aspect][["text", "polarity", "confidence"]]
        st.dataframe(every.sort_values("confidence", ascending=False), use_container_width=True, hide_index=True)


def render_chart(table):
    import altair as alt

    long = table.reset_index().melt(
        id_vars=["aspect_term", "mentions"],
        value_vars=["positive_share", "neutral_share", "negative_share"],
        var_name="polarity", value_name="share",
    )
    long["polarity"] = long["polarity"].str.replace("_share", "", regex=False)
    chart = (
        alt.Chart(long)
        .mark_bar(cornerRadius=3)
        .encode(
            y=alt.Y("aspect_term:N", sort=table.index.tolist(), title=None),
            x=alt.X("share:Q", stack="normalize", axis=alt.Axis(format="%", grid=False), title="Share of mentions"),
            color=alt.Color(
                "polarity:N",
                scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
            order=alt.Order("polarity:N", sort="descending"),
            tooltip=["aspect_term", "polarity", alt.Tooltip("share:Q", format=".0%"), "mentions"],
        )
        .properties(height=max(120, 32 * len(table)), background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=PALETTE["ink"], titleColor=PALETTE["muted"], domainColor=PALETTE["line"])
        .configure_legend(labelColor=PALETTE["ink"])
    )
    st.altair_chart(chart, use_container_width=True)


def render_phrases(comments, rows):
    phrases = frequent_phrases(comments, rows)
    if phrases.empty:
        st.caption("Not enough repeated phrases yet. Try more comments.")
        return
    tags = "".join(
        f'<span class="oo-tag {r.polarity}">{html.escape(r.phrase)}<i>{r.comments}</i></span>'
        for r in phrases.itertuples()
    )
    show_html(f'<div class="oo-tags">{tags}</div>')
    st.caption("Colour shows the average predicted polarity of the comments that use the phrase. "
               "The number is how many comments use it.")


def show_results(comments, rows, min_conf, min_mentions):
    if rows.empty:
        st.warning("No aspects were found in these comments. Add patterns in src/opinion_owl/aspect_patterns.py.")
        return

    table = summarise_aspects(rows, min_conf, min_mentions)
    excluded = int((rows["confidence"] < min_conf).sum())
    render_kpis(comments, rows, table, excluded)

    if table.empty:
        st.info("No aspect reached the minimum number of mentions. Lower the threshold in the sidebar.")
        return

    render_say_card(comments, rows, table, min_conf)

    st.markdown("### Sentiment by aspect")
    chart_tab, table_tab = st.tabs(["Chart", "Table"])
    with chart_tab:
        render_chart(table)
    with table_tab:
        display = table[["mentions", "verdict", "positive", "neutral", "negative"]].copy()
        display.insert(1, "positive %", (table["positive_share"] * 100).round(0).astype(int))
        display.insert(2, "negative %", (table["negative_share"] * 100).round(0).astype(int))
        st.dataframe(display, use_container_width=True)

    st.markdown("### Frequently used phrases")
    render_phrases(comments, rows)

    st.download_button(
        "Download all predictions (CSV)",
        rows.to_csv(index=False).encode("utf-8"),
        file_name="opinion_owl_predictions.csv",
        mime="text/csv",
    )
    show_html('<div class="oo-foot">Opinion Owl &middot; predictions come from a fine-tuned DeBERTa-v3 model '
              'and can be wrong, so read the quotes before drawing conclusions.</div>')


def main():
    st.set_page_config(page_title="Opinion Owl", page_icon=":owl:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    granularity, min_conf, min_mentions = sidebar()
    render_header(has_results="result" in st.session_state)

    comments = get_comments_from_upload()

    if comments:
        progress = st.progress(0.0, text="Predicting polarity...")
        try:
            rows = analyse(comments, granularity, progress=progress)
        except FileNotFoundError as error:
            st.error(str(error))
            return
        progress.empty()
        # Keep the result in session state: Streamlit reruns the script on every
        # slider change, and we do not want to re-run the model each time.
        st.session_state["result"] = (comments, rows)
        st.rerun()   # redraw so the header switches to its compact form

    if "result" in st.session_state:
        comments, rows = st.session_state["result"]
        show_results(comments, rows, min_conf, min_mentions)


if __name__ == "__main__":
    main()
