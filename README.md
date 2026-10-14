# Opinion Owl

**Turn a pile of laptop comments into a clear answer: what do people like, and what do they complain about?**

Opinion Owl reads laptop discussions from Reddit, finds the things people talk about (battery, keyboard, price, RAM and so on), and works out whether each opinion is **positive**, **neutral** or **negative**. A browser extension exports the comments of a thread, and a Streamlit dashboard turns them into a short summary, a chart and the quotes behind every number.

---

## Table of contents

1. [What problem does it solve?](#what-problem-does-it-solve)
2. [How it works](#how-it-works)
3. [Models and results](#models-and-results)
4. [Using the dashboard](#using-the-dashboard)
5. [The browser extension](#the-browser-extension)
6. [Getting started](#getting-started)
7. [Project structure](#project-structure)
8. [The notebooks](#the-notebooks)
9. [Data](#data)
10. [Limitations](#limitations)
11. [Future ideas](#future-ideas)
12. [Tech stack](#tech-stack)
13. [About the author](#about-the-author)

---

## What problem does it solve?

A laptop thread with 300 comments is hard to read. One comment can praise the screen, complain about the battery and shrug at the price, all in two sentences. A normal sentiment tool gives that comment a single score, which hides the details.

Opinion Owl does **aspect-based sentiment analysis (ABSA)**. It gives each topic its own verdict.

> "Great for school and Office work, but the 8GB RAM is restrictive."

| Aspect | Verdict |
|---|---|
| school | positive |
| 8gb ram | negative |

Run over a whole thread, this becomes a per-aspect summary such as "battery: mostly negative, price: mostly positive".

---

## How it works

```
Reddit thread -> extension -> CSV -> clean text -> split into sentences -> find aspects
              -> predict polarity for each (aspect, sentence) -> count per aspect -> dashboard
```

1. **Export.** The browser extension saves the comments of a Reddit thread as a CSV.
2. **Clean and split.** The dashboard removes links, markdown and quoted replies, drops very short comments and duplicates, and cuts every comment into sentences. You can also choose clauses (split on "but" or "however") or whole comments.
3. **Find aspects.** A rule-based extractor (`src/opinion_owl/aspect_patterns.py`) spots words like "8gb, ram" or "fans, loud" and maps them to one clean aspect name such as `ram` or `noise`.
4. **Classify.** Each (aspect, sentence) pair goes to the fine-tuned DeBERTa-v3 model, which predicts negative, neutral or positive with a confidence.
5. **Summarise.** Predictions below the confidence threshold are ignored. The rest are counted per aspect and turned into a verdict, a chart, a short "Commenters say" paragraph and example quotes.

---

## Models and results

Three models were trained on the same data and scored on the same held-out test set of **265 labelled rows**. The split keeps every sentence in one part only, so a model is never tested on a sentence it saw in training with a different aspect. The labels are three classes: `conflict` counts as `neutral`.

| Model | Idea | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|---|
| **DeBERTa-v3** (fine-tuned) | Large pretrained language model that reads the aspect and the sentence together | **0.909** | **0.899** | 0.923 | 0.844 | 0.929 |
| TF-IDF + linear classifier | Classic word-count baseline | 0.691 | 0.651 | 0.691 | 0.490 | 0.772 |
| ATAE-LSTM | Small neural network with attention that focuses on the words relevant to the aspect | 0.657 | 0.619 | 0.683 | 0.434 | 0.741 |

**How to read this**

- DeBERTa is clearly the best model and is the one the dashboard uses. Its lead is about 25 points of macro-F1, far more than the test set's margin of error.
- With 265 rows, an accuracy of 0.91 has a margin of roughly plus or minus 3.5 points (95% range). TF-IDF and ATAE-LSTM are only 3 points apart, so the test set cannot say which of the two is better.
- Neutral is the hardest class for every model.
- Many rows that every model gets wrong look like label noise rather than model weakness.

The test set was used only for the final comparison, never for training or for choosing a best epoch. Your own numbers can differ slightly, because training is not perfectly repeatable across machines.

---

## Using the dashboard

```powershell
streamlit run app.py
```

1. Upload the CSV exported by the extension. It needs a `text` column.
2. Choose the settings in the sidebar.
3. Press **Analyse CSV**.

**Settings**

| Setting | What it does |
|---|---|
| Analyse at the level of | Sentence (default), Clause (also splits on but/however) or Whole comment. The model was trained on sentence-level data, so Sentence or Clause usually works best. |
| Ignore predictions below confidence | Drops shaky predictions. Default 0.50. |
| Minimum mentions per aspect | Hides aspects that are mentioned too rarely. Default 2. |

**What you get**

- Four counters: comments, aspect mentions, aspects shown and low-confidence predictions ignored.
- A **"Commenters say"** paragraph. It is a template filled in from the verdicts, not text written by a language model.
- Aspect chips with a trend arrow. Select one to see its counts, a one-line summary and the best quotes, with the matched words highlighted.
- A stacked bar chart and a table of the positive, neutral and negative share per aspect.
- The most repeated two- and three-word phrases, coloured by the average sentiment of the comments that use them.
- A button to download every prediction as a CSV.

**How a verdict is chosen:** 60% or more positive is *Mostly positive*, 60% or more negative is *Mostly negative*, at least 25% of each is *Mixed*, and otherwise the aspect is *Positive-leaning*, *Negative-leaning* or *Neutral*.

---

## The browser extension

The extension in `extension/` exports a Reddit thread as a CSV, ready for the dashboard.

1. Open a Reddit post page. It must be the new Reddit design, not `old.reddit.com`.
2. Click the extension icon in the toolbar.
3. The page scrolls and expands "more replies" on its own, and a small status box shows the progress.
4. A CSV named `owl_<post id>_<timestamp>.csv` downloads. The icon badge shows the number of comments saved, or `ERR` if something went wrong.

Details:

- It collects up to **1,000 comments** and follows up to **15** "continue this thread" branches. Both limits are constants at the top of `extension/background.js`.
- Deleted and removed comments and AutoModerator posts are skipped. The post text itself is saved as the first row.
- The CSV columns are `comment_id`, `parent_id`, `depth`, `author`, `score`, `permalink`, `text`, `post_title`, `post_url` and `scraped_at`.
- It only runs on `reddit.com` post pages.

To install it: open `chrome://extensions` in Chrome or Edge, turn on **Developer mode**, click **Load unpacked** and choose the `extension/` folder.

---

## Getting started

**You need:** Python 3.11 or newer (the project was developed on Python 3.14). A GPU is optional but makes DeBERTa training much faster. The default settings were chosen for a 6 GB laptop GPU. Leave a few GB of free disk space, because the pretrained DeBERTa model, its training checkpoints and the GloVe vectors are large.

### 1. Clone and install

```powershell
git clone https://github.com/<your-username>/opinion-owl.git
cd opinion-owl
python setup_project.py
```

This creates a `.venv`, installs the libraries, installs the project itself and downloads the GloVe word vectors (822 MB download, once).

Useful options:

```powershell
python setup_project.py --no-venv      # use the Python you already have (conda etc.)
python setup_project.py --skip-glove   # skip the GloVe download
```

Then activate the environment:

```powershell
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
```

### 2. Run the notebooks

Open the `notebooks/` folder and run the notebooks **in order, 00 to 06**. Each one says what it needs and what it creates. The pretrained DeBERTa model is downloaded automatically the first time notebook 03 runs. Select the `.venv` as the Jupyter kernel.

### 3. Start the dashboard

```powershell
streamlit run app.py
```

---

## Project structure

```
opinion-owl/
├── app.py                  Streamlit dashboard
├── setup_project.py        One-time setup (packages + GloVe)
├── pyproject.toml          Package definition
├── requirements.txt        Library list
├── .streamlit/config.toml  Dashboard theme
├── extension/              Browser extension (manifest.json, background.js, content.js)
├── notebooks/              The pipeline, run in order 00 to 06
├── src/opinion_owl/
│   ├── config.py           Folder paths and settings
│   ├── aspect_patterns.py  Rule-based aspect extraction
│   ├── atae.py             ATAE-LSTM model and tokenizer
│   └── inference.py        Loading the models and predicting
├── data/raw/               The two labelled source datasets (kept in Git)
│   └── (processed, splits, embeddings and comments are created for you)
├── models/                 Trained models (created by the notebooks)
└── results/                Predictions and comparison tables (created by the notebooks)
```

Folders that are rebuilt automatically (`data/processed`, `data/splits`, `data/embeddings`, `data/comments`, `models`, `results`) are not stored in Git.

---

## The notebooks

| # | Notebook | What it does |
|---|---|---|
| 00 | `00_prepare_data` | Combines the raw datasets, cleans labels, removes duplicates and contradictions, and makes the train, validation and test splits |
| 01 | `01_train_tfidf` | Trains the TF-IDF baseline |
| 02 | `02_train_atae_lstm` | Trains the ATAE-LSTM using GloVe vectors |
| 03 | `03_train_deberta` | Fine-tunes DeBERTa-v3 |
| 04 | `04_compare_models` | Scores all three on the test set, checks for overlap with training data, adds confidence ranges, saves the winner for the dashboard |
| 05 | `05_extract_aspects` | Finds aspects in a set of comments (built-in test comments or a real CSV) |
| 06 | `06_predict_comments` | Predicts the sentiment of each aspect with all three models and shows where they agree |

---

## Data

| | Rows |
|---|---|
| Total after cleaning | 2,642 |
| SemEval laptop reviews (`data/raw/semeval_train.csv`) | 2,276 |
| Reddit comments (`data/raw/reddit_labeled.csv`) | 366 |
| Train / validation / test | 2,113 / 264 / 265 |

- **SemEval laptop reviews:** sentences with a labelled aspect term and its polarity.
- **Reddit comments:** comments labelled by hand, so the models also see informal internet language. Long comments are cut to a window of 80 words around the aspect, because the models only read the start of a text.

| Label | Train | Validation | Test |
|---|---|---|---|
| positive | 932 | 116 | 117 |
| negative | 781 | 98 | 98 |
| neutral | 365 | 46 | 46 |
| conflict | 35 | 4 | 4 |

The dashboard shows three classes, so `conflict` is counted as `neutral` when scoring. The split is made once, with a fixed seed, and keeps all rows of the same sentence together. Its report (`data/splits/split_report.json`) holds a fingerprint of the data, so any result can be tied to the exact data behind it.

> Add the source and licence of each dataset here.

---

## Limitations

- **Small test set.** 265 rows means scores can move by a few points by chance. The gap between DeBERTa and the other two is large enough to trust, but TF-IDF and ATAE-LSTM cannot be separated.
- **Small labelled Reddit set.** Only 366 rows come from Reddit, and only 43 rows carry the `conflict` label.
- **Roughly 1 in 11 aspect labels is wrong** even for the best model (accuracy 0.91). Read the quotes behind any number that matters.
- **Rule-based aspects.** The extractor only finds aspects its patterns know about, and it can miss new phrasing.
- **Sentence-level opinions.** A sentence with two aspects and opposite opinions can confuse the model. The Clause setting helps with "but" and "however".
- **Laptop focus.** The models are trained on laptop language. Other products would need new data.
- **Reddit only, new design only.** The extension works on new Reddit post pages and collects at most 1,000 comments per thread.
- **English only.**

---

## Future ideas

- Replace the rule-based aspect extractor with a trained model, so it also learns new aspects.
- Add more labelled data, especially Reddit comments and neutral examples, and re-test on a larger test set.
- Bring back a real `conflict` class so mixed opinions are shown as mixed.
- Support other products (phones, headphones, cameras) and other sites besides Reddit.
- Add a "trend over time" view so you can see opinion change after a software update or a new model release.
- Compare two threads or two laptops side by side.
- Make the dashboard faster with a smaller distilled model for use on a laptop without a GPU.
- Package the extension for the Chrome Web Store.

---

## Tech stack

Python, PyTorch, Hugging Face Transformers (DeBERTa-v3), scikit-learn, GloVe, pandas, Streamlit, Altair, JavaScript (Manifest V3 browser extension), Jupyter.

---

Feedback, ideas and bug reports are welcome. Open an issue or send a message.

---

## Acknowledgements and licence

- SemEval for the aspect-based sentiment datasets.
- Stanford NLP for the GloVe word vectors.
- Microsoft for DeBERTa-v3.

Licence: MIT for the code (see `LICENSE`). The datasets keep their own terms: SemEval-2014 data is for non-commercial research use, and GloVe vectors are released under the Public Domain Dedication and License.
