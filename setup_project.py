#!/usr/bin/env python
"""Setup for Opinion Owl. Run it once after cloning:

    python setup_project.py              # creates .venv, installs packages, downloads GloVe
    python setup_project.py --no-venv    # install into the Python you are already using (conda etc.)
    python setup_project.py --skip-glove # do not download GloVe

What it does:
  1. Installs the packages (requirements.txt) and the project itself (pip install -e .).
  2. Downloads the GloVe word vectors (needed by notebook 02, the ATAE-LSTM).

Everything else is done by running the notebooks in notebooks/ in order (00 to 06).
The pretrained DeBERTa model is downloaded by notebook 03 the first time it runs.
Files that already exist are skipped, so it is safe to run again.
"""
import argparse
import os
import subprocess
import sys
import urllib.request
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"   # 822 MB, holds 50d to 300d
GLOVE_MEMBER = "glove.6B.100d.txt"
GLOVE_PATH = ROOT / "data" / "embeddings" / GLOVE_MEMBER                # same path as config.py

MIN_PYTHON = (3, 10)


def run(command):
    subprocess.check_call([str(part) for part in command], cwd=ROOT)


def install_packages(use_venv):
    if use_venv:
        folder, name = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
        python = ROOT / ".venv" / folder / name
        if not python.exists():
            print("  Creating .venv")
            venv.create(ROOT / ".venv", with_pip=True)
    else:
        python = Path(sys.executable)

    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "-r", "requirements.txt"])
    run([python, "-m", "pip", "install", "-e", "."])


def download_glove(target=GLOVE_PATH, url=GLOVE_URL):
    if target.exists():
        print(f"  Already there: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.parent / "glove.6B.zip"

    def progress(blocks, block_size, total):
        if total > 0:
            print(f"\r  {min(blocks * block_size, total) / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)

    print(f"  Downloading GloVe (large, one time) from {url}")
    urllib.request.urlretrieve(url, archive, progress)
    print()
    with zipfile.ZipFile(archive) as zf:
        zf.extract(GLOVE_MEMBER, target.parent)   # keep only the 100d file
    archive.unlink()
    print(f"  Saved: {target}")


def main():
    parser = argparse.ArgumentParser(description="Setup for Opinion Owl.")
    parser.add_argument("--no-venv", action="store_true", help="install into the current Python instead of creating .venv")
    parser.add_argument("--skip-glove", action="store_true", help="do not download GloVe")
    args = parser.parse_args()

    if sys.version_info < MIN_PYTHON:
        print(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required (you have {sys.version.split()[0]}).")
        return 1

    print("\n[1/2] Installing packages")
    install_packages(use_venv=not args.no_venv)

    print("\n[2/2] GloVe word vectors")
    if args.skip_glove:
        print("  skipped (--skip-glove)")
    else:
        try:
            download_glove()
        except Exception as error:
            print(f"\n  GloVe download failed: {error}")
            print(f"  Download glove.6B.zip yourself and put {GLOVE_MEMBER} in {GLOVE_PATH.parent}")

    print("\nDone. Next:")
    if not args.no_venv:
        print(r"  .venv\Scripts\activate        (macOS/Linux: source .venv/bin/activate)")
    print("  Run the notebooks in notebooks/ in order, 00 to 06.")
    print("  Then start the app:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
