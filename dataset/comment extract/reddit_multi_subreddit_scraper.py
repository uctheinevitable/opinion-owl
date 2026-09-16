"""
Multi-Subreddit Comment Scraper -> Doccano-ready labeling export
====================================================================
FIX (v3): previous version could hang indefinitely on some posts. Root
cause: expand_more_comments() was called on EVERY outer scroll round and
could get stuck repeatedly "clicking" a button that doesn't actually
change page state (e.g. faceplate-partial[loading='action'] can match a
mid-load spinner, not a real stub -- clicking it does nothing but the
loop still counts it as progress and keeps retrying). Combined with no
overall time budget, this produced a silent infinite-feeling loop.

Fixes in this version:
  1. Hard wall-clock timeout per post (--per-post-timeout, default 180s).
     If a post exceeds this, it's abandoned with whatever was collected
     so far instead of hanging forever.
  2. expand_more_comments() is now called far less often (every 5th
     scroll round, not every round) and has its own short time budget,
     so a stuck button can't eat the whole run.
  3. Removed the ambiguous faceplate-partial[loading='action'] selector
     -- it matches loading-state indicators, not reliably clickable
     stubs, and was a likely source of the phantom-click loop.
  4. Verbose per-round logging (every round, not every 10th) so a hang
     is visible immediately instead of going silent for minutes.
  5. Explicit heartbeat print with elapsed time, so you can see the
     script is alive even when comment count isn't changing.

INSTALL:
    pip install playwright pandas
    playwright install firefox

USAGE:
    python reddit_multi_subreddit_scraper.py --urls-file domain_url.txt --max-per-post 3000 --per-post-timeout 180
"""

import json
import time
import argparse
from urllib.parse import urlparse

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

VALID_NETLOCS = {"reddit.com", "www.reddit.com", "old.reddit.com", "new.reddit.com"}


def validate_reddit_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.netloc not in VALID_NETLOCS:
        raise ValueError(f"Invalid Reddit URL: {url}")
    if "/comments/" not in parsed.path:
        raise ValueError(f"Not a Reddit post URL: {url}")
    return url.strip()


def expand_more_comments(page, deadline: float, max_rounds: int = 3):
    """Clicks visible 'load more' style buttons. Bounded by both a max
    round count AND a wall-clock deadline so a stuck/phantom button can
    never consume more than its allotted budget."""
    button_selectors = [
        "button:has-text('more replies')",
        "button:has-text('more comments')",
        "button:has-text('View more comments')",
    ]
    for _ in range(max_rounds):
        if time.time() > deadline:
            return
        clicked_any = False
        for sel in button_selectors:
            buttons = page.locator(sel)
            n = min(buttons.count(), 10)  # cap per-selector to avoid pathological click storms
            for i in range(n):
                if time.time() > deadline:
                    return
                try:
                    btn = buttons.nth(i)
                    if btn.is_visible():
                        btn.click(timeout=1500)
                        clicked_any = True
                        page.wait_for_timeout(600)
                except Exception:
                    continue
        if not clicked_any:
            return


def normalize_id(raw_id: str) -> str:
    if not raw_id:
        return raw_id
    return raw_id if raw_id.startswith("t1_") or raw_id.startswith("t3_") else f"t1_{raw_id}"


def scrape_one_post(browser, url: str, max_comments: int, min_words: int, per_post_timeout: int) -> dict:
    validate_reddit_url(url)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.set_default_timeout(20000)

    start_time = time.time()
    hard_deadline = start_time + per_post_timeout

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("shreddit-post", timeout=25000)
        page.wait_for_timeout(2000)
        expand_more_comments(page, deadline=hard_deadline)

        prev_count, stable = -1, 0
        round_num = 0

        while time.time() < hard_deadline:
            round_num += 1
            page.mouse.wheel(0, 8000)
            page.wait_for_timeout(1000)

            # Only run the (potentially slow) button-expansion sweep
            # every 5th round -- most progress comes from scrolling, and
            # this stops a stuck button from eating every single cycle.
            if round_num % 5 == 0:
                expand_more_comments(page, deadline=min(hard_deadline, time.time() + 15))

            count = page.locator("shreddit-comment").count()
            elapsed = time.time() - start_time
            print(f"    [round {round_num}] {count} comments loaded ({elapsed:.0f}s elapsed)")

            if count == prev_count:
                stable += 1
                if stable >= 6:
                    print("    No new comments after 6 rounds, stopping scroll.")
                    break
            else:
                stable = 0
            prev_count = count

            if count >= max_comments:
                print("    Reached --max-per-post cap, stopping scroll.")
                break

        if time.time() >= hard_deadline:
            print(f"    Hit {per_post_timeout}s per-post timeout, taking what was collected so far.")

        post = page.locator("shreddit-post").first
        subreddit = post.get_attribute("subreddit-name") or urlparse(url).path.split("/")[2]
        title = post.get_attribute("post-title")

        comment_elements = page.locator("shreddit-comment")
        total = min(comment_elements.count(), max_comments)
        print(f"  Final DOM comment count: {comment_elements.count()} (processing {total})")

        comments = []
        skipped_short = 0
        for i in range(total):
            comment = comment_elements.nth(i)
            try:
                body_loc = comment.locator('[slot="comment"]')
                text = body_loc.first.inner_text().strip() if body_loc.count() > 0 else ""
                if not text:
                    continue
                if len(text.split()) < min_words:
                    skipped_short += 1
                    continue
                comments.append({
                    "id": normalize_id(comment.get_attribute("thingid")),
                    "score": comment.get_attribute("score"),
                    "text": text,
                })
            except Exception:
                continue

        if skipped_short:
            print(f"  Skipped {skipped_short} comments under {min_words} words")

        return {"url": url, "subreddit": subreddit, "title": title, "comments": comments}

    except PWTimeout:
        print(f"  Page load timed out for {url}, skipping.")
        return {"url": url, "subreddit": None, "title": None, "comments": []}
    finally:
        page.close()


def load_url_list(path: str) -> list:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            url = parts[0].strip()
            domain_tag = parts[1].strip() if len(parts) > 1 else "unspecified"
            entries.append((url, domain_tag))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Scrape multiple Reddit posts across subreddits for ABSA labeling.")
    parser.add_argument("--urls-file", required=True)
    parser.add_argument("--max-per-post", type=int, default=1000)
    parser.add_argument("--min-words", type=int, default=0)
    parser.add_argument("--per-post-timeout", type=int, default=180,
                         help="Max seconds to spend scrolling/expanding a single post before moving on regardless.")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--raw-out", default="reddit_raw_scraped.json")
    parser.add_argument("--labeling-out", default="reddit_for_labeling.csv")
    args = parser.parse_args()

    entries = load_url_list(args.urls_file)
    print(f"Loaded {len(entries)} URLs to scrape.")

    all_results = []
    labeling_rows = []

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=not args.headed)

        for i, (url, domain_tag) in enumerate(entries, 1):
            print(f"[{i}/{len(entries)}] Scraping {url} (domain={domain_tag})")
            t0 = time.time()
            result = scrape_one_post(browser, url, args.max_per_post, args.min_words, args.per_post_timeout)
            result["domain_tag"] = domain_tag
            all_results.append(result)
            print(f"  -> Collected {len(result['comments'])} comments in {time.time()-t0:.0f}s.")

            for c in result["comments"]:
                labeling_rows.append({
                    "text": c["text"],
                    "score": c["score"],
                    "subreddit": result["subreddit"],
                    "domain_tag": domain_tag,
                    "source_permalink": url,
                    "aspect_term": "",
                    "polarity": "",
                })

        browser.close()

    with open(args.raw_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nRaw scrape saved to: {args.raw_out}")

    df = pd.DataFrame(labeling_rows)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df.to_csv(args.labeling_out, index=False)

    print(f"Labeling-ready CSV saved to: {args.labeling_out} ({len(df)} unique comments)")
    print("\n=== Domain balance ===")
    print(df["domain_tag"].value_counts().to_string())
    print("\n=== Per-post breakdown ===")
    for r in all_results:
        print(f"  {r['url']}: {len(r['comments'])} comments")


if __name__ == "__main__":
    main()
