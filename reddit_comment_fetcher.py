"""
Reddit Comment Fetcher for Aspect-Based Sentiment Analysis (ABSA)
==================================================================

DEPTH LIMIT NOTES (read before relying on this for very deep threads):

Reddit itself has no hard architectural depth cap, but its UI does: once a
reply chain exceeds a render cutoff (roughly depth 8-10 on modern
reddit.com, historically as low as 5-6 on old.reddit.com), Reddit replaces
the rest of that branch with a "Continue this thread ->" LINK to a separate
page focused on just that sub-thread, instead of lazy-loading it inline.

Earlier versions of this script only clicked "X more replies" / "X more
comments" BUTTONS on the same page -- it never followed "continue this
thread" links, so any reply chain longer than the render cutoff was
silently truncated with no warning.

This version detects "continue this thread" links, opens each one in a
new page, scrapes that branch's comments (which restart their own
button-expansion + depth counting on the focused page), and splices them
back into the main tree by parent_id. This lets the script follow chains
arbitrarily deep, at the cost of one extra page load per truncated branch
-- so very argumentative threads will take noticeably longer to scrape.

max_depth_traversals caps how many "continue this thread" links get
followed per run, as a safety valve against pathological threads with
hundreds of deep branches (each one is a full page navigation).

INSTALL:
    pip install playwright
    playwright install firefox

USAGE:
    python reddit_comment_fetcher.py "https://www.reddit.com/r/xyz/comments/abc123/title/"
    python reddit_comment_fetcher.py <url> --max-comments 300 --max-depth-traversals 15 --headed
"""

import json
import time
import argparse
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


VALID_NETLOCS = {"reddit.com", "www.reddit.com", "old.reddit.com", "new.reddit.com"}


def validate_reddit_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.netloc not in VALID_NETLOCS:
        raise ValueError("Please enter a valid Reddit URL (reddit.com/...).")
    if "/comments/" not in parsed.path:
        raise ValueError("The URL does not appear to be a Reddit post.")
    return url.strip()


def get_post_body_text(post_locator) -> str:
    candidates = [
        "shreddit-post-text-body div[slot='text-body']",
        "shreddit-post-text-body",
        "div[slot='text-body']",
    ]
    for selector in candidates:
        loc = post_locator.locator(selector)
        if loc.count() == 0:
            continue
        try:
            return loc.first.inner_text().strip()
        except Exception:
            continue
    return ""


def expand_more_comments(page, max_rounds: int = 20):
    """Clicks in-page 'more replies'/'more comments' buttons. Does NOT
    handle 'continue this thread' links -- those are separate pages,
    handled by expand_continue_threads()."""
    button_selectors = [
        "button:has-text('more replies')",
        "button:has-text('more comments')",
        "button:has-text('View more comments')",
        "faceplate-partial[loading='action']",
    ]
    for _ in range(max_rounds):
        clicked_any = False
        for sel in button_selectors:
            buttons = page.locator(sel)
            for i in range(buttons.count()):
                try:
                    btn = buttons.nth(i)
                    if btn.is_visible():
                        btn.click(timeout=2000)
                        clicked_any = True
                        page.wait_for_timeout(1000)
                except Exception:
                    continue
        if not clicked_any:
            break
        page.wait_for_timeout(500)


def normalize_id(raw_id: str) -> str:
    if not raw_id:
        return raw_id
    return raw_id if raw_id.startswith("t1_") or raw_id.startswith("t3_") else f"t1_{raw_id}"


def scrape_flat_comments(page, max_comments: int) -> list:
    comment_elements = page.locator("shreddit-comment")
    total = min(comment_elements.count(), max_comments)

    flat = []
    for i in range(total):
        comment = comment_elements.nth(i)
        try:
            comment_id = comment.get_attribute("thingid")
            parent_id_raw = comment.get_attribute("parentid")
            comment_author = comment.get_attribute("author")
            comment_score = comment.get_attribute("score")
            depth = comment.get_attribute("depth")
            permalink = comment.get_attribute("permalink")

            body_loc = comment.locator('[slot="comment"]')
            text = body_loc.first.inner_text().strip() if body_loc.count() > 0 else ""
            if not text:
                continue

            comment_id_norm = normalize_id(comment_id)
            parent_id_norm = normalize_id(parent_id_raw)
            is_top_level = bool(parent_id_norm and parent_id_norm.startswith("t3_"))

            flat.append({
                "id": comment_id_norm,
                "parent_id": None if is_top_level else parent_id_norm,
                "author": comment_author,
                "score": comment_score,
                "depth": int(depth) if depth and depth.isdigit() else depth,
                "permalink": f"https://reddit.com{permalink}" if permalink else None,
                "text": text,
            })
        except Exception:
            continue

    return flat


def find_continue_thread_links(page) -> list:
    """
    'Continue this thread ->' renders as an <a> tag pointing to a deep-linked
    permalink (e.g. /r/x/comments/postid/_/commentid/) that focuses the page
    on that single sub-thread. Collect the hrefs so we can visit each one.
    """
    links = page.locator("a:has-text('Continue this thread')")
    hrefs = []
    for i in range(links.count()):
        try:
            href = links.nth(i).get_attribute("href")
            if href:
                hrefs.append(href if href.startswith("http") else f"https://www.reddit.com{href}")
        except Exception:
            continue
    return hrefs


def scrape_continued_thread(browser, url: str, max_comments: int) -> list:
    """Opens a 'continue this thread' link in a fresh page and scrapes just
    that branch, including recursively following any further 'continue this
    thread' links inside it (deep chains can nest multiple times)."""
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.set_default_timeout(15000)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("shreddit-comment", timeout=15000)
        page.wait_for_timeout(1500)
        expand_more_comments(page, max_rounds=10)

        flat = scrape_flat_comments(page, max_comments)
        further_links = find_continue_thread_links(page)
    except PWTimeout:
        flat, further_links = [], []
    finally:
        page.close()

    return flat, further_links


def fetch_via_browser(
    url: str,
    max_comments: int,
    headless: bool = True,
    max_depth_traversals: int = 15,
) -> dict:
    validate_reddit_url(url)

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.set_default_timeout(15000)

        print("Opening Reddit post...")
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        try:
            page.wait_for_selector("shreddit-post", timeout=20000)
        except PWTimeout:
            browser.close()
            raise RuntimeError(
                "Reddit post did not render. You may be behind a verification "
                "challenge -- retry with --headed to see what's happening."
            )

        page.wait_for_timeout(3000)

        print("Expanding collapsed comment branches...")
        expand_more_comments(page)

        print("Scrolling to load lazy comments...")
        prev_count, stable_rounds = -1, 0
        for _ in range(20):
            page.mouse.wheel(0, 6000)
            page.wait_for_timeout(1200)
            expand_more_comments(page, max_rounds=3)
            current_count = page.locator("shreddit-comment").count()
            if current_count == prev_count:
                stable_rounds += 1
                if stable_rounds >= 3:
                    break
            else:
                stable_rounds = 0
            prev_count = current_count
            if current_count >= max_comments:
                break

        post = page.locator("shreddit-post").first
        title = post.get_attribute("post-title")
        author = post.get_attribute("author")
        score = post.get_attribute("score")
        post_id = post.get_attribute("id") or post.get_attribute("post-id")
        post_body = get_post_body_text(post)

        flat_comments = scrape_flat_comments(page, max_comments)

        pending_links = find_continue_thread_links(page)
        traversals_done = 0
        seen_links = set(pending_links)

        print(f"Found {len(pending_links)} 'continue this thread' branch(es) to follow.")

        while pending_links and traversals_done < max_depth_traversals and len(flat_comments) < max_comments:
            link = pending_links.pop(0)
            traversals_done += 1
            print(f"  Following deep branch {traversals_done}/{max_depth_traversals}: {link}")

            remaining_budget = max_comments - len(flat_comments)
            branch_comments, further_links = scrape_continued_thread(browser, link, remaining_budget)
            flat_comments.extend(branch_comments)

            for fl in further_links:
                if fl not in seen_links:
                    seen_links.add(fl)
                    pending_links.append(fl)

        # De-duplicate: a comment near the boundary of a "continue this
        # thread" link can appear both on the parent page and in the
        # branch page.
        deduped = {}
        for c in flat_comments:
            deduped[c["id"]] = c
        flat_comments = list(deduped.values())

        browser.close()

    comment_tree, orphan_count = build_comment_tree(flat_comments)

    return {
        "source": "reddit",
        "fetch_method": "Playwright browser (shreddit web components + continue-thread traversal)",
        "source_url": url,
        "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "post": {
            "id": post_id,
            "title": title,
            "author": author,
            "score": score,
            "text": post_body,
        },
        "comments": comment_tree,
        "comments_fetched": len(flat_comments),
        "orphaned_replies": orphan_count,
        "deep_branches_followed": traversals_done,
    }


def build_comment_tree(flat_comments: list) -> tuple:
    by_id = {}
    for c in flat_comments:
        node = dict(c)
        node["children"] = []
        by_id[node["id"]] = node

    roots = []
    orphan_count = 0

    for c in flat_comments:
        node = by_id[c["id"]]
        parent_id = c["parent_id"]

        if parent_id is None:
            roots.append(node)
        elif parent_id in by_id:
            by_id[parent_id]["children"].append(node)
        else:
            orphan_count += 1
            roots.append(node)

    return roots, orphan_count


def tree_max_depth(nodes: list, current: int = 0) -> int:
    if not nodes:
        return current
    return max(tree_max_depth(n["children"], current + 1) for n in nodes)


def main():
    parser = argparse.ArgumentParser(description="Fetch a Reddit post + nested comment tree for ABSA.")
    parser.add_argument("url", nargs="?", help="Reddit post URL")
    parser.add_argument("--max-comments", type=int, default=300)
    parser.add_argument("--max-depth-traversals", type=int, default=15,
                         help="Max 'continue this thread' links to follow (each is a full page load)")
    parser.add_argument("--headed", action="store_true", help="Show the browser window (recommended for debugging)")
    parser.add_argument("--out", default="reddit_data.json")
    args = parser.parse_args()

    url = args.url or input("Enter Reddit post URL: ").strip()

    try:
        data = fetch_via_browser(
            url,
            max_comments=args.max_comments,
            headless=not args.headed,
            max_depth_traversals=args.max_depth_traversals,
        )

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        actual_depth = tree_max_depth(data["comments"])

        print()
        print("Scraping completed!")
        print(f"Post score: {data['post']['score']} | Comments collected: {data['comments_fetched']}")
        print(f"Top-level comment threads: {len(data['comments'])}")
        print(f"Deepest reply chain captured: {actual_depth} levels")
        print(f"Deep branches followed via 'continue this thread': {data['deep_branches_followed']}")
        if data["orphaned_replies"]:
            print(f"Note: {data['orphaned_replies']} replies had no matching parent in this scrape.")
        print(f"Saved to: {args.out}")

    except Exception as e:
        print()
        print("Scraping failed:")
        print(e)


if __name__ == "__main__":
    main()
