"""Fetch the Counter-Strike 2 fandom wiki page and related concept pages.

Uses the MediaWiki action=parse API instead of scraping HTML directly,
because fandom's Cloudflare layer 403s ordinary requests.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

WIKI_BASE = "https://counterstrike.fandom.com"
API_URL = f"{WIKI_BASE}/api.php"
ROOT_PAGE = "Counter-Strike_2"
USER_AGENT = "Mozilla/5.0 (compatible; x-ego-cs2-corpus/1.0)"


def api_parse(page: str, timeout: int = 30) -> dict:
    """Call MediaWiki action=parse and return the parsed dict."""
    params = {
        "action": "parse",
        "page": page,
        "format": "json",
        "prop": "text|links|sections|displaytitle",
        "redirects": 1,
    }
    resp = requests.get(
        API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"API error for {page}: {data['error']}")
    return data["parse"]


def slug_from_title(title: str) -> str:
    return title.replace(" ", "_").replace("/", "_")


def slug_from_url(url: str) -> str:
    path = urlparse(url).path
    if path.startswith("/wiki/"):
        return path[len("/wiki/") :]
    return path.strip("/").replace("/", "_")


def extract_concept_pages(parsed: dict, max_links: int) -> list[str]:
    """Pull internal-link page titles from the API response.

    The 'links' field already lists wiki-internal links with namespace info.
    Falls back to scanning anchor tags in the HTML for completeness.
    """
    seen: set[str] = set()
    pages: list[str] = []

    for link in parsed.get("links", []):
        # only main-namespace articles (ns == 0); existing pages have 'exists' key
        if link.get("ns") != 0:
            continue
        if "exists" not in link:
            continue
        title = link["*"]
        slug = slug_from_title(title)
        if slug in seen:
            continue
        seen.add(slug)
        pages.append(title)
        if len(pages) >= max_links:
            return pages

    # fallback: parse anchors from the rendered HTML
    html = parsed.get("text", {}).get("*", "")
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/wiki/"):
            continue
        slug = href[len("/wiki/") :].split("#", 1)[0]
        if not slug or ":" in slug.split("/", 1)[0]:
            continue
        if slug in seen:
            continue
        seen.add(slug)
        pages.append(slug.replace("_", " "))
        if len(pages) >= max_links:
            break
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/wiki/raw_html"))
    parser.add_argument("--max-pages", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Fetching root page via API: {ROOT_PAGE}")
    root = api_parse(ROOT_PAGE)
    root_html = root.get("text", {}).get("*", "")
    root_path = args.out / f"{ROOT_PAGE}.html"
    root_path.write_text(root_html, encoding="utf-8")
    print(f"  saved {root_path} ({len(root_html)} bytes)")

    pages = extract_concept_pages(root, args.max_pages)
    print(f"Found {len(pages)} concept pages to follow.")

    for i, title in enumerate(pages, 1):
        slug = slug_from_title(title)
        out_path = args.out / f"{slug}.html"
        if out_path.exists():
            print(f"[{i}/{len(pages)}] cached {slug}")
            continue
        try:
            parsed = api_parse(title)
        except (requests.RequestException, RuntimeError) as exc:
            print(f"[{i}/{len(pages)}] FAILED {slug}: {exc}")
            continue
        html = parsed.get("text", {}).get("*", "")
        if not html:
            print(f"[{i}/{len(pages)}] empty {slug}")
            continue
        out_path.write_text(html, encoding="utf-8")
        print(f"[{i}/{len(pages)}] saved {slug} ({len(html)} bytes)")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
