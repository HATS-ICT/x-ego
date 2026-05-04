"""Parse fetched CS2 wiki HTML into raw text, concepts, and bigrams."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

from stopwords import STOPWORDS

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")


def clean_html_to_text(html: str) -> tuple[str, str, list[str]]:
    """Return (title, body_text, headings) extracted from a fandom article page."""
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.find(id="firstHeading") or soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""

    content = soup.find("div", class_="mw-parser-output")
    if content is None:
        return title, "", []

    # Strip non-content noise.
    drop_selectors = [
        "table.infobox",
        "table.navbox",
        "div.navbox",
        "div.toc",
        "sup.reference",
        "div.references",
        "div.thumb",
        "figure",
        "style",
        "script",
        "div.notice",
        "div.hatnote",
    ]
    for sel in drop_selectors:
        for el in content.select(sel):
            el.decompose()

    headings = [
        h.get_text(" ", strip=True)
        for h in content.find_all(["h2", "h3", "h4"])
        if h.get_text(strip=True)
    ]

    paragraphs = []
    for el in content.find_all(["p", "li"]):
        text = el.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)

    body = "\n".join(paragraphs)
    body = re.sub(r"\[\d+\]", "", body)  # citation markers
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{2,}", "\n\n", body).strip()
    return title, body, headings


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def filter_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def extract_concepts(html: str) -> list[str]:
    """Concepts = internal-link anchor texts + section headings."""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output") or soup
    concepts: list[str] = []

    for a in content.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/wiki/"):
            continue
        slug = href[len("/wiki/") :].split("#", 1)[0]
        if not slug or ":" in slug.split("/", 1)[0]:
            continue
        text = a.get_text(" ", strip=True)
        if text and len(text) <= 80:
            concepts.append(text)

    title_el = soup.find(id="firstHeading")
    if title_el:
        concepts.append(title_el.get_text(strip=True))

    return concepts


def bigrams_from_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    """Bigrams of consecutive non-stopword tokens."""
    out: list[tuple[str, str]] = []
    for a, b in zip(tokens, tokens[1:]):
        if a in STOPWORDS or b in STOPWORDS:
            continue
        if len(a) <= 1 or len(b) <= 1:
            continue
        out.append((a, b))
    return out


def process_file(path: Path) -> dict:
    html = path.read_text(encoding="utf-8", errors="ignore")
    title, body, headings = clean_html_to_text(html)
    raw_tokens = tokenize(body)
    filtered = filter_tokens(raw_tokens)
    bigrams = bigrams_from_tokens(raw_tokens)
    concepts = extract_concepts(html)

    return {
        "slug": path.stem,
        "title": title,
        "headings": headings,
        "raw_text": body,
        "tokens": filtered,
        "concepts": concepts,
        "bigrams": [" ".join(bg) for bg in bigrams],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="in_dir",
        type=Path,
        default=Path("data/wiki/raw_html"),
        help="Directory of fetched HTML files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/wiki"),
        help="Output directory for parsed corpus.",
    )
    parser.add_argument("--top-bigrams", type=int, default=2000)
    parser.add_argument("--top-concepts", type=int, default=2000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pages_dir = args.out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    raw_text_dir = args.out / "raw_text"
    raw_text_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(args.in_dir.glob("*.html"))
    if not html_files:
        raise SystemExit(f"No HTML files in {args.in_dir}; run fetch.py first.")

    token_counter: Counter[str] = Counter()
    bigram_counter: Counter[str] = Counter()
    concept_counter: Counter[str] = Counter()
    all_pages_meta = []

    combined_text_chunks: list[str] = []

    for path in html_files:
        rec = process_file(path)
        # per-page json
        (pages_dir / f"{rec['slug']}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # per-page raw text
        (raw_text_dir / f"{rec['slug']}.txt").write_text(
            f"# {rec['title']}\n\n{rec['raw_text']}\n", encoding="utf-8"
        )
        if rec["raw_text"]:
            combined_text_chunks.append(f"# {rec['title']}\n\n{rec['raw_text']}")

        token_counter.update(rec["tokens"])
        bigram_counter.update(rec["bigrams"])
        concept_counter.update(rec["concepts"])

        all_pages_meta.append(
            {
                "slug": rec["slug"],
                "title": rec["title"],
                "n_tokens": len(rec["tokens"]),
                "n_concepts": len(rec["concepts"]),
                "n_bigrams": len(rec["bigrams"]),
            }
        )
        print(f"parsed {rec['slug']}: {len(rec['tokens'])} tokens")

    # Combined raw text corpus
    (args.out / "corpus.txt").write_text(
        "\n\n".join(combined_text_chunks), encoding="utf-8"
    )

    # Aggregated concepts (sorted by frequency)
    concepts_sorted = concept_counter.most_common(args.top_concepts)
    (args.out / "concepts.json").write_text(
        json.dumps(
            [{"concept": c, "count": n} for c, n in concepts_sorted],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Aggregated bigrams
    bigrams_sorted = bigram_counter.most_common(args.top_bigrams)
    (args.out / "bigrams.json").write_text(
        json.dumps(
            [{"bigram": b, "count": n} for b, n in bigrams_sorted],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Token frequency (post stopword filter)
    (args.out / "tokens.json").write_text(
        json.dumps(
            [{"token": t, "count": n} for t, n in token_counter.most_common()],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (args.out / "pages_index.json").write_text(
        json.dumps(all_pages_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Stopword list snapshot for transparency
    (args.out / "stopwords.json").write_text(
        json.dumps(sorted(STOPWORDS), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nDone. {len(html_files)} pages, "
        f"{sum(token_counter.values())} tokens, "
        f"{len(bigram_counter)} unique bigrams, "
        f"{len(concept_counter)} unique concepts."
    )
    print(f"Outputs in {args.out}/")


if __name__ == "__main__":
    main()
