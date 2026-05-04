# CS2 Wiki Corpus Builder

Scripts that scrape the [Counter-Strike 2 fandom wiki](https://counterstrike.fandom.com/wiki/Counter-Strike_2)
and turn it into a small CS2 corpus.

## Files

- `fetch.py` — downloads the root CS2 page plus internal-link concept pages into `data/wiki/raw_html/`.
- `parse.py` — parses the HTML into raw text, concepts, bigrams, token frequencies.
- `stopwords.py` — English stopword list used to filter common words (`a`, `and`, `or`, `the`, ...).
- `run.py` — convenience driver that runs fetch then parse.

## Usage

```bash
pip install requests beautifulsoup4

# from repo root
python src/scripts/parse_wiki/run.py
```

Or run each step:

```bash
python src/scripts/parse_wiki/fetch.py --out data/wiki/raw_html --max-pages 80
python src/scripts/parse_wiki/parse.py --in data/wiki/raw_html --out data/wiki
```

## Outputs (under `data/wiki/`)

- `raw_html/<slug>.html` — fetched HTML, one per wiki page.
- `raw_text/<slug>.txt` — cleaned raw text per page.
- `pages/<slug>.json` — per-page record: title, headings, tokens, concepts, bigrams.
- `corpus.txt` — concatenated raw text for all pages.
- `concepts.json` — aggregated key concepts (anchor texts + page titles), sorted by frequency.
- `bigrams.json` — aggregated bigrams of consecutive non-stopword tokens, sorted by frequency.
- `tokens.json` — token frequencies after stopword filtering.
- `pages_index.json` — index of fetched pages with token/concept/bigram counts.
- `stopwords.json` — snapshot of the stopword list used.
