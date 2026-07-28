"""
Prompt template bank for the semantic-drift / concept-ranking probe.

The published analysis used a single template (``this video shows "[concept]"``).
Reviewer HHbf asked how sensitive the ranking trajectories are to that choice, so
this module enumerates a bank of 43 singleton templates in 7 systematic families
plus 4 mean-pooled ensembles.

Families are enumerated systematically (not hand-picked) so the bank is not open to
a selection objection. Every template is stamped with its family letter in the
output CSVs via TEMPLATE_FAMILY.

Note that get_text_embeddings() lowercases its input before tokenization, so casing
differences between templates are not a confound.
"""

# ---------------------------------------------------------------------------
# Singleton templates, grouped into families. "{concept}" is the substitution slot.
# ---------------------------------------------------------------------------

PROMPT_FAMILIES: dict[str, dict[str, str]] = {
    # A. bare / minimal
    "A": {
        "bare": "{concept}",
        "bare_period": "{concept}.",
        "bare_article": "a {concept}",
    },
    # B. generic visual, CLIP-style engineered prompts
    "B": {
        "photo": "a photo of {concept}",
        "photo_article": "a photo of a {concept}",
        "image": "an image of {concept}",
        "picture": "a picture of {concept}",
        "screenshot": "a screenshot of {concept}",
        "photo_cropped": "a cropped photo of {concept}",
        "photo_lowres": "a low resolution photo of {concept}",
        "photo_bright": "a bright photo of {concept}",
    },
    # C. video / temporal. "paper" is the template used in the submission.
    "C": {
        "video": "a video of {concept}",
        "video_showing": "a video showing {concept}",
        "this_video_shows": "this video shows {concept}",
        "paper": 'this video shows "{concept}"',
        "video_frame": "a video frame of {concept}",
        "clip": "a clip of {concept}",
        "footage": "footage of {concept}",
        "recording": "a recording of {concept}",
    },
    # D. gameplay, game-generic
    "D": {
        "gameplay": "gameplay showing {concept}",
        "gameplay_screenshot": "a gameplay screenshot of {concept}",
        "videogame_screenshot": "a video game screenshot of {concept}",
        "ingame_footage": "in-game footage of {concept}",
        "fp_game_view": "a first person game view of {concept}",
        "game_scene": "a game scene with {concept}",
        "esports_footage": "esports footage of {concept}",
        "fps_screenshot": "a first person shooter screenshot showing {concept}",
    },
    # E. Counter-Strike specific. The first three are verbatim from
    # contra_cluster_v2/analyze_contrastive_space.py::make_prompt_variants so the two
    # pipelines stay comparable.
    "E": {
        "cs_screenshot": "a Counter-Strike gameplay screenshot showing {concept}",
        "cs_fp_scene": "a first person Counter-Strike scene with {concept}",
        "cs_footage": "gameplay footage of {concept}",
        "cs2_match": "a Counter-Strike 2 match showing {concept}",
        "cs2_round": "a CS2 round with {concept}",
        "cs_pro_match": "a professional Counter-Strike match showing {concept}",
    },
    # F. caption / declarative
    "F": {
        "there_is": "there is {concept}",
        "scene_contains": "the scene contains {concept}",
        "this_is": "this is {concept}",
        "example_of": "an example of {concept}",
        "is_visible": "{concept} is visible",
        "showing": "showing {concept}",
    },
    # G. instruction / interrogative
    "G": {
        "what_happening": "what is happening: {concept}",
        "describe_scene": "describe the scene: {concept}",
        "caption": "caption: {concept}",
        "scene_description": "scene description: {concept}",
    },
}

FAMILY_LABELS: dict[str, str] = {
    "A": "bare/minimal",
    "B": "generic visual",
    "C": "video/temporal",
    "D": "gameplay generic",
    "E": "CS2-specific",
    "F": "caption/declarative",
    "G": "instruction/interrogative",
    "ENS": "ensemble",
}

# Flattened key -> format string, and the reverse key -> family letter.
PROMPT_TEMPLATES: dict[str, str] = {}
TEMPLATE_FAMILY: dict[str, str] = {}
for _family, _templates in PROMPT_FAMILIES.items():
    for _key, _fmt in _templates.items():
        if _key in PROMPT_TEMPLATES:
            raise ValueError(f"Duplicate template key: {_key}")
        PROMPT_TEMPLATES[_key] = _fmt
        TEMPLATE_FAMILY[_key] = _family

# Singleton keys in a stable, family-ordered sequence.
SINGLETON_TEMPLATE_KEYS: list[str] = list(PROMPT_TEMPLATES.keys())

# ---------------------------------------------------------------------------
# Ensembles: each concept is embedded under every variant, then mean-pooled and
# re-normalized (see language_utils.get_text_embeddings_ensemble).
# ---------------------------------------------------------------------------

PROMPT_ENSEMBLES: dict[str, list[str]] = {
    # The 4-variant ensemble already used by contra_cluster_v2.
    "ens_cs4": [
        "{concept}",
        "a Counter-Strike gameplay screenshot showing {concept}",
        "a first person Counter-Strike scene with {concept}",
        "gameplay footage of {concept}",
    ],
    "ens_generic8": list(PROMPT_FAMILIES["B"].values()),
    "ens_video8": list(PROMPT_FAMILIES["C"].values()),
    "ens_all": list(PROMPT_TEMPLATES.values()),
}

for _key in PROMPT_ENSEMBLES:
    TEMPLATE_FAMILY[_key] = "ENS"

ENSEMBLE_TEMPLATE_KEYS: list[str] = list(PROMPT_ENSEMBLES.keys())
ALL_TEMPLATE_KEYS: list[str] = SINGLETON_TEMPLATE_KEYS + ENSEMBLE_TEMPLATE_KEYS

# ---------------------------------------------------------------------------
# Named subsets and back-compat aliases.
# ---------------------------------------------------------------------------

# The five templates named in the rebuttal draft. Reported in the legacy
# ranking_summary.txt layout as a labelled subset of the full bank.
REBUTTAL_TEMPLATES: list[str] = ["bare", "paper", "screenshot", "video", "gameplay"]

# The two modes the script shipped with. Kept so the existing artifact layout
# (sample_aggregate/{direct,prompted}/) and the two-panel Figure 5 keep working.
TEMPLATE_ALIASES: dict[str, str] = {
    "direct": "bare",
    "prompted": "paper",
}

LEGACY_MODE_KEYS: dict[str, str] = {
    "direct": "bare",
    "prompted": "paper",
}

# Canonical key -> legacy artifact subdirectory name, so runs of the two published
# arms land in the same paths as before and stay diffable against old artifacts.
LEGACY_DIR_NAMES: dict[str, str] = {v: k for k, v in LEGACY_MODE_KEYS.items()}

# Human-readable descriptions used in the ranking_summary.txt header. The two legacy
# strings are reproduced exactly so old and new summary files stay diffable.
LEGACY_MODE_DESCRIPTIONS: dict[str, str] = {
    "bare": "Direct term comparison",
    "paper": "Prompted: 'this video shows \"xxx\"'",
}


def resolve_template_key(key: str) -> str:
    """Map a user-supplied template name (incl. legacy aliases) to a canonical key."""
    canonical = TEMPLATE_ALIASES.get(key, key)
    if canonical not in PROMPT_TEMPLATES and canonical not in PROMPT_ENSEMBLES:
        raise KeyError(
            f"Unknown template '{key}'. Known: {', '.join(ALL_TEMPLATE_KEYS)} "
            f"(aliases: {', '.join(TEMPLATE_ALIASES)})"
        )
    return canonical


def is_ensemble(key: str) -> bool:
    return key in PROMPT_ENSEMBLES


def build_texts(key: str, concepts: list[str]) -> list[str] | list[list[str]]:
    """
    Render a template over a concept list.

    Returns a flat list of strings for singleton templates, or a list of
    per-concept variant lists for ensembles.
    """
    key = resolve_template_key(key)
    if is_ensemble(key):
        variants = PROMPT_ENSEMBLES[key]
        return [[fmt.format(concept=c) for c in concepts] for fmt in variants]
    fmt = PROMPT_TEMPLATES[key]
    return [fmt.format(concept=c) for c in concepts]


def describe_template(key: str) -> str:
    """One-line human-readable description for summary headers."""
    key = resolve_template_key(key)
    if key in LEGACY_MODE_DESCRIPTIONS:
        return LEGACY_MODE_DESCRIPTIONS[key]
    if is_ensemble(key):
        return f"Ensemble of {len(PROMPT_ENSEMBLES[key])} variants ({key})"
    return f"Template: '{PROMPT_TEMPLATES[key].format(concept='xxx')}'"


if __name__ == "__main__":
    print(f"Singleton templates: {len(SINGLETON_TEMPLATE_KEYS)}")
    for family, templates in PROMPT_FAMILIES.items():
        print(f"  {family} ({FAMILY_LABELS[family]}): {len(templates)}")
    print(f"Ensembles: {len(ENSEMBLE_TEMPLATE_KEYS)} -> {ENSEMBLE_TEMPLATE_KEYS}")
    print(f"Total arms: {len(ALL_TEMPLATE_KEYS)}")
    print("\nRendered examples (concept='smoke grenade'):")
    for key in REBUTTAL_TEMPLATES:
        print(f"  {key:<12} {build_texts(key, ['smoke grenade'])[0]!r}")
