"""Load the SNIPS 2017 NLU benchmark and adapt it to the demo pipeline.

The 7-intent SNIPS dataset is downloaded once from the public snipsco/nlu-benchmark
GitHub repo, cached under `data/snips/`, parsed into `(text, intent)` pairs, then
split with the same `stratified_split` used by the synthetic dataset. No new pip
dependency is required: downloading uses `urllib` from the standard library.

Each per-intent JSON looks like::

    {"PlayMusic": [{"data": [{"text": "Play "}, {"text": "Jazz", "entity": "genre"}]}, ...]}

so the full utterance is the concatenation of every chunk's ``text`` field.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from .config import Config, DEFAULT_CONFIG
from .data import Dataset, stratified_split

SNIPS_INTENTS = [
    "AddToPlaylist",
    "BookRestaurant",
    "GetWeather",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
]

_BASE_URL = (
    "https://raw.githubusercontent.com/snipsco/nlu-benchmark/master/"
    "2017-06-custom-intent-engines"
)
_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "snips"


def _download(intent: str) -> Path:
    """Download (and cache) the full training JSON for one SNIPS intent."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"train_{intent}_full.json"
    local = _CACHE_DIR / fname
    if not local.exists():
        url = f"{_BASE_URL}/{intent}/{fname}"
        print(f"Downloading SNIPS intent {intent} ...")
        with urllib.request.urlopen(url) as resp:
            local.write_bytes(resp.read())
    return local


def _parse(path: Path, intent: str) -> list[tuple[str, str]]:
    """Parse one cached SNIPS intent file into (text, intent) pairs."""
    raw = path.read_bytes()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        # SNIPS files ship with some latin-1 encoded accented characters.
        content = raw.decode("latin-1")
    obj = json.loads(content)
    samples: list[tuple[str, str]] = []
    for utterance in obj[intent]:
        sentence = "".join(chunk["text"] for chunk in utterance["data"]).strip()
        if sentence:
            samples.append((sentence, intent))
    return samples


def load_snips_samples(config: Config = DEFAULT_CONFIG) -> list[tuple[str, str]]:
    """Download/parse SNIPS, subsampling to `samples_per_intent` per class.

    Subsampling keeps the demo fast on CPU while staying class-balanced; raise
    `config.samples_per_intent` to train on more of the data.
    """
    rng = random.Random(config.seed)
    samples: list[tuple[str, str]] = []
    for intent in SNIPS_INTENTS:
        items = _parse(_download(intent), intent)
        rng.shuffle(items)
        samples.extend(items[: config.samples_per_intent])
    rng.shuffle(samples)
    return samples


def build_snips_dataset(config: Config = DEFAULT_CONFIG) -> Dataset:
    """Load SNIPS and split it into train/val/test in one call."""
    return stratified_split(load_snips_samples(config), config)
