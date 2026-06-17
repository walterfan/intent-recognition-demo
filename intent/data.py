"""Synthetic multi-class intent dataset generation and splitting.

The dataset is template-based: each intent owns a set of phrase templates and
slot words. Combining them yields diverse, controllably-balanced samples that
run offline on CPU. This is a teaching dataset, not a realistic corpus.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import Config, DEFAULT_CONFIG

# Each intent maps to (templates, slot_fillers). "{x}" in a template is filled
# from the slot list; templates without a slot are used as-is.
INTENT_TEMPLATES: dict[str, tuple[list[str], list[str]]] = {
    "greeting": (
        ["hello there", "hi {x}", "good {x}", "hey how are you", "greetings {x}",
         "nice to meet you", "hey there {x}", "morning {x}"],
        ["morning", "afternoon", "evening", "friend", "everyone", "doctor"],
    ),
    "goodbye": (
        ["goodbye {x}", "see you {x}", "bye for now", "talk to you later", "have a good {x}",
         "take care", "catch you {x}", "i have to go now"],
        ["soon", "tomorrow", "later", "day", "night", "everyone"],
    ),
    "ask_illness": (
        ["what are the symptoms of {x}", "i feel {x} what should i do", "is {x} serious",
         "how do i treat a {x}", "my {x} hurts a lot", "i think i have {x}",
         "should i see a doctor for {x}", "can you help me with my {x}"],
        ["flu", "a fever", "a headache", "the cold", "a sore throat", "a cough", "stomach pain"],
    ),
    "play_music": (
        ["play some {x}", "put on {x} music", "i want to listen to {x}", "play {x} for me",
         "start playing {x}", "can you play {x}", "i'm in the mood for {x}", "could you put on {x}"],
        ["jazz", "rock", "classical", "pop", "relaxing", "my favorite song"],
    ),
    "set_alarm": (
        ["set an alarm for {x}", "wake me up at {x}", "remind me at {x}", "create an alarm at {x}",
         "alarm at {x} please", "i need an alarm for {x}", "can you wake me at {x}", "set a reminder for {x}"],
        ["seven am", "eight thirty", "noon", "six in the morning", "ten pm", "five am"],
    ),
    "weather": (
        ["what is the weather {x}", "will it rain {x}", "how hot is it {x}", "is it sunny {x}",
         "weather forecast for {x}", "do i need an umbrella {x}", "is it going to be cold {x}",
         "what's the temperature {x}"],
        ["today", "tomorrow", "this weekend", "tonight", "outside", "this afternoon"],
    ),
}

# Filler words used by the optional noise injector to mimic casual speech.
_FILLERS = ["um", "uh", "please", "hey", "so", "well", "like", "okay"]


def _inject_noise(text: str, rng: random.Random, prob: float) -> str:
    """Optionally roughen a clean sentence to mimic messy real-world input.

    Two independent perturbations, each gated by `prob`: insert a filler word at
    a random position, and introduce a small adjacent-character swap (typo).
    """
    if prob <= 0:
        return text
    tokens = text.split()
    if rng.random() < prob:
        tokens.insert(rng.randint(0, len(tokens)), rng.choice(_FILLERS))
    if tokens and rng.random() < prob:
        i = rng.randint(0, len(tokens) - 1)
        w = tokens[i]
        if len(w) > 3:
            j = rng.randint(0, len(w) - 2)
            tokens[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(tokens)


@dataclass
class Dataset:
    train: list[tuple[str, str]]
    val: list[tuple[str, str]]
    test: list[tuple[str, str]]
    labels: list[str]


def generate_samples(config: Config = DEFAULT_CONFIG) -> list[tuple[str, str]]:
    """Generate labeled (text, intent) samples for every configured intent."""
    rng = random.Random(config.seed)
    samples: list[tuple[str, str]] = []
    for intent, (templates, fillers) in INTENT_TEMPLATES.items():
        seen: set[str] = set()
        attempts = 0
        # Try to produce diverse phrasings; fall back to allowing repeats only
        # if the template/filler space is exhausted.
        while len(seen) < config.samples_per_intent and attempts < config.samples_per_intent * 20:
            attempts += 1
            template = rng.choice(templates)
            text = template.format(x=rng.choice(fillers)) if "{x}" in template else template
            text = _inject_noise(text, rng, config.noise_prob)
            if text not in seen:
                seen.add(text)
                samples.append((text, intent))
        # If unique space is too small, pad by resampling to keep classes balanced.
        while sum(1 for _, lbl in samples if lbl == intent) < config.samples_per_intent:
            template = rng.choice(templates)
            text = template.format(x=rng.choice(fillers)) if "{x}" in template else template
            text = _inject_noise(text, rng, config.noise_prob)
            samples.append((text, intent))
    rng.shuffle(samples)
    return samples


def stratified_split(
    samples: list[tuple[str, str]], config: Config = DEFAULT_CONFIG
) -> Dataset:
    """Split samples into train/val/test, preserving per-class representation."""
    rng = random.Random(config.seed)
    by_label: dict[str, list[tuple[str, str]]] = {}
    for text, label in samples:
        by_label.setdefault(label, []).append((text, label))

    train: list[tuple[str, str]] = []
    val: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)
        # Guarantee at least one sample per class in every split.
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    labels = sorted(by_label)
    return Dataset(train=train, val=val, test=test, labels=labels)


def build_dataset(config: Config = DEFAULT_CONFIG) -> Dataset:
    """Generate and split a dataset in one call."""
    return stratified_split(generate_samples(config), config)


def build_dataset_by_name(name: str, config: Config = DEFAULT_CONFIG) -> Dataset:
    """Build a dataset by source name: 'synthetic' (default) or 'snips'."""
    if name == "snips":
        from .snips import build_snips_dataset  # lazy import avoids a cycle

        return build_snips_dataset(config)
    if name == "synthetic":
        return build_dataset(config)
    raise ValueError(f"Unknown dataset '{name}'. Choose 'synthetic' or 'snips'.")
