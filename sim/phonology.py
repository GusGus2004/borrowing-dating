"""Phoneme strings for simulated cognate classes, evolving by regular sound change.

Forms are strings of single-character segments from :data:`INVENTORY`
(``c`` = [tʃ], ``x`` = [x], ``j`` = [j]; ASCII keeps alignment code simple).

* **Birth.** A new class gets a random protoform of 2-3 CV syllables with an
  optional final consonant.
* **Regular sound change.** Each lineage undergoes changes as a Poisson
  process in calendar time (rate ``change_rate``). A change is a rule
  ``x > y / context`` with ``y`` a phonetically plausible outcome of ``x``
  (:data:`OUTCOMES`; ``""`` = deletion) and context one of: anywhere,
  between vowels, word-initial, word-final, before a front vowel. The rule is
  applied at that instant to *every* word the lineage carries; that
  regularity is what lets the comparative method (and the pair HMM) spot
  forms that do not fit.
* **Sporadic change.** Independently, at rate ``sporadic_rate`` per concept
  per lineage, one word of the lineage gets a one-segment substitution,
  insertion or deletion (analogy, irregular developments, noise).
* **Suffixation.** At rate ``suffix_rate`` per concept per lineage, a word
  without a suffix gains one from ``suffixes`` (Mayan-style -Vl, -Vx). Its
  cognacy is unchanged; later sound changes apply to the whole word. This
  is what makes whole-word cognate detection split true cognates.
* **Borrowing** (in :mod:`sim.borrowing`) copies the donor's current form;
  from then on only the recipient's own changes apply to it.

Targets are drawn from segments actually present in the lineage's words
(weighted by frequency), so every change does something.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

VOWELS = "aeiou"
CONSONANTS = "ptkqbdgmnsxhlrwjfvzc"
INVENTORY = CONSONANTS + VOWELS
FRONT = "ei"

# Plausible outcomes of each segment under sound change ("" = loss).
OUTCOMES: dict[str, list[str]] = {
    "p": ["b", "f", "w"], "t": ["d", "s", "c"], "k": ["g", "x", "c", "q"], "q": ["k", "x", "h"],
    "b": ["p", "v", "w", "m"], "d": ["t", "z", "r", "l"], "g": ["k", "x", "j", ""],
    "m": ["n", "b"], "n": ["m", "l"], "s": ["h", "z", "c", ""], "x": ["h", "k", ""],
    "h": ["", "x"], "l": ["r", "j", "n"], "r": ["l", "d", ""], "w": ["v", "b", "", "u"],
    "j": ["", "c", "i"], "f": ["h", "p", "v"], "v": ["b", "w", "f"], "z": ["s", "r", "d"],
    "c": ["s", "t", "k"],
    "a": ["e", "o"], "e": ["i", "a"], "i": ["e", "j"], "o": ["u", "a"], "u": ["o", "w"],
}
CONTEXTS = ("any", "intervocalic", "initial", "final", "before_front")
_V = f"[{VOWELS}]"


@dataclass(frozen=True)
class PhonologyParams:
    change_rate: float = 1 / 250   # regular sound changes per lineage per year
    sporadic_rate: float = 2e-5    # sporadic changes per concept per lineage per year
    min_syllables: int = 2
    max_syllables: int = 3
    p_final_consonant: float = 0.5
    # Segments that sound change rarely targets (weight ``stable_weight``
    # relative to 1 for the rest), family-wide. Real families have such
    # segments, which leaves short roots made of them nearly unchanged
    # everywhere (e.g. Mayan *ha'* "water", *uk'* "louse").
    stable_segments: str = ""
    stable_weight: float = 1.0
    suffix_rate: float = 0.0       # suffix gains per concept per lineage per year
    suffixes: tuple = ("il", "al", "el", "ax")

    def __post_init__(self) -> None:
        if self.change_rate < 0 or self.sporadic_rate < 0:
            raise ValueError("rates must be >= 0")
        if not set(self.stable_segments) <= set(INVENTORY):
            raise ValueError("stable_segments must be in INVENTORY")
        if not 0 <= self.stable_weight <= 1:
            raise ValueError("stable_weight must be in [0, 1]")
        if not 1 <= self.min_syllables <= self.max_syllables:
            raise ValueError("need 1 <= min_syllables <= max_syllables")


@dataclass(frozen=True)
class SoundChange:
    target: str
    output: str
    context: str

    def pattern(self) -> re.Pattern:
        x = re.escape(self.target)
        return re.compile({
            "any": x,
            "intervocalic": f"(?<={_V}){x}(?={_V})",
            "initial": f"^{x}",
            "final": f"{x}$",
            "before_front": f"{x}(?=[{FRONT}])",
        }[self.context])

    def apply(self, form: str) -> str:
        out = self.pattern().sub(self.output, form)
        return out if out else form  # never delete a whole word

    def __str__(self) -> str:
        ctx = {"any": "", "intervocalic": " / V_V", "initial": " / #_", "final": " / _#",
               "before_front": " / _[e,i]"}[self.context]
        return f"{self.target} > {self.output or '∅'}{ctx}"


def random_protoform(p: PhonologyParams, rng: np.random.Generator) -> str:
    n = int(rng.integers(p.min_syllables, p.max_syllables + 1))
    s = "".join(CONSONANTS[rng.integers(len(CONSONANTS))] + VOWELS[rng.integers(len(VOWELS))]
                for _ in range(n))
    if rng.random() < p.p_final_consonant:
        s += CONSONANTS[rng.integers(len(CONSONANTS))]
    return s


def draw_sound_change(forms: list[str], rng: np.random.Generator,
                      p: Optional[PhonologyParams] = None) -> Optional[SoundChange]:
    """A rule whose target occurs in ``forms`` (None if there are no forms).

    Targets are drawn by frequency, down-weighted for ``p.stable_segments``.
    """
    text = "".join(forms)
    if not text:
        return None
    segs, counts = np.unique(list(text), return_counts=True)
    w = counts.astype(float)
    if p is not None and p.stable_segments:
        w *= np.where(np.isin(segs, list(p.stable_segments)), p.stable_weight, 1.0)
    if w.sum() == 0:
        return None
    for _ in range(20):  # retry until the rule matches at least one word
        x = str(segs[rng.choice(len(segs), p=w / w.sum())])
        y = OUTCOMES[x][rng.integers(len(OUTCOMES[x]))]
        ctx = CONTEXTS[rng.integers(len(CONTEXTS))]
        if y == "" and x in VOWELS:
            continue
        rule = SoundChange(x, y, ctx)
        if any(rule.apply(f) != f for f in forms):
            return rule
    return None


def sporadic_change(form: str, rng: np.random.Generator) -> str:
    kind = rng.integers(3) if len(form) > 2 else rng.integers(2)
    i = int(rng.integers(len(form)))
    if kind == 0:  # substitution by a plausible outcome
        outs = [o for o in OUTCOMES[form[i]] if o]
        return form[:i] + outs[rng.integers(len(outs))] + form[i + 1:]
    if kind == 1:  # insertion
        seg = INVENTORY[rng.integers(len(INVENTORY))]
        return form[:i] + seg + form[i:]
    return form[:i] + form[i + 1:]  # deletion


class LineagePhonology:
    """Pre-drawn schedule of phonological events on one branch.

    Events are (age, kind) with kind "rule" or "sporadic", ages descending
    from the top of the branch. Drawing the schedule up front from a separate
    RNG keeps the cognate process's random stream untouched.
    """

    def __init__(self, top_age: float, bottom_age: float, p: PhonologyParams, n_concepts: int,
                 rng: np.random.Generator):
        span = top_age - bottom_age
        n_rule = rng.poisson(p.change_rate * span)
        n_spor = rng.poisson(p.sporadic_rate * n_concepts * span)
        ages = np.concatenate([top_age - rng.random(n_rule) * span,
                               top_age - rng.random(n_spor) * span])
        kinds = ["rule"] * n_rule + ["sporadic"] * n_spor
        if p.suffix_rate > 0:  # drawn only when enabled, so default runs are unchanged
            n_suf = rng.poisson(p.suffix_rate * n_concepts * span)
            ages = np.concatenate([ages, top_age - rng.random(n_suf) * span])
            kinds += ["suffix"] * n_suf
        order = np.argsort(-ages)
        self.events = [(float(ages[i]), kinds[i]) for i in order]
        self.pos = 0
        self.p = p
        self.rules: list[tuple[float, str]] = []

    def advance(self, to_age: float, forms: dict[int, str], rng: np.random.Generator) -> None:
        """Apply every scheduled event with age >= ``to_age`` to ``forms`` in place."""
        while self.pos < len(self.events) and self.events[self.pos][0] >= to_age:
            age, kind = self.events[self.pos]
            self.pos += 1
            if not forms:
                continue
            keys = sorted(forms)
            if kind == "rule":
                rule = draw_sound_change([forms[k] for k in keys], rng, self.p)
                if rule is None:
                    continue
                for k in keys:
                    forms[k] = rule.apply(forms[k])
                self.rules.append((age, str(rule)))
            elif kind == "suffix":
                k = keys[int(rng.integers(len(keys)))]
                if not any(forms[k].endswith(x) for x in self.p.suffixes):
                    forms[k] += self.p.suffixes[int(rng.integers(len(self.p.suffixes)))]
            else:
                k = keys[int(rng.integers(len(keys)))]
                forms[k] = sporadic_change(forms[k], rng)
