#!/usr/bin/env python3
"""Write ``conversion/cheung-ediv.txt``: the display profile for Cheung's modern-language forms.

Run from the git root after ``import_source.py --install``. The profile lists every grapheme
cluster that occurs in the installed modern-language rows (``MODERN`` languages), so the
build never meets an unprofiled symbol. Old and Middle Iranian rows keep their philological
transliteration and are not routed to any profile (see the YAML and README).

Consonants follow the Iranic batch plan (``source_checklists/20261025-iranic-extension.md``)
and ``conversion/cdial.txt``: š→ś ž→ź č→c ǰ→j c→ʦ ʒ→ʣ δ→ð ϑ→θ γ→ɣ x̌→χ w→v, plus the
Central-dialect IPA of Lecoq's notation (ʃ→ś, tʃ→c, dʒ→j, ``V:`` length → macron). Printed
vowel quality and accents are kept; ʌ→a per ``profile_policy``; ы→ɨ (Wakhi ы is the central
high vowel). Typographic variants with a breve instead of a caron (x̆, γ̆) are read as x̌, γ̌
(flagged for review in proposals.md). The Ossetic/Pashto glottalisation apostrophe becomes ʼ.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

import regex

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FORMS = ROOT / "data/other/forms/20261025-cheung-ediv.csv"
PROFILE = ROOT / "conversion/cheung-ediv.txt"

OLD_MIDDLE = {"Av", "OPers", "MPrs", "Parth", "Khot", "Tumsh", "Sogd", "Chvar", "Bactr"}

BASE = {
    "š": "ś", "ž": "ź", "č": "c", "ǰ": "j", "c": "ʦ", "ʒ": "ʣ", "δ": "ð", "ϑ": "θ",
    "γ": "ɣ", "w": "v", "ʃ": "ś", "ʌ": "a", "ы": "ɨ", "’": "ʼ", "ɫ": "ɫ", "ġ": "ɣ",
}
CLUSTER = {
    "x̌": "χ", "x̆": "χ", "γ̌": "ɣ̌", "γ̆": "ɣ̌", "ṣ̌": "ṣ̌", "ẓ̌": "ẓ̌", "č̣": "ʦ̣",
}
MULTI = {"tʃ": "c", "dʒ": "j"}
# length is a macron on the vowel (profile_policy): ä: → ā̈, å: → ā̊, ö: → ō̈, ü: → ṻ
LENGTH = {"a": "ā", "e": "ē", "i": "ī", "o": "ō", "u": "ū", "ɛ": "ɛ̄", "ɔ": "ɔ̄", "å": "ā̊",
          "ä": "ā̈", "æ": "ǣ", "ə": "ə̄", "ö": "ō̈", "ü": "ṻ"}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def convert_cluster(cluster: str) -> str:
    """Map the longest mapped prefix (base letter plus its own marks), keep further marks."""
    decomposed = unicodedata.normalize("NFD", cluster)
    table = {unicodedata.normalize("NFD", k): v for k, v in {**BASE, **CLUSTER}.items()}
    for k in range(len(decomposed), 0, -1):
        prefix = decomposed[:k]
        if prefix in table:
            return nfc(unicodedata.normalize("NFD", table[prefix]) + decomposed[k:])
    return nfc(decomposed)


def modern_forms():
    with FORMS.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if row[0] not in OLD_MIDDLE:
                yield nfc(row[2])


def graphemes(forms) -> set[str]:
    out = set()
    for form in forms:
        for part in form.split(","):
            for g in regex.findall(r"\X", part.strip()):
                if not g.isspace():
                    out.add(nfc(g))
    return out


def rules(inventory: set[str]) -> dict[str, str]:
    table = {g: convert_cluster(g) for g in inventory}
    for seq, out in MULTI.items():
        table[seq] = out
    for vowel, long in LENGTH.items():
        table[f"{vowel}:"] = long
    table[":"] = ":"
    return table


def main() -> None:
    table = rules(graphemes(modern_forms()))
    lines = ["Grapheme\tIPA"]
    for g in sorted(table, key=lambda k: (-len(k), k)):
        lines.append(f"{g}\t{table[g]}")
    PROFILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(table)} profile rules to {PROFILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
