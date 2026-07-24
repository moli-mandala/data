"""Apply a conservative second pass of Backstrom etymologies.

This pass permits small phonological differences rather than requiring exact
forms.  Candidates must have the same gloss, a clearly best already-curated
Backstrom etymon, and a broad-normalized edit distance of at most 25%.  A small
audited allowlist adds transparent loans and dialect correspondences just beyond
that threshold.  Known lookalikes and inherited Iranian cognates that would be
misrepresented as Indo-Aryan borrowings are explicitly excluded.

The output review CSV is deliberately persistent: an idempotent rerun does not
erase the evidence from the original linking pass.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
FORMS = HERE.parent / "20230416-northern.csv"
REVIEW = HERE / "northern_nontrivial_etymologies.csv"

# Source rows rejected after inspection.  These include accidental phonetic
# neighbours and inherited Iranian cognates that must not become IA "loans".
EXCLUDED_ROWS = {
    20,    # Pashto jiba 'tongue': inherited Iranian, not < OIA jihva
    139,   # Pashto pana 'leaf': not the pattra etymon selected by distance
    374,   # Pashto kala 'when': false match to kalayati
    694,   # Balti grong 'village': native Tibetan, not < grama
    233,   # 'older brother': shared adjective, but a different head noun
    4374, 4375, 4376,  # Wakhi spouse forms: relationship to jamatr is uncertain
    4633, 4634,  # Wakhi bist 'twenty': inherited Iranian, not an IA borrowing
    10760,  # 'you give!': shared pronoun, but a different verb
}

# Audited loans/correspondences with a broad edit distance between 25% and 34%.
ADDITIONAL_ROWS = {
    # Urdu
    13, 51, 72, 86, 98, 99, 165, 206, 220, 279, 284, 350, 356, 362, 366,
    # Balti/Purki loanword 'hammer'
    747, 748, 749, 750, 751, 752, 753,
    # Burushaski and Wakhi loans, plus close variants
    2209, 2275, 2438,
    2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598,
    2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616,
    2662, 2663, 2664,
    2706, 2707, 2708, 2709, 2710, 2711, 2712,
    2873, 3986, 4166, 4168, 4174, 4198, 4199, 4248,
    4482, 4483, 4484, 4925,
    # Shina dialect correspondences
    6010,
    7895, 7896, 7897, 7898, 7900, 7901, 7905, 7916, 7918, 7920, 7921,
    7922, 7924, 7925, 7928,
    8609, 8629,
    8800, 8806, 8810, 8814, 8816, 8821, 8823,
}

TRANSLATION = str.maketrans(
    {
        "ˈ": "", "ˌ": "", " ": "", "+": "", "-": "", "͡": "",
        "ʌ": "ə", "ɜ": "ə", "ɪ": "i", "ʊ": "u", "ɑ": "a",
        "ɛ": "e", "ɔ": "o", "j": "y", "β": "w", "ɾ": "r",
        "ɽ": "r", "ʈ": "t", "ɖ": "d", "ɕ": "ʃ", "ʂ": "ʃ",
        "ʐ": "ʒ", "ʰ": "h", "ː": "",
    }
)


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFD", value.strip().lower())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return value.translate(TRANSLATION)


def distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def tokens(value: str) -> list[str]:
    return [normalized(part) for part in re.split(r"[ +\-]+", value) if normalized(part)]


def main() -> None:
    with FORMS.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    # Do not let accepted rows recursively create progressively weaker matches
    # on later runs.  The review file identifies this pass's derived evidence.
    reviewed_rows: set[int] = set()
    if REVIEW.exists():
        with REVIEW.open(encoding="utf-8") as handle:
            reviewed_rows = {
                int(row["Row"]) for row in csv.DictReader(handle) if row.get("Row")
            }

    evidence_by_gloss: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    token_evidence: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for row_number, row in enumerate(rows, 1):
        if row[1] and row_number not in reviewed_rows:
            gloss = row[3].strip().lower()
            evidence_by_gloss[gloss].append(
                (row[1], normalized(row[5] or row[2]), row[0], row[2])
            )
            for token in tokens(row[5] or row[2]):
                token_evidence[(gloss, token)].append((row[1], row[0], row[2]))

    changes: list[list[str]] = []
    for row_number, row in enumerate(rows, 1):
        if row[1] or row_number in EXCLUDED_ROWS:
            continue
        target = normalized(row[5] or row[2])
        evidence = evidence_by_gloss.get(row[3].strip().lower(), [])
        if not target or not evidence:
            continue

        best_by_etymon: dict[str, tuple[float, str, str]] = {}
        for parameter_id, candidate, language, form in evidence:
            score = distance(target, candidate) / max(len(target), len(candidate), 1)
            if parameter_id not in best_by_etymon or score < best_by_etymon[parameter_id][0]:
                best_by_etymon[parameter_id] = (score, language, form)
        ranked = sorted(
            (score, parameter_id, language, form)
            for parameter_id, (score, language, form) in best_by_etymon.items()
        )
        best_score, parameter_id, evidence_language, evidence_form = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 1.0

        automatic = (
            best_score <= 0.25
            and second_score - best_score >= 0.25
            and (len(target) >= 4 or best_score == 0)
        )
        audited = row_number in ADDITIONAL_ROWS and best_score <= 0.34
        token_matches = [
            (token, *match)
            for token in tokens(row[5] or row[2])
            if len(token) >= 3
            for match in token_evidence.get((row[3].strip().lower(), token), [])
        ]
        token_etymologies = {match[1] for match in token_matches}
        lexical_component = len(token_etymologies) == 1
        if not (automatic or audited or lexical_component):
            continue

        if lexical_component and not (automatic or audited):
            _, parameter_id, evidence_language, evidence_form = token_matches[0]

        row[1] = parameter_id
        changes.append(
            [
                str(row_number), row[0], row[2], row[3], row[5], parameter_id,
                f"{best_score:.3f}", evidence_language, evidence_form,
                "audited close correspondence" if audited and not automatic
                else "shared lexical component" if lexical_component and not automatic
                else "unique close phonological match",
            ]
        )

    with FORMS.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    if changes or not REVIEW.exists():
        previous_review: list[list[str]] = []
        if changes and REVIEW.exists():
            with REVIEW.open(encoding="utf-8") as handle:
                previous_review = list(csv.reader(handle))[1:]
        with REVIEW.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Row", "Language_ID", "Form", "Gloss", "Phonemic",
                    "Parameter_ID", "Distance", "Evidence_Language_ID",
                    "Evidence_Form", "Basis",
                ]
            )
            writer.writerows(previous_review + changes)

    print(f"Linked {len(changes)} nontrivial forms in {FORMS}")
    if changes:
        print(f"Wrote review details to {REVIEW}")
    else:
        print(f"No changes; preserved existing review details in {REVIEW}")


if __name__ == "__main__":
    main()
