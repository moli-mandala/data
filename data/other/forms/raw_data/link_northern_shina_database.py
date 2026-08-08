"""Review Backstrom Shina forms against all linked Shinaic database forms.

Run this after building the unified ``cldf/forms.csv``.  With no arguments the
script regenerates an editable review CSV.  Set ``Decision`` to ``yes`` for
accepted rows, then run with ``--apply`` to copy those Parameter_ID values into
the Backstrom source CSV.  Evidence must be attested outside Backstrom and must
have the same complete gloss/sense; shared words inside longer glosses are not
enough (for example ``she`` must not match ``she-goat``).
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3]
RAW_FORMS = HERE.parent / "20230416-northern.csv"
FORMS = DATA / "cldf/forms.csv"
LANGUAGES = DATA / "cldf/languages.csv"
REVIEW = HERE / "northern_shina_database_etymologies.csv"

# Audited after candidate generation.  Keeping this explicit makes the accepted
# batch reproducible while leaving Decision editable for subsequent review.
ACCEPTED_ROWS = {
    6324, 6325, 6327, 6329, 6331, 6333, 6336, 6337, 6340, 6341,
    6342, 6343, 6350, 6351, 6352, 6353, 6354, 6355, 6356, 6357, 6358,
    8969, 8974, 8977, 8981, 8983, 8985, 8987, 8990, 8992, 8994,
    8996, 8997, 9000, 9002, 9004,
    10153, 10157, 10160, 10171, 10172, 10173, 10179, 10180, 10181,
    10431, 11081,
    11180, 11197, 11199, 11201, 11203, 11209, 11210, 11211, 11245,
}

LANGUAGE_CHANGES = {
    "Dras": "dr",
    "Gilgit": "gil",
    "Palas": "pales",
    "Punial": "punl",
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


def normalized_form(value: str) -> str:
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


def senses(value: str) -> set[str]:
    """Return whole gloss senses, with light imperative/infinitive cleanup."""
    value = html.unescape(re.sub(r"<[^>]+>", " ", value)).lower()
    result = set()
    for part in re.split(r"[;,/]|\bor\b", value):
        part = re.sub(r"\[[^]]*]|\([^)]*\)", " ", part)
        part = re.sub(r"[^a-z -]+", " ", part)
        part = re.sub(r"\s+", " ", part).strip()
        part = re.sub(r"^(?:you|to) ", "", part)
        part = re.sub(r"^(?:become|becomes) ", "", part)
        if part:
            result.add(part)
    return result


def root_id(row: dict[str, str], forms: dict[str, dict[str, str]]) -> str:
    """Return the highest Indo-Aryan etymon, never its Indo-Iranian parent.

    A non-IA loan source is retained only when the ancestry contains no
    Indo-Aryan node at all (for example Persian ``e50``).  Shina reflexes of an
    IA term borrowed from another family attach to that IA term, not directly
    to the remoter donor-language root.
    """
    seen = set()
    indo_aryan = row if row.get("Language_ID") == "Indo-Aryan" else None
    while row.get("Origin_ID") in forms and row["Origin_ID"] not in seen:
        seen.add(row["ID"])
        row = forms[row["Origin_ID"]]
        if row.get("Language_ID") == "Indo-Aryan":
            indo_aryan = row
        elif indo_aryan is not None:
            break
    return (indo_aryan or row)["ID"]


def generate_review() -> None:
    with FORMS.open(encoding="utf-8") as handle:
        form_rows = list(csv.DictReader(handle))
    import sys as _sys
    _sys.path.insert(0, str(DATA))
    from edges_util import attach_legacy_graph
    attach_legacy_graph(form_rows, str(DATA / "cldf/edges.csv"))
    forms = {row["ID"]: row for row in form_rows}
    with LANGUAGES.open(encoding="utf-8") as handle:
        clades = {row["ID"]: row["Clade"] for row in csv.DictReader(handle)}

    evidence_by_sense: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    for row in form_rows:
        if clades.get(row["Language_ID"]) != "Shinaic" or not row["Origin_ID"]:
            continue
        sources = {source for source in row["Source"].split(";") if source}
        if sources == {"backstrom1992"}:
            continue
        rid = root_id(row, forms)
        row_senses = senses(";".join((row["Gloss"], row["Description"])))
        spellings = {normalized_form(row["Form"])}
        if row["Phonemic"]:
            spellings.add(normalized_form(row["Phonemic"]))
        for sense in row_senses:
            for spelling in spellings:
                evidence_by_sense[sense].append(
                    (
                        rid,
                        spelling,
                        row["Language_ID"],
                        row["Form"],
                        ";".join(sorted(sources)),
                    )
                )

    previous = {}
    if REVIEW.exists():
        with REVIEW.open(encoding="utf-8") as handle:
            previous = {row["Row"]: row for row in csv.DictReader(handle)}

    with RAW_FORMS.open(encoding="utf-8") as handle:
        raw_rows = list(csv.reader(handle))

    output = []
    for row_number, row in enumerate(raw_rows, 1):
        language = LANGUAGE_CHANGES.get(row[0], row[0])
        if (row[1] and str(row_number) not in previous) or clades.get(language) != "Shinaic":
            continue
        target = normalized_form(row[5] or row[2])
        target_senses = senses(row[3])
        candidates = [item for sense in target_senses for item in evidence_by_sense.get(sense, [])]
        if not target or not candidates:
            continue

        best_by_root = {}
        for rid, candidate, evidence_language, evidence_form, source in candidates:
            score = distance(target, candidate) / max(len(target), len(candidate), 1)
            if rid not in best_by_root or score < best_by_root[rid][0]:
                best_by_root[rid] = (score, evidence_language, evidence_form, source)
        ranked = sorted((score, rid, lang, form, source) for rid, (score, lang, form, source) in best_by_root.items())
        best_score, rid, evidence_language, evidence_form, source = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 1.0
        if best_score > 0.34 or second_score - best_score < 0.12:
            continue

        old = previous.get(str(row_number), {})
        output.append(
            {
                "Row": row_number,
                "Language_ID": row[0],
                "Form": row[2],
                "Gloss": row[3],
                "Phonemic": row[5],
                "Parameter_ID": rid,
                "Distance": f"{best_score:.3f}",
                "Evidence_Language_ID": evidence_language,
                "Evidence_Form": evidence_form,
                "Evidence_Source": source,
                "Decision": old.get("Decision") or ("yes" if row_number in ACCEPTED_ROWS else ""),
                "Notes": old.get("Notes", ""),
            }
        )

    fields = [
        "Row", "Language_ID", "Form", "Gloss", "Phonemic", "Parameter_ID",
        "Distance", "Evidence_Language_ID", "Evidence_Form", "Evidence_Source",
        "Decision", "Notes",
    ]
    with REVIEW.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} candidates to {REVIEW}")


def apply_review() -> None:
    with REVIEW.open(encoding="utf-8") as handle:
        accepted = {
            int(row["Row"]): row["Parameter_ID"]
            for row in csv.DictReader(handle)
            if row["Decision"].strip().lower() in {"yes", "y", "accept", "accepted"}
        }
    with RAW_FORMS.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    changed = 0
    for row_number, parameter_id in accepted.items():
        row = rows[row_number - 1]
        if not row[1]:
            row[1] = parameter_id
            changed += 1
        elif row[1] != parameter_id:
            raise ValueError(f"Row {row_number} already links to {row[1]}, not {parameter_id}")
    with RAW_FORMS.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    print(f"Applied {changed} accepted Shina database assignments to {RAW_FORMS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    apply_review() if args.apply else generate_review()


if __name__ == "__main__":
    main()
