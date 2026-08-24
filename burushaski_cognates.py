"""Build and apply conservative, form-less Proto-Burushaski comparative sets.

The important distinction in this module is between a shared *meaning* and a
shared lexeme.  A concept prompt is never sufficient on its own.  We accept
only (a) dialect correspondences explicitly encoded by Berger or (b) forms
from the same HKAT prompt which are phonologically similar.  Everything else
belongs in a review file, not in the etymology graph.

The checked-in ``data/burushaski_cognates.csv`` is the auditable boundary.
Its members are immutable source keys rather than generated Jambu form IDs.
During ``unify_cldf.py`` each accepted set becomes a Proto-Burushaski grouping
entry and each cited source form becomes a reflex of it. The grouping entry has
no reconstructed form: Jambu does not reconstruct Proto-Burushaski.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "data/burushaski_cognates.csv"
BERGER = ROOT / "data/other/forms/20260726-berger-auto.csv"
HKAT = ROOT / "data/other/forms/20260810-liljegren-hindukush.csv"
YOSHIOKA = ROOT / "data/other/forms/20260726-yoshioka-eastern-burushaski.csv"
SOURCE_KEYS = ROOT / "cldf/form-source-keys.csv"

FIELDS = [
    "Set_ID", "Proto_Form", "Gloss", "Evidence_Keys", "Method", "Status", "Notes"
]
ACCEPTED = {"accepted", "yes", "active"}


@dataclass(frozen=True)
class SourceForm:
    language: str
    form: str
    gloss: str
    key: str
    parent_key: str
    tags: str
    source: str


def read_rich_rows(path: Path) -> list[SourceForm]:
    """Read Jambu's headerless fifteen-column rich import format."""
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for raw in csv.reader(stream):
            if len(raw) != 15:
                raise ValueError(f"{path}: expected 15 columns, found {len(raw)}")
            rows.append(SourceForm(raw[0], raw[2], raw[3], raw[10], raw[11], raw[14], raw[7]))
    return rows


def comparison_form(value: str) -> str:
    """Loose comparison spelling used only after provenance/concept gating."""
    value = unicodedata.normalize("NFD", value.casefold())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("tɕ", "c").replace("dʑ", "j").replace("ts", "c")
    value = value.translate(str.maketrans({
        "ʰ": "h", "ː": "", "_": "", "-": "", " ": "", "ʈ": "t", "ɖ": "d",
        "ʂ": "s", "ʃ": "s", "ś": "s", "ṣ": "s", "ɕ": "s", "ʦ": "c",
        "ʧ": "c", "č": "c", "ć": "c", "ʤ": "j",
        "ʐ": "r", "ʒ": "r", "ŋ": "n", "ṅ": "n", "ɰ": "g", "χ": "q",
    }))
    return "".join(ch for ch in value if ch.isalpha())


def similarity(left: str, right: str) -> float:
    a, b = comparison_form(left), comparison_form(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _safe_set_id(prefix: str, key: str) -> str:
    tail = re.sub(r"[^a-z0-9]+", "-", key.casefold()).strip("-")
    return f"{prefix}-{tail}"


def discover_berger_sets(rows: Sequence[SourceForm]) -> list[dict[str, str]]:
    """Accept Berger's explicit Hunza/Yasin lexical correspondences.

    Rows linked to a CDIAL etymon are excluded: those are potential loans and
    already have an etymological parent.  Treating them as inherited merely
    because two dialects share them would manufacture a false reconstruction.
    """
    by_key = {row.key: row for row in rows if row.key}
    children: dict[str, list[SourceForm]] = defaultdict(list)
    for row in rows:
        parent = by_key.get(row.parent_key)
        if parent and row.language != parent.language:
            children[parent.key].append(row)

    out = []
    for parent_key, variants in sorted(children.items()):
        parent = by_key[parent_key]
        # The second rich column is not retained by SourceForm. It is present in
        # the source citation only for loans, so recover it from the raw rows in
        # generate_catalog before calling this function.
        members = [parent, *variants]
        keys = list(dict.fromkeys(member.key for member in members))
        if len(keys) < 2:
            continue
        out.append({
            "Set_ID": _safe_set_id("berger", parent.key),
            "Proto_Form": "",
            "Gloss": parent.gloss,
            "Evidence_Keys": "|".join(keys),
            "Method": "source-explicit-dialect-correspondence",
            "Status": "accepted",
            "Notes": "Form intentionally blank; Berger explicitly supplies the Hunza/Yasin correspondence.",
        })
    return out


def _hkat_concept(row: SourceForm) -> str:
    match = re.search(r"concept ([^\]]+)\]", row.source)
    return match.group(1) if match else ""


def primary_gloss(value: str) -> str:
    """A strict English first-sense key for cross-source dictionary matching."""
    value = re.sub(r"\([^)]*\)", "", value.casefold())
    value = re.split(r"[,;/]", value, maxsplit=1)[0]
    return re.sub(r"[^a-z]+", " ", value).strip()


def discover_hkat_sets(rows: Sequence[SourceForm], threshold: float = 0.72) -> list[dict[str, str]]:
    """Find cognates, not co-synonyms, in paired Hunza/Nagar elicitation data."""
    by_concept: dict[str, dict[str, list[SourceForm]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.language in {"HKAT-bsk_h", "HKAT-bsk_n"}:
            by_concept[_hkat_concept(row)][row.language].append(row)

    out = []
    for concept, lects in sorted(by_concept.items()):
        hunza, nagar = lects["HKAT-bsk_h"], lects["HKAT-bsk_n"]
        # Connected components of qualifying cross-lect pairs retain genuine
        # within-prompt alternants without grouping an unrelated co-synonym.
        links = [(h, n) for h in hunza for n in nagar if similarity(h.form, n.form) >= threshold]
        seen: set[str] = set()
        for h, n in sorted(links, key=lambda pair: (pair[0].key, pair[1].key)):
            if h.key in seen or n.key in seen:
                continue
            members = [x for x in hunza + nagar if x.key in {h.key, n.key}]
            seen.update(x.key for x in members)
            out.append({
                "Set_ID": _safe_set_id("hkat", concept + "-" + h.key.rsplit("-", 1)[-1]),
                "Proto_Form": "",
                "Gloss": h.gloss,
                "Evidence_Keys": "|".join(x.key for x in members),
                "Method": f"same-concept-form-similarity>={threshold:.2f}",
                "Status": "accepted",
                "Notes": "Form intentionally blank; semantic identity plus phonological similarity establishes only set membership.",
            })
    return out


def attach_yoshioka_evidence(
    sets: list[dict[str, str]],
    hkat_rows: Sequence[SourceForm],
    yoshioka_rows: Sequence[SourceForm],
    threshold: float = 0.72,
) -> None:
    """Attach Eastern Burushaski only when both first sense and form support cognacy.

    Loan-tagged Yoshioka entries are excluded from inherited proto sets. If an
    entry could fit several HKAT sets, only its best phonological match wins.
    """
    hkat_by_key = {row.key: row for row in hkat_rows}
    candidates: dict[str, list[tuple[float, str]]] = defaultdict(list)
    by_set = {row["Set_ID"]: row for row in sets}
    for item in sets:
        if not item["Method"].startswith("same-concept-form-similarity"):
            continue
        members = [hkat_by_key[key] for key in item["Evidence_Keys"].split("|")]
        gloss = primary_gloss(item["Gloss"])
        for eastern in yoshioka_rows:
            if "loanword" in eastern.tags.split() or primary_gloss(eastern.gloss) != gloss:
                continue
            score = max(similarity(eastern.form, member.form) for member in members)
            if score >= threshold:
                candidates[eastern.key].append((score, item["Set_ID"]))

    yoshioka_by_key = {row.key: row for row in yoshioka_rows}
    for key, choices in sorted(candidates.items()):
        _score, set_id = max(choices, key=lambda choice: (choice[0], choice[1]))
        item = by_set[set_id]
        item["Evidence_Keys"] += "|" + key
        suffix = "+english-gloss-and-form-match"
        if suffix not in item["Method"]:
            item["Method"] += suffix
            item["Notes"] += " Eastern Burushaski evidence matched independently by first sense and form."


def generate_catalog(
    berger_path: Path = BERGER,
    hkat_path: Path = HKAT,
    yoshioka_path: Path = YOSHIOKA,
    source_keys_path: Path | None = SOURCE_KEYS,
) -> list[dict[str, str]]:
    available = None
    if source_keys_path and source_keys_path.exists():
        with source_keys_path.open(encoding="utf-8", newline="") as stream:
            available = {row["Source_Key"] for row in csv.DictReader(stream)}
    berger_raw = list(csv.reader(berger_path.open(encoding="utf-8", newline="")))
    # Only unetymologised Berger entries are eligible for inherited Proto-Burushaski sets.
    eligible_keys = {row[10] for row in berger_raw if len(row) == 15 and not row[1]}
    berger = [
        row for row in read_rich_rows(berger_path)
        if row.key in eligible_keys and (available is None or row.key in available)
    ]
    hkat = [
        row for row in read_rich_rows(hkat_path)
        if available is None or row.key in available
    ]
    yoshioka = [
        row for row in read_rich_rows(yoshioka_path)
        if available is None or row.key in available
    ]
    hkat_sets = discover_hkat_sets(hkat)
    attach_yoshioka_evidence(hkat_sets, hkat, yoshioka)
    sets = discover_berger_sets(berger) + hkat_sets
    # make_cldf's legacy cross-source deduper can subsume an OCR record into an
    # older identical spelling. Such a source key is intentionally absent from
    # the graph sidecar and therefore cannot be cited as independent evidence.
    if available is not None:
        sets = [
            row for row in sets
            if all(key in available for key in row["Evidence_Keys"].split("|"))
        ]
    ids = [row["Set_ID"] for row in sets]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate Proto-Burushaski set IDs")
    return sorted(sets, key=lambda row: row["Set_ID"])


def write_catalog(rows: Iterable[dict[str, str]], path: Path = CATALOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_catalog(path: Path = CATALOG) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["Status"].casefold() in ACCEPTED]


def apply_catalog(
    rows: list[list[str]], source_id_by_key: dict[str, str], catalog: Sequence[dict[str, str]]
) -> tuple[list[list[str]], list[tuple[str, str]]]:
    """Create proto nodes and re-home evidence rows as their reflexes."""
    by_id = {row[0]: row for row in rows}
    claimed: dict[str, str] = {}
    proto_rows: list[list[str]] = []
    proto_source_keys: list[tuple[str, str]] = []
    for item in catalog:
        set_id = item["Set_ID"]
        if item.get("Proto_Form"):
            raise ValueError(
                f"Proto-Burushaski {set_id}: Proto_Form must be blank; no reconstruction is proposed"
            )
        proto_id = "pbsk-" + set_id
        keys = [key for key in item["Evidence_Keys"].split("|") if key]
        evidence_ids = [source_id_by_key.get(key, "") for key in keys]
        missing = [key for key, form_id in zip(keys, evidence_ids) if not form_id]
        if missing:
            raise ValueError(f"Proto-Burushaski {set_id}: unknown evidence keys {missing}")
        if len(set(evidence_ids)) < 2:
            raise ValueError(f"Proto-Burushaski {set_id}: fewer than two distinct evidence forms")
        for key, form_id in zip(keys, evidence_ids):
            previous = claimed.setdefault(form_id, set_id)
            if previous != set_id:
                raise ValueError(f"Burushaski form {key} occurs in both {previous} and {set_id}")
            child = by_id[form_id]
            if child[13] not in {"local", "variant"}:
                raise ValueError(
                    f"Proto-Burushaski {set_id}: {key} already has {child[13]} origin {child[11]}"
                )
            child[11], child[13], child[15], child[16] = proto_id, "reflex", "", ""

        note = item["Notes"]
        proto = [
            proto_id, "PBr", "", item["Gloss"], "", "", "",
            "", "", "uncertain", "", "", note,
            "", "", "", "",
        ]
        proto_rows.append(proto)
        proto_source_keys.append((proto_id, "proto-burushaski:" + set_id))
    return proto_rows, proto_source_keys


if __name__ == "__main__":
    rows = generate_catalog()
    write_catalog(rows)
    print(f"wrote {len(rows)} accepted Proto-Burushaski comparative sets to {CATALOG}")
