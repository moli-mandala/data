"""Project Burushaski--Indo-Aryan links into the article-comparison layer.

Legacy Jambu data attached every non-Indo-Aryan form carrying a Turner/CDIAL
identifier directly to the Indo-Aryan entry.  ``unify_cldf.py`` then had no
choice but to serialize the structurally cross-family edge as ``borrowed``.
For Burushaski that overstates the evidence: Berger's ``T`` references, the
Burushaski forms printed by CDIAL, and older editorial wordlist links establish
comparanda, but do not uniformly establish borrowing or its direction.

This module performs the lossless graph migration.  It groups only compatible
Burushaski attestations, creates form-less Proto-Burushaski comparison nodes,
re-homes the attestations as their reflexes/variants, and emits sourced
Proto-Burushaski <-> CDIAL comparisons.  The comparison is deliberately
``related / undetermined``; its confidence describes the source wording, not a
new Jambu historical judgement.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from burushaski_cognates import comparison_form, primary_gloss, similarity


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "data/burushaski-indo-aryan-comparisons-audit.csv"

COMPARISON_FIELDS = [
    "ID", "Entry_ID", "Compared_Entry_ID", "Relation", "Direction", "Confidence",
    "Source", "Evidence",
]
AUDIT_FIELDS = [
    "Status", "Reason", "Proto_Burushaski_ID", "Proto_Source_Key", "CDIAL_ID",
    "Claim_Source_Entry_ID", "Legacy_Form_ID", "Source_Key", "Language_ID", "Form", "Gloss",
    "Source", "Prior_Relation", "New_Relation", "Comparison_IDs",
]

# Internal unified-row positions (kept in sync with unify_cldf.UNIFIED).
ID = 0
LANG = 1
FORM = 2
GLOSS = 3
ORIGINAL = 6
DESCRIPTION = 8
TAGS = 9
SOURCE = 10
ORIGIN = 11
ETYMOLOGY = 12
RELATION = 13
VARIANT_OF = 15
BORROWED_FROM = 16

CDIAL_ID = re.compile(r"\d+[a-z]?(?:-\d+x?)?")
PRIMARY_CLAIM_SOURCES = {"berger", "berger-auto", "backstrom1992", "CDIAL"}
REVIEW_SAMPLE_QUOTAS = {
    "berger-auto": 10,
    "berger": 3,
    "backstrom1992": 4,
    "CDIAL": 3,
}


def _split_citations(value: str) -> list[str]:
    """Split CLDF citations without treating semicolons inside locators as separators."""
    result: list[str] = []
    start = depth = 0
    for index, char in enumerate(value):
        if char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
        elif char == ";" and depth == 0:
            item = value[start:index].strip()
            if item:
                result.append(item)
            start = index + 1
    item = value[start:].strip()
    if item:
        result.append(item)
    return result


def _citation_key(value: str) -> str:
    return value.split("[", 1)[0]


def _claim_citations(
    row: Sequence[str], target_id: str, claim_source_entry_id: str
) -> list[str]:
    citations: list[str] = []
    for raw in _split_citations(row[SOURCE]):
        key = _citation_key(raw)
        if key not in PRIMARY_CLAIM_SOURCES:
            continue
        if "[" in raw:
            citation = raw
        elif key == "CDIAL":
            # A Burushaski item can occur in one CDIAL article as an arrowed variant of an
            # Indo-Aryan reflex whose ancestry resolves to another CDIAL head. Cite the article
            # that actually prints the Burushaski item while retaining the resolved head as the
            # comparison endpoint.
            citation = f"CDIAL[entry {claim_source_entry_id or target_id}]"
        elif key == "berger":
            citation = f"berger[Turner reference {target_id}]"
        else:
            gloss = primary_gloss(row[GLOSS]) or "unlabelled item"
            citation = f"backstrom1992[Burushaski wordlist, {gloss}]"
        if citation not in citations:
            citations.append(citation)
    if not citations:
        citations.append(f"CDIAL[entry {target_id}]")
    return citations


def _source_keys(row: Sequence[str]) -> set[str]:
    return {_citation_key(citation) for citation in _split_citations(row[SOURCE])}


def audit_source_group(row: dict[str, str]) -> str:
    """Assign one primary provenance stratum for the checked relationship sample."""
    keys = {_citation_key(citation) for citation in _split_citations(row["Source"])}
    if "berger-auto" in keys:
        return "berger-auto"
    if "berger" in keys:
        return "berger"
    if "backstrom1992" in keys:
        return "backstrom1992"
    return "CDIAL"


def reviewed_sample_candidates(
    rows: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    """Choose a deterministic, source-stratified 20-row relationship-review sample.

    One attestation per projected set is selected so the review covers 20 distinct comparisons.
    The checked-in sample adds human review fields; this function deliberately does not mark a
    newly selected row as reviewed.
    """
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[audit_source_group(row)].append(row)
    selected: list[dict[str, str]] = []
    for source_group, quota in REVIEW_SAMPLE_QUOTAS.items():
        ordered = sorted(
            grouped[source_group],
            key=lambda row: hashlib.sha256(
                f"{row['Legacy_Form_ID']}|{row['CDIAL_ID']}".encode("utf-8")
            ).hexdigest(),
        )
        seen_sets: set[str] = set()
        for row in ordered:
            if row["Proto_Burushaski_ID"] in seen_sets:
                continue
            selected.append(row)
            seen_sets.add(row["Proto_Burushaski_ID"])
            if len(seen_sets) == quota:
                break
        if len(seen_sets) != quota:
            raise ValueError(
                f"not enough {source_group} comparison sets for {quota}-row review stratum"
            )
    return selected


def _compatible_gloss(left: Sequence[str], right: Sequence[str]) -> bool:
    a, b = primary_gloss(left[GLOSS]), primary_gloss(right[GLOSS])
    if a and b:
        return a == b
    return bool(_source_keys(left) & _source_keys(right))


def _claim_target(row: Sequence[str], by_id: dict[str, Sequence[str]]) -> str:
    """Follow an Indo-Aryan reflex/variant chain to its corresponding CDIAL article."""
    current = row[ORIGIN].lstrip(">~")
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        parent = by_id.get(current)
        if parent is None:
            return ""
        if parent[LANG] == "Indo-Aryan" and CDIAL_ID.fullmatch(parent[ID]):
            return parent[ID]
        current = parent[ORIGIN].lstrip(">~")
    return ""


def _hash(value: str, size: int = 6) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=size).hexdigest()


def _row_anchor(row: Sequence[str], source_key_by_id: dict[str, str]) -> str:
    source_key = source_key_by_id.get(row[ID], "")
    if source_key:
        return "0:" + source_key
    return "1:" + "\x1f".join(
        (row[SOURCE], comparison_form(row[ORIGINAL] or row[FORM]), primary_gloss(row[GLOSS]))
    )


def _representative(rows: Sequence[Sequence[str]]) -> Sequence[str]:
    """Prefer repeated dialect evidence, then cleaner/manual source transcriptions."""
    frequency = defaultdict(int)
    for row in rows:
        frequency[comparison_form(row[ORIGINAL] or row[FORM])] += 1

    def key(row: Sequence[str]):
        sources = _source_keys(row)
        source_rank = (
            0 if "berger" in sources else
            1 if "backstrom1992" in sources else
            2 if "berger-auto" in sources else
            3 if "CDIAL" in sources else 4
        )
        normalized = comparison_form(row[ORIGINAL] or row[FORM])
        return (-frequency[normalized], source_rank, bool(row[VARIANT_OF]), len(normalized), row[ID])

    return min(rows, key=key)


def _component_rows(
    rows: Sequence[Sequence[str]], source_key_by_id: dict[str, str]
) -> list[list[Sequence[str]]]:
    """Keep homonymous CDIAL targets separate unless form/source evidence joins them."""
    parent = {row[ID]: row[ID] for row in rows}
    by_id = {row[ID]: row for row in rows}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in rows:
        sibling = row[VARIANT_OF].lstrip(">~")
        if sibling in by_id:
            union(row[ID], sibling)

    for index, left in enumerate(rows):
        left_form = comparison_form(left[ORIGINAL] or left[FORM])
        if not left_form:
            continue
        for right in rows[index + 1:]:
            right_form = comparison_form(right[ORIGINAL] or right[FORM])
            if not right_form:
                continue
            exact = left_form == right_form
            supported_similarity = _compatible_gloss(left, right) and similarity(
                left[ORIGINAL] or left[FORM], right[ORIGINAL] or right[FORM]
            ) >= 0.72
            if exact or supported_similarity:
                union(left[ID], right[ID])

    grouped: dict[str, list[Sequence[str]]] = defaultdict(list)
    for row in rows:
        grouped[find(row[ID])].append(row)
    return sorted(
        (sorted(group, key=lambda row: _row_anchor(row, source_key_by_id)) for group in grouped.values()),
        key=lambda group: _row_anchor(group[0], source_key_by_id),
    )


def _evidence(
    citation: str, rows: Sequence[Sequence[str]], target: Sequence[str]
) -> str:
    snippets = []
    for row in rows:
        text = " ".join((row[ETYMOLOGY] or "").split())
        if text and text not in snippets:
            snippets.append(text)
    if snippets:
        return " | ".join(snippets)

    forms = []
    for row in rows:
        item = row[ORIGINAL] or row[FORM]
        if row[GLOSS]:
            item += f" ‘{row[GLOSS]}’"
        if item not in forms:
            forms.append(item)
    rendered = "; ".join(forms)
    key = _citation_key(citation)
    if key == "CDIAL":
        match = re.search(r"\bentry\s+(\d+[a-z]?)", citation)
        claim_entry = match.group(1) if match else target[ID]
        if claim_entry != target[ID]:
            return (
                f"CDIAL entry {claim_entry} prints the Burushaski comparandum {rendered}; "
                f"its Indo-Aryan cross-reference chain resolves to CDIAL {target[ID]} "
                f"{target[FORM]}."
            )
        return (
            f"CDIAL entry {target[ID]} includes the Burushaski comparandum {rendered} "
            f"under {target[FORM]}."
        )
    if key == "backstrom1992":
        return (
            f"Legacy Jambu data linked the Backstrom Burushaski attestation {rendered} to "
            f"CDIAL {target[ID]} {target[FORM]}. Backstrom supplies the lexical attestation; "
            "the historical relation and its direction are not asserted here."
        )
    return (
        f"Berger's structured Turner link compares Burushaski {rendered} with "
        f"CDIAL {target[ID]} {target[FORM]}."
    )


def _confidence(citation: str, rows: Sequence[Sequence[str]]) -> str:
    if _citation_key(citation) == "backstrom1992":
        return "low"
    evidence = " ".join(row[DESCRIPTION] + " " + row[ETYMOLOGY] for row in rows)
    return "low" if "?" in evidence else "medium"


def project_claims(
    rows: list[list[str]],
    language_clades: dict[str, str],
    source_id_by_key: dict[str, str],
    indo_aryan_clades: set[str],
    source_entry_by_id: dict[str, str] | None = None,
) -> tuple[list[list[str]], list[tuple[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Return proto rows, source keys, comparisons, and per-attestation audit rows.

    ``rows`` are mutated in place: selected Burushaski rows cease to point into
    Indo-Aryan and instead become reflexes/variants of the newly returned
    Proto-Burushaski nodes.
    """
    by_id = {row[ID]: row for row in rows}
    source_entry_by_id = source_entry_by_id or {}
    source_key_by_id = {form_id: key for key, form_id in source_id_by_key.items()}
    candidates_by_target: dict[str, list[list[str]]] = defaultdict(list)
    for row in rows:
        if row[LANG] == "PBr" or language_clades.get(row[LANG]) != "Burushaski":
            continue
        if row[RELATION] not in {"reflex", "variant", "borrowed"}:
            continue
        immediate = by_id.get(row[ORIGIN].lstrip(">~"))
        if immediate is None or language_clades.get(immediate[LANG]) not in indo_aryan_clades:
            continue
        target = _claim_target(row, by_id)
        if target:
            candidates_by_target[target].append(row)

    proto_rows: list[list[str]] = []
    proto_source_keys: list[tuple[str, str]] = []
    comparisons: list[dict[str, str]] = []
    audits: list[dict[str, str]] = []
    used_set_ids: set[str] = set()

    for target_id, target_rows in sorted(candidates_by_target.items()):
        target = by_id[target_id]
        for component in _component_rows(target_rows, source_key_by_id):
            anchor = _row_anchor(component[0], source_key_by_id)
            set_id = f"ia-{target_id}-{_hash(anchor)}"
            if set_id in used_set_ids:
                raise ValueError(f"duplicate Proto-Burushaski comparison set ID {set_id}")
            used_set_ids.add(set_id)
            proto_id = "pbsk-" + set_id
            proto_source_key = "proto-burushaski:indo-aryan:" + set_id
            representative = _representative(component)
            note = (
                "Form-less Proto-Burushaski grouping node for source-linked Burushaski "
                "attestations. No reconstruction is proposed. The CDIAL link is retained as a "
                "cross-family comparison, not an accepted borrowing or inheritance edge."
            )
            proto_rows.append([
                proto_id, "PBr", "", representative[GLOSS], "", "", "",
                "", "", "uncertain", "", "", note,
                "", "", "", "",
            ])
            proto_source_keys.append((proto_id, proto_source_key))

            component_ids = {row[ID] for row in component}
            by_citation: dict[str, list[Sequence[str]]] = defaultdict(list)
            for row in component:
                prior_relation = row[RELATION]
                claim_source_entry_id = source_entry_by_id.get(row[ID], "")
                sibling = row[VARIANT_OF].lstrip(">~")
                if sibling in component_ids:
                    row[ORIGIN] = proto_id
                    row[RELATION] = "variant"
                else:
                    row[ORIGIN] = proto_id
                    row[RELATION] = "reflex"
                    row[VARIANT_OF] = ""
                row[BORROWED_FROM] = ""
                citations = _claim_citations(row, target_id, claim_source_entry_id)
                for citation in citations:
                    by_citation[citation].append(row)
                audits.append({
                    "Status": "converted",
                    "Reason": "Burushaski--CDIAL claim moved from ancestry to comparison layer",
                    "Proto_Burushaski_ID": proto_id,
                    "Proto_Source_Key": proto_source_key,
                    "CDIAL_ID": target_id,
                    "Claim_Source_Entry_ID": claim_source_entry_id,
                    "Legacy_Form_ID": row[ID],
                    "Source_Key": source_key_by_id.get(row[ID], ""),
                    "Language_ID": row[LANG],
                    "Form": row[FORM],
                    "Gloss": row[GLOSS],
                    "Source": row[SOURCE],
                    "Prior_Relation": prior_relation,
                    "New_Relation": row[RELATION],
                    "Comparison_IDs": "",  # populated after citation rows are minted
                })

            comparison_ids_by_form: dict[str, list[str]] = defaultdict(list)
            for citation, evidence_rows in sorted(by_citation.items()):
                comparison_id = (
                    f"burushaski:{set_id}:cdial:{target_id}:{_hash(citation, 5)}"
                )
                comparisons.append({
                    "ID": comparison_id,
                    "Entry_ID": proto_id,
                    "Compared_Entry_ID": target_id,
                    "Relation": "related",
                    "Direction": "undetermined",
                    "Confidence": _confidence(citation, evidence_rows),
                    "Source": citation,
                    "Evidence": _evidence(citation, evidence_rows, target),
                })
                for row in evidence_rows:
                    comparison_ids_by_form[row[ID]].append(comparison_id)
            for audit in audits[-len(component):]:
                audit["Comparison_IDs"] = "|".join(
                    comparison_ids_by_form[audit["Legacy_Form_ID"]]
                )

    comparisons.sort(key=lambda row: row["ID"])
    audits.sort(key=lambda row: (row["Proto_Burushaski_ID"], row["Legacy_Form_ID"]))
    return proto_rows, proto_source_keys, comparisons, audits


def write_audit(rows: Iterable[dict[str, str]], path: Path = AUDIT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_comparisons(
    rows: Iterable[dict[str, str]], path: Path = ROOT / "cldf/comparisons.csv"
) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COMPARISON_FIELDS:
            raise ValueError(f"unexpected comparisons schema in {path}")
        # ``unify_cldf.py`` may be rerun without the upstream comparison extractor. Replace this
        # projection's namespace while preserving every independently curated comparison.
        combined = [row for row in reader if not row["ID"].startswith("burushaski:")]
    combined.extend(rows)
    combined.sort(key=lambda row: row["ID"])
    ids = [row["ID"] for row in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate cross-family comparison IDs after Burushaski projection")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined)
