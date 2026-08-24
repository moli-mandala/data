"""Map free-text glosses to Concepticon concept sets.

Runs after make_cldf: reads cldf/forms.csv, assigns each form zero or more Concepticon
concepts (via pysem's bundled mapper), and writes a concept catalogue (cldf/concepts.csv)
plus form->concept links (cldf/form_concepts.csv).
"""

import csv
import re
import sys

from pysem import to_concepticon
from pysem.glosses import parse_gloss

# minimum mapper similarity to accept a Concepticon match (pysem's score; higher = better)
MIN_SCORE = 3

# Disable pysem's own splitting after we have split only at top-level separators.
# Its splitter runs before bracket parsing and would otherwise split citations or
# etymological notes containing commas and semicolons.
NEVER_SPLIT = r"(?!x)x"


def _split_top_level(text):
    """Split enumerated senses without splitting inside brackets."""
    closers = {"(": ")", "[": "]", "{": "}", "（": "）"}
    expected = []
    parts = []
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char in closers:
            expected.append(closers[char])
        elif expected and char == expected[-1]:
            expected.pop()
        elif not expected:
            separator_length = 0
            if char in ",;":
                separator_length = 1
            elif text[i : i + 4].casefold() == " or ":
                separator_length = 4
            if separator_length:
                part = text[start:i].strip(" .?")
                if part:
                    parts.append(part)
                i += separator_length
                start = i
                continue
        i += 1
    final = text[start:].strip(" .?")
    if final:
        parts.append(final)
    return parts


def sense_candidates(gloss):
    """Return distinct ``(text, pos)`` candidates for a source gloss.

    The complete gloss is retained so pysem can recognize combined Concepticon
    labels.  Its parsed constituents are also returned so enumerated senses can
    each receive a concept.  Unlike the old regular-expression splitter, this
    preserves bracketed context and lets pysem infer parts of speech from markers
    such as ``to`` and ``(v.)``.
    """
    text = re.sub(r"<[^>]+>", "", gloss).strip(" .?")
    if not text:
        return []

    candidates = [(text, "")]
    for constituent in _split_top_level(text):
        item = parse_gloss(constituent, language="en", splitter=NEVER_SPLIT)[0]
        candidates.append((item.gloss.strip(" .?"), item.pos))
    return list(dict.fromkeys(candidate for candidate in candidates if candidate[0]))


def _legacy_senses(gloss):
    """Return candidates produced by the pre-parse_gloss compatibility path."""
    text = re.sub(r"<[^>]+>", "", gloss)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    parts = re.split(r"[;,]| or ", text)
    return [part.strip(" .?") for part in parts if part.strip(" .?")]


def map_glosses(glosses):
    """Return {gloss: [(concepticon_id, label, pos), ...]} for a list of raw gloss strings.
    Distinct senses are mapped in one batch; multiple concepts per gloss are kept."""
    # Collect distinct (text, POS) candidates across all glosses.  to_concepticon
    # keys its output only by text, so batch separately by POS to avoid collisions.
    candidates_by_gloss = {}
    candidate_groups = {}
    for g in glosses:
        candidates = [
            (text, pos, NEVER_SPLIT) for text, pos in sense_candidates(g)
        ]
        # Keep the former mapper as a compatibility channel.  Its default pysem
        # splitter is significant for some slash-delimited historical glosses.
        candidates.extend((text, "", None) for text in _legacy_senses(g))
        candidates = list(dict.fromkeys(candidates))
        candidates_by_gloss[g] = candidates
        for text, pos, splitter in candidates:
            candidate_groups.setdefault((pos, splitter), {}).setdefault(
                text.casefold(), text
            )

    sense_concepts = {}
    for (pos, splitter), texts_by_key in candidate_groups.items():
        concepts = [{"gloss": text, "pos": pos} for text in texts_by_key.values()]
        kwargs = {"splitter": splitter} if splitter is not None else {}
        hits = to_concepticon(concepts, language="en", pos_ref="pos", **kwargs)
        for key, text in texts_by_key.items():
            res = hits.get(text) or []
            sense_concepts[(key, pos, splitter)] = [
                (cid, label, match_pos)
                for cid, label, match_pos, score in res
                if score >= MIN_SCORE
            ]

    out = {}
    for g in glosses:
        seen = {}
        for text, pos, splitter in candidates_by_gloss[g]:
            key = (text.casefold(), pos, splitter)
            for cid, label, match_pos in sense_concepts.get(key, []):
                seen[cid] = (cid, label, match_pos)
        out[g] = list(seen.values())
    return out


def assign(rows):
    """rows: list of (form_id, gloss). Returns (catalogue, links).
    catalogue: {concept_id: (label, category)}; links: list of (form_id, concept_id)."""
    glosses = sorted({g for _, g in rows if g})
    gmap = map_glosses(glosses)
    catalogue = {}
    links = []
    for fid, gloss in rows:
        if not gloss:
            continue
        concepts = gmap.get(gloss, [])
        for cid, label, cat in concepts:
            catalogue[cid] = (label, cat)
            links.append((fid, cid))
    return catalogue, links


def main(forms_path="cldf/forms.csv"):
    rows = []
    with open(forms_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["ID"], r.get("Gloss", "")))
    catalogue, links = assign(rows)
    with open("cldf/concepts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Name", "Category"])
        for cid, (label, cat) in sorted(catalogue.items()):
            w.writerow([cid, label, cat])
    with open("cldf/form_concepts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Form_ID", "Concept_ID"])
        w.writerows(links)
    forms_with = len({fid for fid, _ in links})
    print(f"concepts: {len(catalogue)}")
    print(f"links: {len(links)}; forms with >=1 concept: {forms_with}")


if __name__ == "__main__":
    main(*sys.argv[1:])
