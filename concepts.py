"""Map free-text glosses to Concepticon concept sets.

Runs after make_cldf: reads cldf/forms.csv, assigns each form zero or more Concepticon
concepts (via pysem's bundled mapper), and writes a concept catalogue (cldf/concepts.csv)
plus form->concept links (cldf/form_concepts.csv).
"""

import csv
import re
import sys

from pysem import to_concepticon

# minimum mapper similarity to accept a Concepticon match (pysem's score; higher = better)
MIN_SCORE = 3


def senses(gloss):
    """Split a gloss into candidate senses: drop markup and parenthetical citations,
    then split on ';', ',' and ' or '."""
    g = re.sub(r"<[^>]+>", "", gloss)
    g = re.sub(r"\([^)]*\)", " ", g)
    g = re.sub(r"\[[^\]]*\]", " ", g)
    parts = re.split(r"[;,]| or ", g)
    return [p.strip(" .?") for p in parts if p.strip(" .?")]


def map_glosses(glosses):
    """Return {gloss: [(concepticon_id, label, pos), ...]} for a list of raw gloss strings.
    Distinct senses are mapped in one batch; multiple concepts per gloss are kept."""
    # collect the distinct senses across all glosses, map them once
    sense_set = {}
    for g in glosses:
        for s in senses(g):
            sense_set.setdefault(s.lower(), s)
    keys = list(sense_set)
    hits = to_concepticon([{"gloss": sense_set[k]} for k in keys], language="en")
    sense_concepts = {}
    for k in keys:
        res = hits.get(sense_set[k]) or []
        sense_concepts[k] = [
            (cid, label, pos) for (cid, label, pos, score) in res if score >= MIN_SCORE
        ]
    out = {}
    for g in glosses:
        seen = {}
        for s in senses(g):
            for cid, label, pos in sense_concepts.get(s.lower(), []):
                seen[cid] = (cid, label, pos)
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
