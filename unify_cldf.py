"""
unify_cldf.py — fold the etyma (parameters.csv) and the attested reflexes (forms.csv) into ONE
table: cldf/forms.csv, one row per node in the etymon graph. Reflexes point at their etymon via a
self-referential `Origin_ID`; etyma have an empty `Origin_ID`. parameters.csv is then removed.

The etymon is NOT duplicated as its own reflex. For each etymon we find its self-reflex — the form
in the etymon's own language whose Form equals the head-word — and fold that reflex's parsed data
(gloss, tags, native/phonemic/original, source) up onto the etymon node, then drop it. The etymon's
free-text etymological header (the CDIAL entry HTML) is stored in a dedicated `Etymology` column,
leaving `Gloss` for the parsed short meaning.

Same-language, non-head-word forms (e.g. OIA variant spellings / reconstructions) are kept but
marked `Relation = variant`; genuine daughter-language reflexes are `Relation = reflex`; etyma have
an empty `Relation`.

Column mapping:
    etymon : Form=headword, Gloss=parsed meaning (from self-reflex), Etymology=CDIAL entry HTML,
             Tags/Native/Phonemic/Original/Source folded from the self-reflex, Description=Etyma,
             Origin_ID="", Relation=""
    reflex : Form=form, Gloss=meaning, Description=notes, Origin_ID=<etymon id>,
             Relation="reflex" | "variant"

Run LAST in the data pipeline (after make_cldf.py, align.py, link_refs.py), since those read the
split files.
"""

import csv
import os
import re
import sys
import unicodedata
from collections import defaultdict

_ADD_PTR = re.compile(r"\s*Add\.\s*\d+\.?")  # the now-defunct "Add. N" pointer after a merge

UNIFIED = [
    "ID", "Language_ID", "Form", "Gloss", "Native", "Phonemic", "Original", "Cognateset",
    "Description", "Tags", "Source", "Origin_ID", "Etymology", "Relation", "Redirect", "Variant_Of",
]


def strip_marker(pid: str) -> str:
    """Reflex Parameter_IDs may carry a borrowing / semi-tatsama marker (>, ~)."""
    return pid[1:] if pid and pid[0] in ">~" else pid


def nfc(s: str) -> str:
    """Head-words and forms can disagree on Unicode normalisation (precomposed vs combining); compare
    them NFC-folded so a self-reflex like OIA aṅgúli is recognised as the head-word."""
    return unicodedata.normalize("NFC", s)


def main():
    with open("cldf/parameters.csv", encoding="utf-8") as f:
        params = list(csv.DictReader(f))
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = list(csv.DictReader(f))

    params_by_id = {p["ID"]: p for p in params}
    forms_by_param = defaultdict(list)
    for r in forms:
        forms_by_param[r["Parameter_ID"]].append(r)

    # For each etymon, locate its self-reflex: the form in the etymon's own language whose Form is
    # the head-word. Its parsed data folds up onto the etymon and the reflex row is then dropped.
    self_reflex_ids = set()
    folded = {}  # etymon ID -> the self-reflex form row
    for p in params:
        for r in forms_by_param.get(p["ID"], ()):
            if r["Language_ID"] == p["Language_ID"] and nfc(r["Form"]) == nfc(p["Name"]):
                self_reflex_ids.add(r["ID"])
                folded.setdefault(p["ID"], r)

    # addenda→main merges (computed by head-word in link_refs.py)
    merges = {}
    if os.path.exists("cldf/merges.csv"):
        with open("cldf/merges.csv", encoding="utf-8") as f:
            merges = {r["Addendum_ID"]: r["Main_ID"] for r in csv.DictReader(f)}

    # ---- build etymon rows -------------------------------------------------
    G, NA, PH, OR, DE, TG, SR_, ET, RD = 3, 4, 5, 6, 8, 9, 10, 12, 14  # column indices
    etyma_rows, etyma_by_id = [], {}
    for p in params:
        header = p["Description"]
        is_html = header.startswith("<html")
        # CDIAL entries carry the full dictionary entry HTML as the "header"; other sources
        # (Dravidian/Munda/Nuristani) put the plain meaning there instead.
        gloss = "" if is_html else header
        etymology = header if is_html else ""
        native = phonemic = original = tags = source = ""
        sr = folded.get(p["ID"])
        if sr:  # fold the self-reflex's parsed data into empty etymon fields
            gloss = gloss or sr["Gloss"]
            tags = tags or sr.get("Tags", "")
            native = native or sr["Native"]
            phonemic = phonemic or sr["Phonemic"]
            original = original or sr["Original"]
            source = source or sr["Source"]
        row = [p["ID"], p["Language_ID"], p["Name"], gloss, native, phonemic, original,
               "", p.get("Etyma", ""), tags, source, "", etymology, "", "", ""]
        etyma_rows.append(row)
        etyma_by_id[p["ID"]] = row

    # ---- build reflex rows (self-reflexes dropped; addenda reflexes re-parented) ------
    n_reflex = n_variant = 0
    reflex_rows = []
    for r in forms:
        if r["ID"] in self_reflex_ids:
            continue
        pid = strip_marker(r["Parameter_ID"])
        parent = params_by_id.get(pid)
        # two kinds of variant: a comma-listed alternate of a main reflex (Variant_Of set by
        # make_cldf), or a same-language non-head-word form (a variant of the etymon head itself).
        vof = r.get("Variant_Of", "")
        if vof and vof not in self_reflex_ids:  # alternate of a main reflex
            relation = "variant"
            n_variant += 1
        elif parent and r["Language_ID"] == parent["Language_ID"]:  # head variant (of the etymon)
            relation = "variant"
            vof = ""
            n_variant += 1
        else:
            relation = "reflex"
            vof = ""
            n_reflex += 1
        origin = r["Parameter_ID"]
        if pid in merges:  # this reflex belongs to a merged addendum → hang it on the main entry
            mk = origin[0] if origin and origin[0] in ">~" else ""
            origin = mk + merges[pid]
        reflex_rows.append([
            r["ID"], r["Language_ID"], r["Form"], r["Gloss"], r["Native"], r["Phonemic"],
            r["Original"], r["Cognateset"], r["Description"], r.get("Tags", ""), r["Source"],
            origin, "", relation, "", vof,
        ])

    # ---- fold each addendum's content up onto its main entry, then redirect it -------
    n_merged = 0
    for n, m in merges.items():
        nrow, mrow = etyma_by_id.get(n), etyma_by_id.get(m)
        if not nrow or not mrow:
            continue
        for i in (G, NA, PH, OR, TG):
            mrow[i] = mrow[i] or nrow[i]
        mrow[SR_] = ";".join(x for x in (mrow[SR_], nrow[SR_]) if x)
        m_note = _ADD_PTR.sub("", mrow[ET]) if "[" in mrow[ET] else ""  # keep only a real [Cf. …] note
        mrow[ET] = (nrow[ET] or "") + m_note
        nrow[RD] = m  # the addendum redirects to its main entry
        n_merged += 1

    with open("cldf/forms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(UNIFIED)
        w.writerows(etyma_rows)
        w.writerows(reflex_rows)

    os.remove("cldf/parameters.csv")
    print(
        f"unified cldf/forms.csv: {len(etyma_rows)} etyma "
        f"({len(folded)} folded self-reflexes, {n_merged} merged addenda) + {n_reflex} reflexes "
        f"+ {n_variant} variants; removed parameters.csv",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
