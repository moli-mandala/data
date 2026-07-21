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
# separates a main entry's etymology snippet from a merged addendum's; the webapp splits on it and
# renders one accented block per snippet (so no snippet is dropped when addenda fold into a main).
ADD_DELIM = "<!--addendum-->"

UNIFIED = [
    "ID", "Language_ID", "Form", "Gloss", "Native", "Phonemic", "Original", "Cognateset",
    "Description", "Tags", "Source", "Origin_ID", "Etymology", "Relation", "Redirect", "Variant_Of",
    "Borrowed_From",
]


def load_borrowings(path="data/borrowings.csv"):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["Borrower_ID"]: r["Source_ID"] for r in csv.DictReader(f)}


def load_nuristani_cognates(path="data/nuristani_cognates.csv"):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_nuristani_borrowings(path="data/nuristani_borrowings.csv"):
    with open(path, encoding="utf-8") as f:
        return {
            row["Proto_Nuristani_ID"]: row["Indo_Aryan_ID"]
            for row in csv.DictReader(f)
        }


def load_strand_oia_redirects(path="data/strand_oia_redirects.csv"):
    with open(path, encoding="utf-8") as f:
        return {
            row["Strand_ID"]: row["CDIAL_ID"]
            for row in csv.DictReader(f)
        }


def apply_borrowings(rows, borrowings):
    ids = {r[0] for r in rows}
    missing = sorted((borrower, source) for borrower, source in borrowings.items()
                     if borrower not in ids or source not in ids)
    if missing:
        raise ValueError(f"Unknown borrowing IDs: {missing}")
    applied = 0
    for row in rows:
        source = borrowings.get(row[0])
        if source:
            row[11] = source
            row[13] = "borrowed"
            row[16] = source
            applied += 1
    return applied


def apply_nuristani_cognates(rows, cognates):
    origins = {}
    for cognate in cognates:
        ancestor = cognate["Ancestor_ID"]
        for child in (cognate["Proto_Nuristani_ID"], cognate["Indo_Aryan_ID"]):
            existing = origins.setdefault(child, ancestor)
            if existing != ancestor:
                raise ValueError(f"Conflicting Proto-Indo-Iranian ancestors for {child}: {existing}, {ancestor}")

    by_id = {row[0]: row for row in rows}
    expected = set(origins) | {r["Ancestor_ID"] for r in cognates}
    missing = sorted(expected - set(by_id))
    if missing:
        raise ValueError(f"Unknown Nuristani cognate IDs: {missing}")
    for child, ancestor in origins.items():
        row = by_id[child]
        if row[11] and row[11] != ancestor:
            raise ValueError(f"Cannot attach {child} to {ancestor}; it already has origin {row[11]}")
        row[11] = ancestor
        row[13] = "reflex"
    return len(origins)


def apply_nuristani_borrowings(rows, borrowings):
    by_id = {row[0]: row for row in rows}
    missing = sorted(
        (nuristani, indo_aryan)
        for nuristani, indo_aryan in borrowings.items()
        if nuristani not in by_id or indo_aryan not in by_id
    )
    if missing:
        raise ValueError(f"Unknown Nuristani borrowing IDs: {missing}")

    descendants = 0
    for nuristani, indo_aryan in borrowings.items():
        branch = [
            row
            for row in rows
            if row[0] == nuristani or row[11] == nuristani
        ]
        if not branch:
            raise ValueError(f"No Nuristani borrowing branch found for {nuristani}")
        for row in branch:
            row[11] = indo_aryan
            row[13] = "borrowed"
            row[16] = indo_aryan
            if row[0] != nuristani:
                descendants += 1
    return len(borrowings), descendants


def apply_strand_oia_redirects(etyma_rows, reflex_rows, redirects):
    rows = etyma_rows + reflex_rows
    by_id = {row[0]: row for row in rows}
    missing = sorted(
        (strand, cdial)
        for strand, cdial in redirects.items()
        if strand not in by_id or cdial not in by_id
    )
    if missing:
        raise ValueError(f"Unknown Strand OIA redirect IDs: {missing}")

    for strand, cdial in redirects.items():
        if by_id[strand][1] != "Indo-Aryan" or by_id[cdial][1] != "Indo-Aryan":
            raise ValueError(f"Strand OIA redirect must join Indo-Aryan entries: {strand}, {cdial}")

    redirected = 0
    for row in rows:
        if row[0] in redirects:
            continue
        for column in (11, 14, 15, 16):
            target = redirects.get(row[column])
            if target:
                row[column] = target
                redirected += 1

    etyma_rows[:] = [row for row in etyma_rows if row[0] not in redirects]
    return len(redirects), redirected


def apply_borrowings_to_unified():
    with open("cldf/forms.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if header != UNIFIED:
        raise ValueError("cldf/forms.csv is not in unified format")
    applied = apply_borrowings(rows, load_borrowings())
    with open("cldf/forms.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"applied {applied} curated borrowings", file=sys.stderr)


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
        # CDIAL entries carry the full dictionary entry as HTML (starting with a tag — <html><body>
        # or a bare <number>/<b> depending on the bs4 parser); other sources (Dravidian/Munda/
        # Nuristani) put the plain meaning there instead.
        is_html = header.lstrip().startswith("<")
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
               "", p.get("Etyma", ""), tags, source, "", etymology, "", "", "", ""]
        etyma_rows.append(row)
        etyma_by_id[p["ID"]] = row

    # ---- build reflex rows (self-reflexes dropped; addenda reflexes re-parented) ------
    # A CDIAL entry's header lists numbered derived forms (`2. *kṣata-². 3. *kṣaṇana-. …`); each is a
    # lexeme in its own right, so we promote it to an entry derived from the head (form 1 = the
    # etymon). Reflexes are grouped into those forms by the `info` half of their Cognateset
    # ("subnum:info" → info is the form number). Non-numeric info carries forward the most recent
    # form number; form 1 (or no numbered form) stays on the head.
    n_reflex = n_variant = n_section = n_borrowed = 0
    reflex_rows = []
    section_edges = []  # (numbered-form id -> head etymon id)
    all_ids = {r["ID"] for r in forms}  # to keep promoted `<etymon>-<n>` ids collision-free

    # Borrowed sub-reflexes: a CDIAL note "(→ H. …, B. …)" lists forms borrowed FROM that reflex.
    # parse.py already split them into rows tagged with Cognateset "<subnum>:<parent-lang> →"; here we
    # link each back to its parent reflex — the one on the same etymon + section number, in the named
    # language, whose note carries the "(→" marker.
    borrow_parent = {}  # (pid, subnum, lang) -> parent reflex id
    for r in forms:
        if "(→" in (r["Description"] or ""):
            key = (strip_marker(r["Parameter_ID"]), (r["Cognateset"] or "").split(":", 1)[0],
                   r["Language_ID"])
            borrow_parent.setdefault(key, r["ID"])

    for pid_key, group in forms_by_param.items():
        pid = strip_marker(pid_key)
        parent = params_by_id.get(pid)
        cdial = parent is not None and parent["Language_ID"] == "Indo-Aryan" and pid not in merges

        # enumerate the numbered head-forms (same language as the etymon, not the self-reflex, not a
        # comma-alternate) in header order → form 2, 3, …. A promoted form is re-id'd `<etymon>-<n>`.
        section_by_num, promoted_id = {}, {}
        if cdial:
            num = 2
            prev_or = False  # the previous head-form ended in "or" → the next is its alternate
            for r in group:
                if r["Language_ID"] != parent["Language_ID"] or r.get("Variant_Of"):
                    continue
                # a head-form joined to the previous by "or" (e.g. "*dr̥kṣati or *drakṣati") is an
                # alternate of the SAME form-slot, not a new numbered form; skip it so CDIAL's own
                # form numbering — which the reflex sections index into via Cognateset info — is kept.
                alternate = prev_or
                prev_or = re.search(r"\bor$", (r.get("Description") or "").strip()) is not None
                if r["ID"] in self_reflex_ids or alternate:
                    continue
                new_id = f"{pid}-{num}"
                while new_id in all_ids:  # rare clash with a make_cldf `<file>-<row>` id
                    new_id += "x"
                all_ids.add(new_id)
                section_by_num[num] = new_id
                promoted_id[r["ID"]] = new_id
                num += 1

        last_num = 1  # carry-forward form number within this entry (1 = the head itself)
        for r in group:
            if r["ID"] in self_reflex_ids:
                continue
            vof = r.get("Variant_Of", "")
            origin = r["Parameter_ID"]
            marker = origin[:1] if origin[:1] in (">", "~") else ""
            borrowed_from = ""

            # a CDIAL numbered head-form → promote to an entry (id `<etymon>-<n>`) derived from the head
            if r["ID"] in promoted_id:
                new_id = promoted_id[r["ID"]]
                section_edges.append((new_id, pid))
                n_section += 1
                reflex_rows.append([
                    new_id, r["Language_ID"], r["Form"], r["Gloss"], r["Native"], r["Phonemic"],
                    r["Original"], "", r["Description"], r.get("Tags", ""), r["Source"],
                    "", "", "", "", "", "",
                ])
                continue

            cog = r["Cognateset"] or ""
            # a borrowed sub-reflex ("<subnum>:<lang> →") → child of the reflex it was borrowed from
            if "→" in cog:
                sub, _, rest = cog.partition(":")
                plang = rest.split("→")[0].strip()
                borrowed_from = borrow_parent.get((pid, sub, plang), "")

            # two kinds of variant: a comma-listed alternate of a main reflex (Variant_Of set by
            # make_cldf), or a same-language non-head-word form on a non-CDIAL etymon.
            if borrowed_from:
                # the reflex it was borrowed from becomes its parent (origin) — a proper node with
                # this form as a child — so ancestry recurses through it and it owns its borrowings.
                relation = "borrowed"
                vof = ""
                origin = borrowed_from
                n_borrowed += 1
            elif marker:
                origin = strip_marker(origin)
                relation = "borrowed"
                borrowed_from = origin
                vof = ""
                marker_tag = "semi-tatsama" if marker == "~" else "marked borrowing"
                tags = [tag for tag in (r.get("Tags", "") or "").split(";") if tag]
                if marker_tag not in tags:
                    tags.append(marker_tag)
                r["Tags"] = ";".join(tags)
                n_borrowed += 1
            elif vof and vof not in self_reflex_ids:
                relation = "variant"
                n_variant += 1
            elif parent and r["Language_ID"] == parent["Language_ID"]:
                relation = "variant"
                vof = ""
                n_variant += 1
            else:
                relation = "reflex"
                vof = ""
                n_reflex += 1
                if cdial:  # re-home to its numbered head-form via the Cognateset info
                    info = cog.split(":", 1)[1].strip() if ":" in cog else ""
                    m_add = re.match(r"Addenda.*?(\d+)\s*$", info)  # "Addenda: *X. N" → form N
                    if info.isdigit():
                        last_num = int(info)
                    elif m_add:
                        last_num = int(m_add.group(1))
                    elif info == "":
                        last_num = 1  # section-less paragraph → the main entry (the head)
                    # else: a non-numeric label (e.g. "prob") carries forward the last form number
                    if last_num in section_by_num:
                        mk = origin[0] if origin and origin[0] in ">~" else ""
                        origin = mk + section_by_num[last_num]

            if pid in merges and not borrowed_from:  # merged addendum → hang on the main entry
                mk = origin[0] if origin and origin[0] in ">~" else ""
                origin = mk + merges[pid]
            reflex_rows.append([
                r["ID"], r["Language_ID"], r["Form"], r["Gloss"], r["Native"], r["Phonemic"],
                r["Original"], r["Cognateset"], r["Description"], r.get("Tags", ""), r["Source"],
                origin, "", relation, "", vof, borrowed_from,
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
        # keep BOTH etymology snippets as separate blocks — the main entry's own header, then each
        # merged addendum's — joined by ADD_DELIM (the webapp renders one accented block each).
        # Previously the main's snippet was overwritten by the addendum's, silently dropping it for
        # the ~91 mains whose header lacked a "[Cf. …]" note.
        add_et = _ADD_PTR.sub("", nrow[ET] or "").strip()
        mrow[ET] = _ADD_PTR.sub("", mrow[ET] or "").strip()
        if add_et:
            mrow[ET] = mrow[ET] + ADD_DELIM + add_et if mrow[ET] else add_et
        nrow[RD] = m  # the addendum redirects to its main entry
        n_merged += 1

    n_curated_borrowings = apply_borrowings(etyma_rows, load_borrowings())
    n_nuristani_reflexes = apply_nuristani_cognates(
        etyma_rows + reflex_rows, load_nuristani_cognates()
    )
    n_nuristani_borrowings, n_nuristani_borrowed_descendants = apply_nuristani_borrowings(
        etyma_rows + reflex_rows, load_nuristani_borrowings()
    )
    n_strand_oia_redirects, n_strand_oia_references = apply_strand_oia_redirects(
        etyma_rows, reflex_rows, load_strand_oia_redirects()
    )

    with open("cldf/forms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(UNIFIED)
        w.writerows(etyma_rows)
        w.writerows(reflex_rows)

    # add the promoted numbered-form → head edges to the derivation graph (link_refs.py wrote it)
    if section_edges:
        deriv_path = "cldf/derivation.csv"
        existing = []
        if os.path.exists(deriv_path):
            with open(deriv_path, encoding="utf-8") as f:
                existing = list(csv.reader(f))[1:]  # drop header
        seen = set(map(tuple, existing))
        added = [e for e in section_edges if e not in seen]
        with open(deriv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Child_ID", "Parent_ID"])
            w.writerows(existing)
            w.writerows(added)

    os.remove("cldf/parameters.csv")
    print(
        f"unified cldf/forms.csv: {len(etyma_rows)} etyma "
        f"({len(folded)} folded self-reflexes, {n_merged} merged addenda) + {n_reflex} reflexes "
        f"+ {n_variant} variants + {n_section} promoted section-forms + {n_borrowed} borrowed; "
        f"applied {n_curated_borrowings} curated cross-dictionary borrowings; "
        f"attached {n_nuristani_reflexes} PNur/IA nodes as Proto-II reflexes; "
        f"applied {n_nuristani_borrowings} Strand OIA loan branches "
        f"with {n_nuristani_borrowed_descendants} direct borrowed descendants; "
        f"merged {n_strand_oia_redirects} duplicate Strand OIA heads and redirected "
        f"{n_strand_oia_references} references; "
        f"removed parameters.csv",
        file=sys.stderr,
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--borrowings-only"]:
        apply_borrowings_to_unified()
    else:
        main()
