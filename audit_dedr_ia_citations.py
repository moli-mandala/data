"""Reconcile DEDR footers that cite an Indo-Aryan *form* (no Turner number) with CDIAL entries.

``cross_family.py`` only installs comparisons whose source prints a dictionary number.  Some 450
DEDR footers instead cite a Sanskrit/Prakrit/Pali/Marathi form ("Cf. Skt. X", "? < Skt. X"), which
the audit lists as "unresolved".  The table below is the manual review of those footers, resolving
the cited form to a CDIAL etymon by hand; a second block records the few CDIAL entries that quote a
Dravidian form by language name without a DED number, and one DBIA-backed pair.  Entries whose
cited word CDIAL never entered (kīcaka, śaṣkulī, kaupīna, biruda, ...) stay unresolved and are
reported as ``no-cdial-target`` in the audit output.

This is an editorial table, not a generator: relation, direction and confidence follow the
source's wording (DEDR's "?" and "Cf." are carried as low confidence).

Usage: ``uv run python audit_dedr_ia_citations.py [--install]``
"""

import argparse
import csv
import re
from pathlib import Path

from data.cross_family import dedr_citation_locators


ROOT = Path(__file__).parent
AUDIT = ROOT / "data/cross-family-comparisons-audit.csv"
MANUAL = ROOT / "data/manual-cross-family-comparisons.csv"
EXTRACTED = ROOT / "data/cross-family-comparisons.csv"
DBIA = ROOT / "data/dbia/comparisons.csv"
COMPILED = ROOT / "cldf/comparisons.csv"
FORMS = ROOT / "cldf/forms.csv"
OUTPUT = ROOT / "data/dedr-ia-citation-audit.csv"

COMPARISON_COLUMNS = [
    "ID", "Entry_ID", "Compared_Entry_ID", "Relation", "Direction", "Confidence",
    "Source", "Evidence",
]

# (DEDR id, CDIAL id): (citing side, relation, direction, confidence, reviewer note)
PROPOSALS = {
    ("d104", "644"): ("dedr", "loan", "entry-from-compared", "medium", ""),
    ("d1054", "152"): ("dedr", "related", "undetermined", "low", "CDIAL homes M. õvā under ajamōda"),
    ("d1076", "2822"): ("dedr", "related", "undetermined", "low", ""),
    ("d1098", "2626"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d117", "308"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d118", "308"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d1152", "2656"): ("dedr", "related", "undetermined", "low", "two-sided citation"),
    ("d1305", "2920"): ("dedr", "related", "undetermined", "low", ""),
    ("d1404", "4424"): ("dedr", "influence", "entry-from-compared", "medium", ""),
    ("d1431", "2905"): ("dedr", "related", "undetermined", "low", "CDIAL entry lacks the weight/coin sense"),
    ("d1466", "3674"): ("dedr", "influence", "entry-from-compared", "medium", ""),
    ("d1611", "4240"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d1623", "3170"): ("dedr", "related", "undetermined", "low", ""),
    ("d163", "434"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d1657", "4494"): ("dedr", "related", "undetermined", "low", "4494 is already linked to d1871 kūkai; this is the sibling owl set"),
    ("d1684", "3504"): ("dedr", "related", "undetermined", "low", "3504 already ← Drav. d2209; DEDR 1684 is a second candidate"),
    ("d1764", "3483"): ("dedr", "related", "undetermined", "medium", ""),
    ("d1785", "3320"): ("dedr", "related", "undetermined", "low", ""),
    ("d187", "576"): ("dedr", "related", "undetermined", "low", ""),
    ("d196", "1347"): ("dedr", "loan", "entry-from-compared", "medium", ""),
    ("d199", "11002"): ("dedr", "related", "undetermined", "medium", ""),
    ("d199", "695"): ("dedr", "related", "undetermined", "low", ""),
    ("d201", "10679"): ("dedr", "loan", "entry-from-compared", "high", ""),
    ("d2058", "4336"): ("dedr", "influence", "entry-from-compared", "medium", ""),
    ("d2076", "3257"): ("dedr", "related", "undetermined", "low", "CDIAL 3257 glosses 'corpse' (itself ← Drav.); MW has both senses"),
    ("d2201", "4271"): ("dedr", "related", "undetermined", "low", "4271 already ← Drav. d2069 in the 'kernel' sense"),
    ("d2238", "3350"): ("dedr", "related", "undetermined", "low", ""),
    ("d2272", "4963"): ("dedr", "related", "undetermined", "low", ""),
    ("d2327", "4594"): ("dedr", "related", "undetermined", "medium", ""),
    ("d2328", "4983a"): ("dedr", "influence", "entry-from-compared", "low", ""),
    ("d2337", "4673"): ("dedr", "related", "undetermined", "low", "4673 already ← Drav. d2335 'slap'"),
    ("d236", "708"): ("dedr", "loan", "compared-from-entry", "medium", "Dravidian → IA claim"),
    ("d2477", "5213"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d2484", "3675"): ("dedr", "related", "undetermined", "low", "CDIAL homes exactly this M. sār under 3675"),
    ("d2548", "13388"): ("dedr", "related", "undetermined", "low", ""),
    ("d26", "55"): ("dedr", "loan", "entry-from-compared", "medium", "also DBIA 5"),
    ("d2627", "4844"): ("dedr", "related", "undetermined", "low", ""),
    ("d2629", "4843"): ("dedr", "loan", "entry-from-compared", "medium", ""),
    ("d2629", "4910"): ("dedr", "loan", "entry-from-compared", "medium", ""),
    ("d2647", "13453"): ("dedr", "related", "undetermined", "low", ""),
    ("d2703", "13512"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d2748", "12243"): ("dedr", "related", "undetermined", "low", ""),
    ("d2775", "5779"): ("dedr", "influence", "entry-from-compared", "low", ""),
    ("d2810", "13935"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d3002", "5622"): ("dedr", "related", "undetermined", "medium", ""),
    ("d3013", "5426"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d3020", "6618"): ("dedr", "related", "undetermined", "low", ""),
    ("d3051", "6128"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d3055", "5668"): ("dedr", "related", "undetermined", "low", ""),
    ("d3080", "6632"): ("dedr", "related", "undetermined", "medium", ""),
    ("d3081", "5686a"): ("dedr", "related", "undetermined", "medium", ""),
    ("d3088", "5617"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d3128", "5703"): ("dedr", "related", "undetermined", "low", ""),
    ("d3160", "5754"): ("dedr", "related", "undetermined", "low", ""),
    ("d3163", "5774"): ("dedr", "related", "undetermined", "medium", ""),
    ("d3164", "5779"): ("dedr", "related", "undetermined", "low", ""),
    ("d3182", "13766"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d3298", "6064"): ("dedr", "related", "undetermined", "low", "CDIAL gloss lacks the plant sense"),
    ("d3402", "12573"): ("dedr", "related", "undetermined", "low", ""),
    ("d349", "1127"): ("dedr", "related", "undetermined", "medium", ""),
    ("d3568", "6924"): ("dedr", "related", "undetermined", "medium", ""),
    ("d3694", "10482"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d390", "4746"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d3930", "9650"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d3968", "8435"): ("dedr", "related", "undetermined", "low", ""),
    ("d3970", "13809"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d3973", "7910"): ("dedr", "related", "undetermined", "low", ""),
    ("d400", "1380"): ("dedr", "related", "undetermined", "low", ""),
    ("d4075", "8034"): ("dedr", "related", "undetermined", "medium", ""),
    ("d4096", "8128"): ("dedr", "related", "undetermined", "low", ""),
    ("d4183", "8174"): ("dedr", "related", "undetermined", "low", ""),
    ("d4203", "8264"): ("dedr", "related", "undetermined", "low", ""),
    ("d4518", "8264"): ("dedr", "related", "undetermined", "low", ""),
    ("d4316", "9553"): ("dedr", "related", "undetermined", "low", ""),
    ("d4323", "8297"): ("dedr", "related", "undetermined", "low", ""),
    ("d4384", "8384"): ("dedr", "related", "undetermined", "medium", ""),
    ("d4394", "8377"): ("dedr", "related", "undetermined", "low", ""),
    ("d4442", "8164"): ("dedr", "related", "undetermined", "low", "8164 already ↔ d4388; sibling set"),
    ("d4459", "8391"): ("dedr", "related", "undetermined", "low", ""),
    ("d4525", "9278"): ("dedr", "related", "undetermined", "low", ""),
    ("d4672", "9731"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d4674", "9731"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d4742", "9902"): ("dedr", "related", "undetermined", "medium", ""),
    ("d4790", "11469"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d4879", "12659"): ("dedr", "related", "undetermined", "low", "M. miśī 'moustache' reflex not verified in cdial.csv; CDIAL's M. miśī is 'tooth-paste' (10137)"),
    ("d4916", "10184"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d4975", "10211"): ("dedr", "related", "undetermined", "low", ""),
    ("d4977", "10211"): ("dedr", "related", "undetermined", "low", ""),
    ("d5026", "10231"): ("dedr", "loan", "entry-from-compared", "medium", ""),
    ("d5138", "10348"): ("dedr", "related", "undetermined", "low", ""),
    ("d5256", "11199"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d5260", "11311"): ("dedr", "related", "undetermined", "low", ""),
    ("d5438", "11900"): ("dedr", "related", "undetermined", "low", ""),
    ("d5472", "11714"): ("dedr", "related", "undetermined", "low", ""),
    ("d550", "13385"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d5532", "12139"): ("dedr", "related", "undetermined", "low", ""),
    ("d57", "168"): ("dedr", "related", "undetermined", "medium", ""),
    ("d657", "10803"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d76", "179"): ("dedr", "related", "undetermined", "low", "CDIAL homes M. aṭṇẽ under *aṭṭ 'obstruct' (already ↔ d83)"),
    ("d769", "2462"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d814", "625"): ("dedr", "related", "undetermined", "medium", ""),
    ("d921", "13281"): ("dedr", "related", "undetermined", "low", ""),
    ("d1198", "2710"): ("dedr", "related", "undetermined", "low", ""),
    ("d181", "578"): ("dedr", "related", "undetermined", "low", ""),
    ("d35", "118"): ("dedr", "related", "undetermined", "low", ""),
    ("d4434", "8399"): ("dedr", "related", "undetermined", "low", "8399 already ← Drav. d4587"),
    ("d1301", "3848"): ("dedr", "related", "undetermined", "low", ""),
    ("da1", "2978"): ("dedr", "loan", "entry-from-compared", "medium", ""),
    ("da5", "13113"): ("dedr", "loan", "entry-from-compared", "low", "CDIAL glosses *satyakāra 'truthful'; same formation"),
    ("da8", "10561"): ("dedr", "loan", "entry-from-compared", "high", ""),
    ("da10", "1110"): ("dedr", "loan", "entry-from-compared", "high", ""),
    ("da11", "13307"): ("dedr", "loan", "entry-from-compared", "high", ""),
    ("da49", "9042"): ("dedr", "loan", "entry-from-compared", "low", "check that CDIAL 9042 carries the 'comb' reflexes"),
    ("da50", "9440"): ("dedr", "loan", "entry-from-compared", "low", ""),
    ("d1688", "3260"): ("cdial", "related", "undetermined", "medium", "d1688 linked to 3261/7647/9124 but not 3260"),
    ("d4806", "10043"): ("cdial", "related", "undetermined", "medium", ""),
    ("d2716", "4873"): ("cdial", "related", "undetermined", "low", "two-sided citation"),
    ("d5540", "12115"): ("dbia", "loan", "entry-from-compared", "medium", ""),}

# CDIAL entries quoting a Dravidian form by language name: printed bracket text.
CDIAL_EVIDENCE = {
    "3260": "[Cf. Kan. kuṇṭa 'cripple', Tel. kuṇṭi 'lame'. — See list s.v. kuṇṭha-]",
    "10043": "[Cf. Kui māṇi 'bamboo']",
    "4873": "[Cf. cuṇṭī-, °ṭikā- f. 'small well' Suśr., °ṭā-, cuṇḍhī- f., cuṇḍya-, cūḍā- f., °ḍaka-, "
             "cūtaka- m. lex. Non-Aryan, but comparison with Tam. coṭṭai (EWA i 394) not convincing] "
             "— DEDR 2716: '? Cf. Skt. cuṇḍhi- small pond, Pkt. cuṇḍhī- natural pool; Skt. cuṇṭī- well'",
}
DBIA_SOURCES = {
    "12115": ("dbia[p. 61, no. 336]",
              "DBIA 336 lists Ta. vēlai, Ma. vēla, Te. vēla 'time, limit of time' under Skt. vēlā-; "
              "DEDR 5540 vēlai 'work, time' carries these forms without the comparison."),
}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dedr_footers():
    """DEDR entry id -> printed footer text for the audit's unresolved Indo-Aryan citations."""
    return {
        row["Source_Entry_ID"]: " ".join(row["Evidence"].split())
        for row in read(AUDIT)
        if row["Source_Dictionary"] == "dedr" and row["Status"] == "unresolved"
    }


def appendix_ids():
    """Compiled ``f_`` ids of DEDR appendix entries (``da<n>`` in the source tables)."""
    mapping = {}
    with FORMS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            match = re.search(r"dedr\[appendix entry (\d+)", row["Source"])
            if match and row["Status"] == "entry":
                mapping["da" + match.group(1)] = row["ID"]
    return mapping


def source_locator(dedr_id, footer):
    printed = dedr_id[2:] if dedr_id.startswith("da") else dedr_id[1:]
    kind = "appendix entry" if dedr_id.startswith("da") else "entry"
    locators = [f"{kind} {printed}", *dedr_citation_locators(footer)]
    return f"dedr[{', '.join(locators)}]"


def comparison_rows(footers):
    for (dedr_id, cdial_id), (side, relation, direction, confidence, note) in PROPOSALS.items():
        if side == "dedr":
            footer = footers.get(dedr_id)
            if footer is None:
                raise ValueError(f"{dedr_id} is not an unresolved DEDR citation in {AUDIT.name}")
            evidence = footer + (f" — {note}" if note else "")
            yield {
                "ID": f"dedr:{dedr_id}:cdial:{cdial_id}", "Entry_ID": dedr_id,
                "Compared_Entry_ID": cdial_id, "Relation": relation, "Direction": direction,
                "Confidence": confidence, "Source": source_locator(dedr_id, footer),
                "Evidence": evidence,
            }
        elif side == "cdial":
            yield {
                "ID": f"cdial:{cdial_id}:dedr:{dedr_id}", "Entry_ID": cdial_id,
                "Compared_Entry_ID": dedr_id, "Relation": relation,
                "Direction": {"entry-from-compared": "compared-from-entry",
                              "compared-from-entry": "entry-from-compared"}.get(direction, direction),
                "Confidence": confidence, "Source": f"CDIAL[entry {cdial_id}]",
                "Evidence": CDIAL_EVIDENCE[cdial_id] + (f" — {note}" if note else ""),
            }
        else:
            source, evidence = DBIA_SOURCES[cdial_id]
            yield {
                "ID": f"dedr:{dedr_id}:cdial:{cdial_id}", "Entry_ID": dedr_id,
                "Compared_Entry_ID": cdial_id, "Relation": relation, "Direction": direction,
                "Confidence": confidence, "Source": source, "Evidence": evidence,
            }


def validate(rows, valid_ids, existing_ids):
    seen = set()
    for row in rows:
        cid = row["ID"]
        if cid in seen or cid in existing_ids:
            raise ValueError(f"duplicate comparison ID {cid}")
        seen.add(cid)
        for key in ("Entry_ID", "Compared_Entry_ID"):
            if row[key] not in valid_ids:
                raise ValueError(f"{cid}: unknown entry {row[key]}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+\[[^\]]+\]", row["Source"]):
            raise ValueError(f"{cid}: bad source locator {row['Source']!r}")
        if not row["Evidence"].strip():
            raise ValueError(f"{cid}: empty evidence")


def append(path, rows):
    existing = read(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing + rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true", help="append to the manual and compiled tables")
    args = parser.parse_args()

    footers = dedr_footers()
    appendix = appendix_ids()
    rows = list(comparison_rows(footers))

    with FORMS.open(encoding="utf-8", newline="") as handle:
        valid_ids = {row["ID"] for row in csv.DictReader(handle)} | set(appendix)
    existing_ids = {row["ID"] for path in (EXTRACTED, MANUAL, DBIA) for row in read(path)}
    linked_pairs = set()
    for path in (EXTRACTED, MANUAL, DBIA, COMPILED):
        for row in read(path):
            linked_pairs.add((row["Entry_ID"], row["Compared_Entry_ID"]))
            linked_pairs.add((row["Compared_Entry_ID"], row["Entry_ID"]))
    compiled_to_source = {v: k for k, v in appendix.items()}
    already = [
        row["ID"] for row in rows
        if (row["Entry_ID"], row["Compared_Entry_ID"]) in linked_pairs
        or (compiled_to_source.get(row["Entry_ID"], row["Entry_ID"]), row["Compared_Entry_ID"]) in linked_pairs
    ]
    new_rows = [row for row in rows if row["ID"] not in existing_ids and row["ID"] not in already]
    validate(new_rows, valid_ids, existing_ids)

    if args.install and new_rows:
        append(MANUAL, new_rows)
        compiled_rows = []
        for row in new_rows:
            row = dict(row)
            row["Entry_ID"] = appendix.get(row["Entry_ID"], row["Entry_ID"])
            compiled_rows.append(row)
        append(COMPILED, compiled_rows)
        print(f"installed {len(new_rows)} comparisons into {MANUAL.name} and {COMPILED.name}")

    proposed = {dedr_id for dedr_id, _ in PROPOSALS}
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["DEDR_ID", "Status", "CDIAL_IDs", "Footer"], lineterminator="\n")
        writer.writeheader()
        for dedr_id, footer in sorted(footers.items(), key=lambda item: (item[0][:2], int(re.sub(r"\D", "", item[0]) or 0))):
            targets = sorted(c for d, c in PROPOSALS if d == dedr_id)
            status = "proposed" if dedr_id in proposed else "no-cdial-target"
            writer.writerow({"DEDR_ID": dedr_id, "Status": status, "CDIAL_IDs": " ".join(targets), "Footer": footer})
    print(
        f"{len(footers)} unresolved DEDR citations: {len(proposed)} proposed "
        f"({len(rows)} pairs, {len(new_rows)} new, {len(already)} already linked), "
        f"{len(footers) - len(proposed)} without a CDIAL target"
    )


if __name__ == "__main__":
    main()
