"""Reproducible, source-local 15-column Malamuthan proposal from the full chapter."""
import csv
import hashlib
import io
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
units = [json.loads(s) for s in (HERE / "physical-units-first-reading.jsonl").read_text().splitlines()]
graph = json.loads((HERE / "graph-decisions-proposal.json").read_text())
variants = {e["child"]: e["parent"] for e in graph["variant_edges"]}
assert len(variants) == 2
COLUMNS = ["Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
           "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
           "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags"]

def metadata(u):
    section = u["section"]
    gloss = u["source_gloss"]
    form = u["target_form"]
    tag = ""
    note = u["source_claim"]
    if u["note"]:
        note += " " + u["note"]
    if section in {"initial-v-to-m", "medial-v-b", "final-n", "final-n-formal",
                   "final-m-to-n", "nr-to-nj", "alveolar-to-palatal", "y-insertion",
                   "l-to-y", "l-to-yj", "l-to-v", "yy-s-to-yj"}:
        if gloss in {"bangle", "house", "boy", "thief", "nail", "hair", "banana", "time",
                     "forehead", "cockroach", "peg", "rice", "fox", "row", "seed",
                     "rain", "stream", "hole", "worm", "neck", "hand", "moustache"}:
            tag = "noun"
        elif gloss in {"let come", "will call", "come"}:
            tag = "verb"
            if gloss == "will call": tag += " fut"
        elif gloss in {"these people", "this-she", "my", "your"}:
            tag = "pron"
            if gloss == "these people": tag += " 3pl"
            if gloss in {"my", "your"}: tag += " gen poss"
    if section == "nr-cluster":
        tag = "uncertain"
        note += " The source does not supply a lexical gloss."
    if section == "final-n-formal": tag += " formal uncertain"
    if section == "pronouns":
        tag = "pron " + u["note"]
        note = u["source_claim"]
        if form == "ibaḷu":
            note += " The source separately prints this same spelling with both 'she' and 'they'."
    if section == "pronoun-cases":
        tag = "pron " + (u["note"] if u["note"] != "soc" else "uncertain")
        note = u["source_claim"] + " Printed case label: " + u["note"] + "."
        if gloss == "he" and "gen" in tag:
            note += " The printed English gloss says 'he' despite the genitive label."
    if section == "demonstratives": tag = "demonstrative"
    if section == "demonstratives" and form == "-nee": tag += " dist affix"
    if section == "demonstrative-series":
        tag = "demonstrative"
        note += " No individual sense or proximal/distal alignment is supplied for this cell."
    if section == "case-link":
        tag = "affix" if form == "-in" else "pron interr acc"
        if form == "-in": note += " This is a link morph, not a separately glossed case suffix."
    if section == "genitive-no-link": tag = "noun gen"
    if section == "genitive-ca":
        tag = "suffix gen" if form == "-ca" else "gen"
        if form == "iñca": tag += " pron poss"
        elif form in {"makaca", "kaṇṇica"}: tag += " noun"
    if section == "past-uttu": tag = "suffix pret" if form == "-uttu" else "verb pret"
    if section == "negative":
        tag = "neg" if form == "illa" else "neg multiword-expression"
        if form == "illa": note += " The source explicitly gives no separate Malamuthan equivalent for Malayalam alla."
    if u["malayalam_control"]:
        note += " Printed Malayalam comparison: " + u["malayalam_control"] + "."
    return " ".join(dict.fromkeys(tag.split())), note.strip()

rows, audit = [], []
for u in units:
    key = u["unit_key"]
    emitted = u["scope"] == "target" and bool(u["target_form"])
    row = None
    if emitted:
        tag, note = metadata(u)
        row = ["Malamuthan", "", u["target_form"], u["source_gloss"], "", "", note,
               f"nair2013malamuthan[p. {u['printed_page']}, {u['section'].replace('-', ' ')}]",
               "", "", key, variants.get(key, ""), "", "", tag]
        assert len(row) == len(COLUMNS)
        rows.append(row)
    audit.append({**u, "disposition": "source-stage-proposal" if row else "audit-only",
                  "proposed_entry_key": key if row else None,
                  "proposed_tags": row[14] if row else None,
                  "proposed_notes": row[6] if row else None,
                  "exclusion_reason": None if row else (
                      "No source-attributed Malamuthan lexical form" if u["scope"] != "malayalam-control"
                      else "Malayalam comparison, not Malamuthan")})

assert len(units) == 79 and len(rows) == 67 and len(audit) == 79
assert len({r[10] for r in rows}) == len(rows)
assert all(unicodedata.is_normalized("NFC", cell) for r in rows for cell in r)
assert {r[10]: r[11] for r in rows if r[11]} == variants
assert set(variants.values()) <= {r[10] for r in rows}
assert all(not r[12] and not r[13] for r in rows)
out = io.StringIO(newline="")
csv.writer(out).writerows(rows)
(HERE / "proposed.csv").write_text(out.getvalue())
(HERE / "proposal-audit.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audit))
status = {"status": "full-chapter-proposal-pending-independent-review",
          "source_pdf_sha256": units[0]["source_pdf_sha256"],
          "physical_units": len(units), "proposed_rows": len(rows), "audit_only_units": len(audit) - len(rows),
          "proposed_variant_edges": len(variants),
          "csv_sha256": hashlib.sha256((HERE / "proposed.csv").read_bytes()).hexdigest(),
          "audit_sha256": hashlib.sha256((HERE / "proposal-audit.jsonl").read_bytes()).hexdigest(),
          "canonical_changes": False, "full_build": "deferred-by-user", "browser_db": "not-authorized"}
(HERE / "proposal-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
print(status)
