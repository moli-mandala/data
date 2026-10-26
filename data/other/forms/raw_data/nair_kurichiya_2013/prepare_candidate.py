"""Reproduce an unreviewed, complete-chapter Kurichiya source-stage proposal."""
import csv
import hashlib
import io
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
WITNESSES = HERE / "first-reading" / "unicode-preview.jsonl"
EXPECTED_SHA = "a0bf4c4529128f72762162619db42bbf0b1fc0c27bee9c6c37d9321e684de458"
assert hashlib.sha256(WITNESSES.read_bytes()).hexdigest() == EXPECTED_SHA
units = [json.loads(s) for s in WITNESSES.read_text().splitlines()]
assert len(units) == 117
graph = json.loads((HERE / "graph-review-proposal.json").read_text())
variants = {edge["child"]: edge["parent"] for edge in graph["proposed_variant_pairs"]}
assert len(variants) == 8

COLUMNS = ["Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes", "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags"]
rows = []
audit = []
target_blocks = {
    "a-to-e", "final-u", "i-to-e", "v-to-b", "i-to-y-or-v", "s-to-y-or-c",
    "medial-elision", "tt-to-rr", "question-laa", "initial-r-loss", "nn-to-nr",
    "past-tense", "pronoun-table", "question-ending", "particle-1", "particle-2",
    "utterance-1", "utterance-2",
}
assert {u["block"] for u in units} == target_blocks | {"inclusive-prose"}

for unit in units:
    key = unit["witness_key"]
    page = unit["pdf_page"]
    side = unit["source_side"]
    block = unit["block"]
    form = unit["literal_unicode_proposal"]
    assert unicodedata.is_normalized("NFC", form)
    assert not any("\ue000" <= c <= "\uf8ff" for c in form)
    excluded = block == "inclusive-prose"
    if excluded:
        audit.append({"witness_key": key, "pdf_page": page, "block": block,
                      "literal_unicode_proposal": form, "disposition": "audit-only repeated prose",
                      "linked_table_cell": unit["linked_table_cell"], "reason": unit["reason"]})
        continue
    language = "Malayalam" if side == "Malayalam" else "Kurichiya"
    gloss = unit.get("printed_gloss", "")
    notes = ""
    tags = ""
    if side == "Malayalam":
        notes = "Explicit Malayalam comparison in Kurichiya chapter."
    if block == "pronoun-table":
        tags = unit["provisional_tags"]
        notes = unit["printed_scope"]
        if unit["typed_uncertainty"]:
            notes += "; " + unit["typed_uncertainty"]
            tags += " uncertain"
    elif block == "question-ending":
        tags = "suffix interr"
        notes = unit["source_analysis"]
    elif block in ("particle-1", "particle-2"):
        tags = "part"
        notes = unit["source_analysis"]
    elif block.startswith("utterance-"):
        tags = "sentential multiword-expression"
        notes = unit["source_analysis"]
    elif block == "question-laa":
        tags = "interr sentential"
    elif block == "past-tense":
        tags = "verb pret"
    elif block == "final-u":
        tags = "copula"
    notes = unicodedata.normalize("NFC", notes.replace("\uf019", "\u0303").replace("\uf02c", "\u0323"))
    if "/" in form:
        # A slash is a printed alternative separator in two Kurichiya cells.
        assert language == "Kurichiya" and block in ("s-to-y-or-c", "medial-elision")
        alternatives = form.split("/")
        assert len(alternatives) == 2 and all(alternatives)
    else:
        alternatives = [form]
    emitted = []
    for i, alternative in enumerate(alternatives, 1):
        entry_key = key if len(alternatives) == 1 else f"{key}:alt{i}"
        row = [language, "", alternative, gloss, "", "", notes,
               f"nair2013kurichiya[p. {page}, {block.replace('-', ' ')}]", "", "",
               entry_key, variants.get(entry_key, ""), "", "", tags]
        assert len(row) == len(COLUMNS)
        rows.append(row)
        emitted.append(entry_key)
    audit.append({"witness_key": key, "pdf_page": page, "block": block,
                  "source_side": side, "source_raw_text_layer": unit["source_raw_text_layer"],
                  "literal_unicode_proposal": form, "printed_gloss": gloss,
                  "disposition": "unreviewed candidate", "proposed_entry_keys": emitted,
                  "variant_of_keys": {child: variants[child] for child in emitted if child in variants},
                  "uncertainty": unit.get("typed_uncertainty", "")})

assert len(audit) == 117
assert len(rows) == 118  # 116 accepted witness slots, plus two printed slash splits
assert sum(r[0] == "Malayalam" for r in rows) == 47
assert sum(r[0] == "Kurichiya" for r in rows) == 71
assert len({r[10] for r in rows}) == len(rows)
assert {r[10]: r[11] for r in rows if r[11]} == variants
assert set(variants.values()) <= {r[10] for r in rows}
out = io.StringIO(newline="")
csv.writer(out).writerows(rows)
(HERE / "proposed.csv").write_text(out.getvalue())
(HERE / "proposal-audit.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audit))
status = {"status": "unreviewed complete-chapter source-stage proposal",
          "pdf_pages": [34, 35, 36], "witness_slots": 117,
          "candidate_rows": 118, "kurichiya_rows": 71, "malayalam_control_rows": 47,
          "audit_only_repeated_prose": 1, "printed_slash_splits": 2,
          "source_explicit_variant_edges": len(variants),
          "csv_sha256": hashlib.sha256((HERE / "proposed.csv").read_bytes()).hexdigest(),
          "audit_sha256": hashlib.sha256((HERE / "proposal-audit.jsonl").read_bytes()).hexdigest(),
          "remaining": ["independent post-graph frozen-output audit", "source metadata/profile validation", "guarded acceptance and installation"]}
(HERE / "proposal-status.json").write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status, indent=2))
