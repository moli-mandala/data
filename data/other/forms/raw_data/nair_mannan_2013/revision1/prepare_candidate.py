"""Complete source-local Mannan review proposal; never installs canonical data."""
import csv
import hashlib
import io
import json
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "data/SOURCE_INGESTION_CHECKLIST.md").exists())
READING = HERE / "physical-units-first-reading.jsonl"
EXPECTED = "7e756a167297b07d0f03cddf135bc5ed97a6e102e21ab99947d1de7d6148edf5"
COLUMNS = ["Language_ID","Parameter_ID","Form","Gloss","Native","Phonemic","Notes","Source","Cognateset","Etymology","Entry_Key","Variant_Of_Key","Borrowed_From_Key","Derivation_Parent_Keys","Tags"]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(READING) == EXPECTED
    units = [json.loads(s) for s in READING.read_text().splitlines()]
    rows, audit = [], []
    symbols = Counter()
    for unit in units:
        emitted = []
        records = []
        for witness in unit["source_witnesses"]:
            scope = witness["language_scope"]
            form = witness["literal_first_reading"]
            key = witness["witness_key"]
            assert unicodedata.is_normalized("NFC",form)
            assert not any("\ue000" <= c <= "\uf8ff" for c in form)
            language = {"Mannan target":"Mannan", "Malayalam comparison":"Malayalam"}.get(scope)
            if unit["disposition"] != "review_candidate": language = None
            reason = "eligible source-explicit named-language attestation" if language else "documentary/context or unspecified generic-language record; preserved without fabricated lexical route"
            records.append(dict(witness_key=key, raw_source_record=witness, proposed_language_id=language,
                disposition="review_candidate" if language else "audit_only", reason=reason))
            if language is None: continue
            gloss = witness["printed_gloss"]
            tags = set(witness["provisional_tags"].split())
            if any(c.isspace() for c in form): tags.add("multiword-expression")
            if gloss in ("one","three","seven"): tags.add("num")
            section = unit["source_unit_key"].split(":")[-2]
            notes = unit["source_claim"]
            if language == "Malayalam": notes = "Explicit Malayalam comparison. " + notes
            etymology = ""
            if section == "other-plural" and unit["kind"] == "bound morphology":
                etymology = "Nair tentatively suggests a relationship with the usual Dravidian plural -kaḷ (p. 47); the comparison has no unique language/stage attribution, so no ancestry edge is asserted."
            if section == "demonstratives":
                tags.add("dist" if unit["source_claim"].startswith("Dist") else "prox")
                notes += "; source gives gender/distance but does not repeat number headings for this subsection."
            if section == "vocabulary" and gloss == "calling a bor":
                tags.add("uncertain")
                notes = "Source prints ‘calling a bor’; apparent English typographical anomaly retained, without silently supplying boy."
            if section == "vocabulary" and gloss == "an expression of surprise,":
                gloss = "an expression of surprise"
                tags.add("interj")
                notes = "Source gloss ends in a comma and has an unmatched opening quote; raw punctuation is retained in audit."
            row = [language,"",form,gloss,"","",notes,
                f"nair2013mannan[p. {unit['printed_page']}]","",etymology,key,"","",""," ".join(sorted(tags))]
            assert len(row) == 15
            rows.append(row); emitted.append(key)
            for char in unicodedata.normalize("NFD",form): symbols[char] += 1
        audit.append(dict(source_unit_key=unit["source_unit_key"], printed_page=unit["printed_page"],
            printed_pages=unit.get("printed_pages",[unit["printed_page"]]), source_record=unit,
            proposed_entry_keys=emitted, witness_dispositions=records, accepted_for_installation=False,
            disposition="review_candidate" if emitted else unit["disposition"]))
    assert len(units) == len(audit) == 140
    assert len(rows) == 154 and Counter(r[0] for r in rows) == {"Mannan":123,"Malayalam":31}
    assert len({r[10] for r in rows}) == len(rows)
    assert {k for a in audit for k in a["proposed_entry_keys"]} == {r[10] for r in rows}
    out = io.StringIO(newline=""); csv.writer(out).writerows(rows)
    (HERE / "proposed.csv").write_text(out.getvalue())
    (HERE / "proposal-audit.jsonl").write_text("".join(json.dumps(a,ensure_ascii=False,sort_keys=True)+"\n" for a in audit))
    (HERE / "symbol-inventory.json").write_text(json.dumps({"nfd_symbols":[dict(character=c,codepoint=f"U+{ord(c):04X}",count=n) for c,n in sorted(symbols.items())],
        "conversion":"convert:false; preserve all literal source doubling, dotbelow, lowline, tilde and breve; no invented IPA",
        "native_phonemic":"blank; Roman source transcription is Form/Original, not independently established IPA or native script"},ensure_ascii=False,indent=2)+"\n")
    status = dict(status="frozen owner review candidate; independent original/metadata/output acceptance pending",
        audit_groups=len(audit), source_witness_records=208, emitted_rows=len(rows), languages=dict(Counter(r[0] for r in rows)),
        excluded_witness_records=54, graph_edges=0, inputs={"first_reading":EXPECTED,"builder":sha(Path(__file__))},
        outputs={name:sha(HERE / name) for name in ("proposed.csv","proposal-audit.jsonl","symbol-inventory.json")},
        deferred=["complete independent original review","fresh final-output >=20 audit","canonical source-stage install authorization","full CLDF/full suite/compiled identity/graph validation/browser QA"])
    (HERE / "proposal-status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(status,ensure_ascii=False))


if __name__ == "__main__": main()
