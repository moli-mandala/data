"""Draft every source-attributed Betta Kurumba occurrence on Nair 2013 pp. 12–18.

The physical-unit first reading is the authority. Arrowed realizations are the
display form; the printed segmented side remains in Notes and the complete
physical record remains in the audit. No reconstructed pronunciation is added.
"""
import collections
import csv
import hashlib
import io
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
units = [json.loads(s) for s in (HERE / "physical-units-first-reading.jsonl").read_text().splitlines()]
assert len(units) == 225 and len({u["unit_key"] for u in units}) == 225
COLS = ["Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
        "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
        "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags"]


def tags_for(u):
    s, lab, f = u["section"], u["grammatical_label"], u["form"]
    tags = []
    def add(*ts):
        for t in ts:
            if t not in tags:
                tags.append(t)
    if s == "gender-nouns": add("noun", lab)
    if s in {"plural-markers", "diagnostic-plural"}: add("suffix", "pl")
    if s == "plural-examples": add("noun", "pl")
    if s == "diagnostic-nine" or s == "numeral-nine": add("num")
    if s == "numeral-adjectives": add("num", "adj")
    if s in {"diagnostic-case-markers", "accusative-markers", "instrumental-marker",
             "dative-markers", "ablative-markers", "locative-markers", "causative-case-marker",
             "purposive-marker"}:
        add("suffix")
    for word, tag in (("accusative", "acc"), ("instrumental", "instr"), ("dative", "dat"),
                      ("ablative", "abl"), ("locative", "loc"), ("genitive", "gen"),
                      ("purposive", "purposive")):
        if word in s or word in lab: add(tag)
    if s == "sociative-sentence": add("multiword-expression", "sentential", "instr")
    if s == "diagnostic-link": add("affix")
    if s in {"diagnostic-participles", "relative-participle-markers", "verbal-participle-markers"}:
        add("suffix", "participle")
    if s == "relative-participle-examples": add("verb", "participle")
    if s == "verbal-participle-examples": add("verb", "participle")
    if s == "personal-pronouns":
        add("pron")
        for t in lab.split():
            if t in {"1sg", "1pl", "2sg", "2pl", "3sg", "3pl", "proximal", "distal", "inclusive", "exclusive", "oblique"}:
                add({"proximal": "prox", "distal": "dist", "oblique": "obl"}.get(t, t))
    if s in {"interrogative-bases", "interrogative-pronouns"}: add("pron", "interr")
    if s == "indefinite-marker": add("suffix", "indef")
    if s == "indefinite-examples": add("pron", "indef")
    if s in {"past-markers", "nonpast-markers"}: add("suffix", "pret" if s == "past-markers" else "non-past")
    if s in {"past-examples", "nonpast-examples"}: add("verb", "pret" if s == "past-examples" else "non-past")
    if s == "transitive-markers": add("suffix", "tr")
    if s == "transitive-examples": add("verb", "tr")
    if s == "infinitive-markers": add("suffix", "inf")
    if s == "infinitive-examples": add("verb", "inf")
    if "negative" in s:
        add("neg")
        if "markers" in s or "marker" in s: add("suffix")
        elif "examples" in s: add("verb")
        if "past-negative" in s: add("pret")
        if "nonpast-negative" in s: add("non-past")
    if s in {"modal-ku", "potential-aappa", "prohibitive-marker", "attempted-marker",
             "negative-potential", "definitive-auxiliaries", "completive-marker",
             "durative-auxiliaries"}:
        add("auxiliary")
        if f.startswith("-"): add("suffix")
    if s in {"modal-ku-examples", "potential-examples", "negative-potential-examples",
             "prohibitive-example", "attempted-examples", "past-definitive-examples",
             "nonpast-definitive-example", "completive-examples", "durative-examples",
             "reflexive-examples"}:
        add("verb")
    if "potential" in s: add("modal")
    if "prohibitive" in s: add("prohibitive", "neg")
    if "completive" in s: add("completive")
    if "reflexive" in s: add("refl")
    if s == "reflexive-base": add("pron", "refl", "obl")
    if "definitive" in s and s.endswith("examples"): add("pret")
    if s == "person-suffix-table":
        add("suffix")
        person, tense, *_ = lab.split()
        add(person, "pret" if tense == "past" else "non-past")
    if s == "negative-person-examples": add("verb", "neg", "pret")
    return " ".join(tags)


def note_for(u):
    bits = []
    if u["surface"]:
        bits.append("Printed segmented analysis: " + u["form"] + ".")
    s = u["section"]
    if s in {"accusative-markers", "dative-markers", "ablative-markers", "locative-markers",
             "instrumental-marker", "locative-pili-examples", "causative-case-marker",
             "purposive-marker", "past-markers", "nonpast-markers", "nonpast-examples",
             "transitive-markers", "diagnostic-link", "diagnostic-participles", "indefinite-marker",
             "infinitive-markers", "durative-negative-markers", "past-negative-markers",
             "negative-potential", "prohibitive-marker", "verbal-participle-markers",
             "modal-ku", "potential-aappa", "attempted-marker", "definitive-auxiliaries",
             "completive-marker", "durative-auxiliaries", "reflexive-base"}:
        if u["source_claim"]:
            bits.append(u["source_claim"])
    if s == "verbal-participle-examples" and u["form"] == "ba-nd-um":
        bits.append("The printed -um conflicts with the prose rule stating -un in this environment.")
    if s == "diagnostic-case-markers" and u["form"] == "-ooda":
        bits.append("This diagnostic prints plain d; the later instrumental paradigm prints underdotted ḍ.")
    if s == "durative-negative-examples":
        bits.append("Both English descriptions are printed for this one form.")
    if s == "person-suffix-table":
        bits.append("One person-by-tense table cell; slash alternatives are source co-listed.")
    return " ".join(bits)


rows, audit = [], []
for u in units:
    emitted = u["role"] == "target"
    made = []
    if emitted:
        raw_form = u["surface"] or u["form"]
        assert raw_form
        alts = raw_form.split("/") if u["section"] == "person-suffix-table" and "/" in raw_form else [raw_form]
        for i, form in enumerate(alts, 1):
            key = u["unit_key"] if len(alts) == 1 else f'{u["unit_key"]}:variant:{i}'
            parent = f'{u["unit_key"]}:variant:1' if i > 1 else ""
            gloss = u["gloss"] or (u["grammatical_label"] if u["section"] == "person-suffix-table" else "")
            row = ["BettaKurumba", "", form, gloss, "", "", note_for(u),
                   f'nair2013bettakurumba[p. {u["printed_page"]}, {u["section"].replace("-", " ")}]',
                   "", "", key, parent, "", "", tags_for(u)]
            assert len(row) == len(COLS)
            rows.append(row)
            made.append(key)
    audit.append({**u, "disposition": "proposed-source-attestation" if made else "audit-only",
                  "proposed_entry_keys": made,
                  "display_form": (u["surface"] or u["form"]) if made else None,
                  "exclusion_reason": None if made else {
                      "attribution-held": "The quoted ethnonym's lect is not unambiguously Betta Kurumba",
                      "comparison-control": "Other-language generic comparison",
                      "comparison-context": "No printed comparison witness",
                      "zero-exponent": "Zero exponent is not a pronounceable form",
                      "blank": "No printed target form",
                      "documentary": "Source context, not a lexical attestation",
                      "bibliographic-context": "Reported authority, not a lexical attestation",
                  }[u["role"]]})

assert len(audit) == 225
assert len(rows) == 206
assert len({r[10] for r in rows}) == len(rows)
assert all(r[2] and r[10] and r[7] for r in rows)
assert all(unicodedata.is_normalized("NFC", c) for r in rows for c in r)
assert all(not r[12] and not r[13] for r in rows)
keys = {r[10] for r in rows}
assert all(not r[11] or r[11] in keys for r in rows)
out = io.StringIO(newline="")
csv.writer(out).writerows(rows)
(HERE / "proposed.csv").write_text(out.getvalue())
(HERE / "proposal-audit.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audit))
status = {"status": "whole-chapter-proposal-pending-independent-review",
          "source_pdf_sha256": units[0]["source_pdf_sha256"],
          "physical_units": len(units), "source_target_units": sum(u["role"] == "target" for u in units),
          "proposed_rows": len(rows), "audit_only_units": sum(not a["proposed_entry_keys"] for a in audit),
          "variant_edges": sum(bool(r[11]) for r in rows),
          "csv_sha256": hashlib.sha256((HERE / "proposed.csv").read_bytes()).hexdigest(),
          "audit_sha256": hashlib.sha256((HERE / "proposal-audit.jsonl").read_bytes()).hexdigest(),
          "canonical_changes": False, "full_build": "deferred-by-user", "browser_db": "not-authorized"}
(HERE / "proposal-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(status, ensure_ascii=False))
