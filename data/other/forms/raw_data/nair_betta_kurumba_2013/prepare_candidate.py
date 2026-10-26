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
import re
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
    if s == "diagnostic-case-markers" and lab == "sociative": add("comitative")
    if s == "sociative-sentence": add("multiword-expression", "sentential", "comitative")
    nominal = {
        "p13:accusative-examples:01": "noun", "p13:accusative-examples:02": "noun",
        "p13:instrumental-examples:01": "proper-noun", "p13:instrumental-examples:02": "noun",
        "p13:dative-examples:01": "pron 3sg", "p13:dative-examples:02": "proper-noun",
        "p13:dative-examples:03": "noun", "p13:dative-examples:04": "pron 3pl",
        "p13:ablative-examples:01": "proper-noun", "p13:ablative-examples:02": "pron 3sg",
        "p13:ablative-examples:03": "noun", "p13:ablative-examples:04": "noun",
        "p14:genitive-examples:01": "noun multiword-expression", "p14:genitive-examples:02": "noun multiword-expression",
        "p14:locative-l-examples:01": "noun", "p14:locative-l-examples:02": "noun",
        "p14:locative-pili-examples:01": "pron 1sg", "p14:locative-pili-examples:02": "proper-noun",
        "p14:causative-case-examples:01": "pron 1sg", "p14:causative-case-examples:02": "noun",
        "p14:purposive-examples:01": "proper-noun", "p14:purposive-examples:02": "noun",
    }
    add(*nominal.get(u["unit_key"].removeprefix("nair2013bettakurumba:"), "").split())
    if s == "diagnostic-link": add("affix")
    if s in {"diagnostic-participles", "relative-participle-markers", "verbal-participle-markers"}:
        add("suffix", "participle")
    if s == "diagnostic-participles" and lab == "relative participle": add("relative")
    if s == "relative-participle-markers": add("relative")
    if s == "relative-participle-examples": add("verb", "participle", "relative")
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
    if s == "infinitive-markers" and u["section_index"] <= 2: add("purposive")
    if s == "infinitive-examples": add("verb", "inf")
    if "negative" in s:
        add("neg")
        if "markers" in s or "marker" in s: add("suffix")
        elif "examples" in s: add("verb")
        if s == "past-negative-examples": add("pret")
        if "nonpast-negative" in s: add("non-past")
    if s in {"modal-ku", "potential-aappa", "prohibitive-marker", "attempted-marker",
             "negative-potential", "definitive-auxiliaries", "completive-marker",
             "durative-auxiliaries"}:
        add("auxiliary")
        if f.startswith(("-", "–")): add("suffix")
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
    if s == "past-negative-markers":
        tags = []
        if f in {"ldid", "-il"}: add("suffix", "neg", "pret")
        elif f == "-ani": add("suffix", "conditional")
        elif f == "-di": add("suffix", "participle")
    if s == "durative-negative-markers" and f == "-a":
        tags = ["suffix", "relative", "participle"]
    if s == "doubt-negative-marker" or s == "doubt-negative-examples": add("dubitative")
    if s == "modal-ku" or s == "modal-ku-examples": add("modal", "possibility")
    if s == "nonpast-negative-examples": add("conditional")
    if s == "relative-participle-examples" and u["section_index"] == 2: add("neg")
    if s == "past-negative-examples":
        add("participle", "relative" if u["section_index"] <= 2 else "conjunctive-participle")
    if s == "verbal-participle-examples" and u["section_index"] <= 2: add("neg")
    if s == "durative-auxiliaries":
        add(*(["pret", "fut"] if u["section_index"] == 1 else ["pres"]))
    # Source-scoped finite-person labels, never a scan of ordinary prose.
    finite = {
        "p15:past-examples:01": "1sg", "p15:past-examples:02": "1sg",
        "p15:past-examples:03": "1sg", "p15:past-examples:04": "3pl",
        "p15:past-examples:05": "1sg", "p15:past-examples:06": "1sg",
        "p15:nonpast-examples:01": "1sg 1pl", "p15:nonpast-examples:02": "2pl",
        "p15:nonpast-examples:03": "3sg", "p15:nonpast-examples:04": "3sg",
        "p15:nonpast-examples:05": "2sg",
        "p16:doubt-negative-examples:01": "3pl",
        "p16:general-negative-examples:01": "3sg pret", "p16:general-negative-examples:02": "2pl pret",
        "p17:attempted-examples:01": "1sg pret", "p17:attempted-examples:02": "1sg fut",
        "p17:past-definitive-examples:01": "1sg pret", "p17:past-definitive-examples:02": "3sg pret",
        "p17:nonpast-definitive-example:01": "1sg 1pl non-past fut",
        "p18:durative-examples:01": "1sg pret", "p18:durative-examples:02": "3sg pres",
        "p18:durative-examples:03": "1sg 1pl fut",
        "p18:reflexive-examples:01": "3sg pret", "p18:reflexive-examples:02": "3sg pres",
        "p18:negative-person-examples:01": "1sg pret", "p18:negative-person-examples:02": "3sg pret",
    }
    add(*finite.get(u["unit_key"].removeprefix("nair2013bettakurumba:"), "").split())
    return " ".join(tags)


def gloss_for(u):
    """Lexical tier only; every exact printed definition remains in the audit."""
    s, raw = u["section"], u["gloss"]
    marker_sections = {
        "diagnostic-case-markers", "diagnostic-plural", "diagnostic-link", "diagnostic-participles",
        "plural-markers", "accusative-markers", "instrumental-marker", "dative-markers", "ablative-markers",
        "locative-markers", "causative-case-marker", "purposive-marker", "interrogative-bases",
        "indefinite-marker", "past-markers", "nonpast-markers", "transitive-markers", "infinitive-markers",
        "durative-negative-markers", "past-negative-markers", "nonpast-negative-marker", "doubt-negative-marker",
        "general-negative-marker", "relative-participle-markers", "verbal-participle-markers", "modal-ku",
        "potential-aappa", "negative-potential", "prohibitive-marker", "attempted-marker", "reflexive-base",
        "person-suffix-table",
    }
    if s in marker_sections:
        return ""
    if s == "durative-auxiliaries":
        return "to be"
    lexical = re.sub(r"\s*\((?:Acc|Dat)\.\)", "", raw)
    if s in {"past-examples", "nonpast-examples", "general-negative-examples", "durative-examples",
             "negative-person-examples"}:
        lexical = re.sub(r"-?(?:I/we|I|you\s*\(Pl\.\)|you\(sg\.\)|he/she/it|he/she|they)$", "", lexical).strip()
    if s == "personal-pronouns" and "oblique base" in lexical:
        lexical = "I"
    return lexical


def note_for(u):
    bits = []
    if u["surface"]:
        bits.append("Printed segmented analysis: " + u["form"] + ".")
    s = u["section"]
    if s in {"accusative-markers", "dative-markers", "ablative-markers", "locative-markers",
             "instrumental-marker", "locative-pili-examples", "causative-case-marker",
             "purposive-marker", "past-markers", "nonpast-markers",
             "transitive-markers", "diagnostic-participles", "indefinite-marker",
             "durative-negative-markers",
             "negative-potential", "prohibitive-marker", "verbal-participle-markers",
             "modal-ku", "potential-aappa", "attempted-marker", "definitive-auxiliaries",
             "completive-marker", "durative-auxiliaries", "reflexive-base"}:
        if u["source_claim"]:
            bits.append(u["source_claim"])
    if s == "diagnostic-link":
        bits.append("Nair reports Jayapal's link-morph diagnostic.")
    if s == "nonpast-examples" and u["section_index"] == 5:
        bits.append("The source's surface form retains parenthesized yi.")
    if s == "infinitive-markers":
        bits.append("Source assigns purposive meaning." if u["section_index"] <= 2 else
                    "Source assigns effective and simultaneous meanings.")
    if s == "past-negative-markers":
        if u["form"] == "ldid":
            bits.append("Source uses ldid before relative-participle -a or conditional -ani.")
        elif u["form"] == "-il":
            bits.append("Source uses -il before verbal-participle -di.")
    if s == "relative-participle-markers":
        bits.append("Source describes -a as common in Dravidian and –u as comparatively rare.")
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
    emissions = []
    if emitted:
        raw_form = u["surface"] or u["form"]
        assert raw_form
        alts = raw_form.split("/") if u["section"] == "person-suffix-table" and "/" in raw_form else [raw_form]
        for i, form in enumerate(alts, 1):
            key = u["unit_key"] if len(alts) == 1 else f'{u["unit_key"]}:variant:{i}'
            parent = f'{u["unit_key"]}:variant:1' if i > 1 else ""
            gloss = gloss_for(u)
            row = ["BettaKurumba", "", form, gloss, "", "", note_for(u),
                   f'nair2013bettakurumba[p. {u["printed_page"]}, {u["section"].replace("-", " ")}]',
                   "", "", key, parent, "", "", tags_for(u)]
            assert len(row) == len(COLS)
            rows.append(row)
            made.append(key)
            emissions.append({"entry_key": key, "form": row[2], "lexical_gloss": row[3],
                              "notes": row[6], "source": row[7], "tags": row[14],
                              "variant_of_key": row[11]})
    audit.append({**u, "disposition": "proposed-source-attestation" if made else "audit-only",
                  "proposed_entry_keys": made,
                  "proposed_emissions": emissions,
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
