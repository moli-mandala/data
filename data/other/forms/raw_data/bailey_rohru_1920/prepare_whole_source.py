"""Stage the entire Rohru source; never write canonical inputs or build CLDF."""
from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = "bailey1920rohru"
INPUTS = ("whole-grammar-first-reading.jsonl", "whole-sentences-first-reading.jsonl", "whole-commentary-first-reading.jsonl")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def literal(text):
    # Both combining low lines represent the same printed continuous underline.
    return unicodedata.normalize("NFC", text.replace("\u0331", "\u0332"))


def prepare():
    review = json.loads((HERE / "whole-second-reading-20260927.json").read_text())
    for name, digest in review["input_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    legacy = list(csv.reader((HERE / "legacy-before-whole-recovery.csv").open()))
    assert len(legacy) == 309
    rows = [row[:] for row in legacy]
    audit = read_jsonl(HERE / "legacy-before-whole-recovery-audit.jsonl")
    fixes = {r["source_unit_key"]: r for r in review["corrections"]}
    assert len(fixes) == len(review["corrections"])
    recovery = {r["source_unit_key"]: r for r in review["glossary_hold_recovery"]}
    for unit in audit:
        key = unit["source_cell_key"]
        if key not in recovery:
            continue
        rec = recovery[key]
        unit["historical_hold"] = {k: unit[k] for k in ("status", "reason", "uncertainty", "printed_form_review")}
        unit.update(status="ingested_uncertain", uncertainty=rec["typed_uncertainty"],
                    recovery=rec, entry_keys=[])
        for i, form in enumerate(rec["forms"], 1):
            ek = key if i == 1 else f"{key}:answer{i}"
            rows.append(["roh", "", literal(form), rec["gloss"], "", "", rec["typed_uncertainty"],
                         f"{SOURCE}[{unit['citation_locator']}]", "", "", ek,
                         key if i > 1 else "", "", "", "uncertain"])
            unit["entry_keys"].append(ek)
    aliases = {"post": ["postp"], "dem": ["demonstrative"], "rel": ["relative"],
               "dat-acc": ["dat", "acc"], "ordinal": ["ord"], "cardinal": [],
               "comp": ["degree"], "superl": ["degree"], "prog": ["progressive"],
               "act": [], "habitual": [], "agent": []}
    records = [r for name in INPUTS for r in read_jsonl(HERE / name)]
    assert len(records) == 403
    for raw in records:
        r = json.loads(json.dumps(raw))
        key = r["source_unit_key"]
        fix = fixes.get(key)
        uncertainty = []
        if fix:
            assert r["forms"] == fix["old_forms"], key
            r["forms"] = fix["forms"]
            r["second_reading_correction"] = fix
            r["raw_source_form"] = fix.get("raw_source_form", r.get("raw_source_form"))
            uncertainty.extend(fix.get("typed_uncertainty", []))
        section = r.get("section", "sentences")
        tags = list(r["tags"])
        notes = []
        # Explicit source commentary, not transcriber process descriptions.
        if "noun-" in section:
            if section == "noun-hindu":
                notes.append("The source does not separately specify number for these three Hindu forms.")
            if str(raw.get("raw_source_form", "")).startswith("-"):
                notes.append(f"Printed abbreviated cell: {r['raw_source_form']}.")
        if section == "pronoun-dem":
            notes.append("The source gives the plural forms as common to both genders.")
        if section == "pronoun-prose" and key.endswith(("interr-gen", "interr-dat", "interr-abl")):
            r["printed_pattern"] = "kauh- ro khĕ khu"
        if section == "pronoun-prose" and key.endswith(("rel-gen", "rel-dat", "rel-abl")):
            r["printed_pattern"] = "dzau-ro khĕ khu"
        if section == "adjective-morphology":
            notes.append("Adjectives used as nouns decline as nouns; otherwise consonant-final adjectives do not decline. Adjectives in -u or -o take the stated endings. Genitives used adjectivally take -e rather than -ā in masculine oblique singular and masculine plural.")
        if section == "comparison":
            notes.append("Bailey states that comparative and superlative have no special forms: the positive combines with khu 'from'.")
            if key.endswith("comparative"):
                notes.append("Literal source gloss: good from this; better than this.")
            elif key.endswith("superlative"):
                notes.append("Literal source gloss: good from all; better than all; best.")
        if "correlative" in tags:
            notes.append("Source labels this correlative.")
        if section == "adverb-comment":
            notes.append("Bailey says these adjective pronouns are often used as adverbs.")
        if key.endswith("p122:prepositions:8"):
            tags.append("instr")
            notes.append("The source explicitly labels the 'with' sense instrumental.")
        if key.endswith(("p122:prepositions:12", "p122:prepositions:13")):
            uncertainty.append("transcription: breve-a in băṭṭi is provisional; the worn italic vowel could be read as breve-u. Two readers lean a from the closed bowl.")
        if section == "be" or section == "be-past":
            tags.append("copula")
            if section == "be":
                notes.append("Source present-tense alternatives are unchanged throughout.")
        if section == "beat-pres-ind":
            notes.append("The source says present conditional has the same forms as present indicative.")
        if section == "beat-imperfect":
            tags.append("ipfv")
            notes.append("Source imperfect: present indicative with the past substantive verb. The feminine auxiliary is printed after a shared verbal form.")
            r["printed_abbreviated_auxiliary"] = "thĕ" if "sg" in tags else "thī"
        if section in {"beat-past", "beat-pluperfect"}:
            notes.append("The source explicitly describes these past forms as agreeing with the object.")
            if section == "beat-pluperfect":
                notes.append("Source labels this pluperfect.")
        if section == "beat-past-conditional-present-participle":
            notes.append("Shared source heading: past conditional and present participle; both labels are retained without asserting simultaneous tense values.")
        if section == "beat-participles":
            if ":act-" in key:
                notes.append("Source present active participle; printed head pīṭdo with endings (-ĕ, -ē, -ī).")
                r["printed_pattern"] = "pīṭdo (-ĕ, -ē, -ī)"
            if ":pass-" in key:
                notes.append("Source passive participle, described as contracted from pīṭo ondo; the source compares Hindi mārā huā.")
            if key.endswith(":habitual"):
                notes.append("Source labels this habitual.")
            if key.endswith(":agent"):
                notes.append("Source labels this agent.")
            if key.endswith(":immediate"):
                notes.append("Source labels this immediate present.")
        if section == "verb-become" and ":present-" in key:
            n = int(key.rsplit("-", 1)[1])
            r["raw_source_form"] = ["ōh-ū", "-ā", "-ā", "-ī", "-ā", "-ā"][n - 1]
            r["printed_pattern"] = "ōh-ū, -ā, -ā, -ī, -ā, -ā"
            notes.append(f"Printed abbreviated form: {r['raw_source_form']}.")
        if key.endswith("p124:verb-give:past"):
            notes.append("Source labels this stative participle.")
        if section == "ability":
            notes.append("Ability uses bŏḷṇo with ī added to the other verb root. In negative sentences the source uses past conditional for present indicative.")
            if key.endswith(":ending"):
                tags = [t for t in tags if t != "inf"] + ["verb", "modal"]
        if section == "intro-morphology":
            notes.append(raw.get("note", "").replace("Source gives short-mark i for feminine noun inflection.", "Source gives -ī for feminine noun inflection.").replace("otherwise end in ero", "otherwise end in eṛo"))
        if section == "intro-cases":
            notes.append("The Rohru introduction explicitly identifies this case function.")
        if section == "intro-verbs":
            notes.append(raw.get("note", ""))
        if section == "Sentence notes":
            notes.append(raw.get("note", "").replace("biāhno", "biāhṇo").replace("Complete independent example; preserve raw transcription and source grammatical account.", "Ability is rendered with bŏḷṇo and an infinitive formed by adding -ī to the root."))
        if key.endswith("p125:sentence:09"):
            uncertainty.append("transcription: final i of Mūī̃ has two stacked marks; upper mark is provisionally encoded as tilde, without asserting nasal phonetic value.")
        if key.endswith("p125:sentence:10"):
            notes.append("The source English literally prints 'cows got grazing is'.")
        if key.endswith("p126:sentence:22"):
            notes.append("Complete translated response ending in lāo 'was-taken'.")
        for u in uncertainty:
            notes.append("Source uncertainty: " + u)
        if uncertainty:
            tags.append("uncertain")
        normalized = []
        for tag in tags:
            normalized.extend(aliases.get(tag, [tag]))
        if any(" " in f for f in r["forms"]):
            normalized.append("multiword-expression")
        r.update(status="ingested_uncertain" if uncertainty else "ingested",
                 typed_uncertainty=uncertainty, exported_entry_keys=[],
                 exported_tags=list(dict.fromkeys(normalized)), exported_notes=" ".join(n for n in notes if n))
        r.pop("transcription_status", None)
        r.pop("uncertainty", None)
        for i, form in enumerate(r["forms"], 1):
            ek = key if i == 1 else f"{key}:answer{i}"
            locator = r.get("source_locator", f"p. {r['printed_page']}, {section}, {key.rsplit(':', 1)[1]}")
            rows.append(["roh", "", literal(form), r["gloss"], "", "", r["exported_notes"],
                         f"{SOURCE}[{locator}]", "", "", ek, key if i > 1 else "", "", "", " ".join(r["exported_tags"])])
            r["exported_entry_keys"].append(ek)
        audit.append(r)
    for r in read_jsonl(HERE / "whole-source-context-census.jsonl"):
        r["exported_entry_keys"] = []
        r["complete_page_context"] = f"whole-source-text-inventory.jsonl:printed_page={r['printed_page']}"
        if r["source_unit_key"].endswith(":vowels"):
            r["source_claims"] = "Bailey says the long-marked vowels differ from unmarked vowels only in length, and unaccented lengths may be omitted phonetically although Oriental-language publications conventionally retain them. Breve-a is compared with English but; breve-e is approximately short e, probably lower; breve-o with hot; breve-u with pull rather than root, without usual lip protrusion or drawing together. These are attributed descriptions, not newly assigned IPA."
        if r["source_unit_key"].endswith(":tone-general"):
            r["source_claims"] = "Bailey says sonant aspiration is practically absent in Koci/Simla except probably through Hindi influence; postvocalic h is generally unsounded and the syllable has high falling tone. He contrasts northern/western Amritsar Panjabi devoicing and low tone. In discussing a proposed Romany–Shina connection he notes that northern Panjabi and Simla dialects except Bilaspur also avoid aspirated sonants. These are source claims, not asserted genetic or borrowing relations."
        if r["status"] == "source_scope_unresolved_context":
            r["status"] = "shared_lect_context"
            r["scope_note"] += " These unglossed examples are explicitly general Koci/Simla, not individually identified as Rohru; source evidence remains here rather than acquiring invented Rohru-only attribution."
        if r["source_unit_key"].endswith("hindi-panjabi"):
            r["forms"] = ["āyā huā lāgā huā", "aundā e lāgā"]
        audit.append(r)
    # Repeated phonetic examples on p121 have explicit citations and mappings.
    by_key = {r[10]: r for r in rows}
    for num, form, target in [(1, "tsŏūo", "p121:ordinals:6"), (2, "tshau", "p120:cardinals:6")]:
        key = SOURCE + ":" + target
        row = by_key[key]
        row[7] += f";{SOURCE}[p. 121, sixth/six aspiration comparison, item {num}]"
        row[6] += " Bailey says the h of sixth is weaker than in six, although the printed sixth form tsŏūo has no h."
        audit.append(dict(source_unit_key=f"{SOURCE}:p121:aspiration-comparison:{num}", printed_page=121,
                          forms=[form], status="reused_exact_attestation", exported_entry_keys=[key],
                          scope_note="Exact repeated target form; source phonetic claim retained without spelling repair."))
    assert rows[:309] == legacy
    assert len(rows) == 723
    assert len({row[10] for row in rows}) == len(rows)
    return rows, audit


if __name__ == "__main__":
    rows, audit = prepare()
    with (HERE / "whole-proposed.csv").open("w", newline="") as out:
        csv.writer(out).writerows(rows)
    (HERE / "whole-proposed-audit.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audit))
    print(json.dumps({"rows": len(rows), "audit_units": len(audit), "legacy_rows_preserved": 309}))
