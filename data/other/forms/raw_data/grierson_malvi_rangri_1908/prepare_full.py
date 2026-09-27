"""Prepare reviewed Rangri grammar/table/specimens; never installs canonical files.

The final source freeze additionally requires both whole specimens and native-witness
accounting. Until then this command creates explicitly partial source-stage proposals.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import re
import json
import unicodedata
from collections import Counter

P = Path(__file__).resolve().parent
DATA = P.parents[4]
SOURCE = "grierson1908malvirangri"
DIALECT = "dialect:Malw:lsi1908-malvi-rangri:Rangri"
TARGET_SCOPES = {"shared-rangri-malvi", "explicit-rangri"}


def read_jsonl(name):
    return [json.loads(line) for line in (P / name).read_text().splitlines() if line]


def nfc(value):
    return unicodedata.normalize("NFC", value)


def table_tags(n):
 if n<=13:return ['num']
 if n<=31:
  person=('1sg' if n<=16 else '1pl' if n<=19 else '2sg' if n<=22 else '2pl' if n<=25 else '3sg' if n<=28 else '3pl')
  return ['pron','personal',person]+(['gen','poss'] if n not in [14,17,20,23,26,29] else [])
 if n<=76:return ['noun']
 if n<=85:return ['verb','impv']
 if n<=91:return ['adv','spatial']
 if n in [92,93]:return ['pron']
 if n==94:return ['adv','interr']
 if n<=97:return ['conj']
 if n in [98,99,100]:return ['interj']+(['neg'] if n==99 else [])
 if n<=118:
  q=n-101 if n<=109 else n-110
  return ['noun','pl' if q>=4 else 'sg']+({1:['gen'],2:['dat'],3:['abl'],6:['gen'],7:['dat'],8:['abl']}.get(q,[]))
 if n<=127:
  q=n-119
  return ['noun','adj','pl' if q>=4 else 'sg']+({1:['gen'],2:['dat'],3:['abl'],6:['gen'],7:['dat'],8:['abl']}.get(q,[]))
 if n<=131:return ['noun','adj']+(['pl'] if n==130 else ['sg'])
 if n<=137:return ['adj']+(['degree'] if n in [133,134,136,137] else [])
 if n<=155:return ['noun','pl' if n in [140,141,144,145,148,149,152,155] else 'sg']
 if n<=167:return ['verb','copula','pres' if n<=161 else 'pret',['1sg','2sg','3sg','1pl','2pl','3pl'][(n-156)%6]]
 if n==168:return ['verb','copula','impv']
 if n==169:return ['verb','copula','inf']
 if n==170:return ['verb','copula','participle']
 if n==171:return ['verb','copula','conjunctive-participle']
 if n in [172,173,174]:return ['verb','copula','1sg']+{172:['subjunctive'],173:['fut'],174:['conditional']}[n]
 if n<=178:return ['verb']+{175:['impv'],176:['inf'],177:['participle'],178:['conjunctive-participle']}[n]
 if n<=190:return ['verb','pres' if n<=184 else 'pret',['1sg','2sg','3sg','1pl','2pl','3pl'][(n-179)%6]]
 if n<=194:return ['verb','1sg']+{191:['pres','progressive'],192:['pret','progressive'],193:['pret','perfect'],194:['subjunctive']}[n]
 if n<=200:return ['verb','fut',['1sg','2sg','3sg','1pl','2pl','3pl'][n-195]]
 if n==201:return ['verb','1sg','conditional']
 if n<=204:return ['verb','1sg','pass',{202:'pres',203:'pret',204:'fut'}[n]]
 if n<=216:return ['verb','pres' if n<=210 else 'pret',['1sg','2sg','3sg','1pl','2pl','3pl'][(n-205)%6]]
 if n==217:return ['verb','impv']
 if n in [218,219]:return ['verb','participle']+(['pret'] if n==219 else [])
 return ['sentential']


def expand_table(unit):
    """Expand only locally printed ellipsis; retain every literal cell in audit."""
    n = unit["prompt_number"]
    raw = unit["provisional_visual_reading"]
    forms, rules = [], []
    for group in raw.split(";"):
        previous = None
        for part in group.split(","):
            form = part.strip()
            if not form:
                continue
            if form.startswith("-"):
                assert previous and "-" in previous, (n, form, previous)
                form = previous.rsplit("-", 1)[0] + form
                rules.append("Explicit abbreviated ending expanded against preceding full form within the same semicolon group.")
            elif n == 109 and form == "sē":
                form = "Bāpā̃-sē"
                rules.append("Bare sē in parallel ablative ending list expanded with Bāpā̃; printed missing hyphen retained in raw cell.")
            elif forms and n in {157, 158, 161, 173, 182, 206, 207, 210}:
                form = raw.split(" ", 1)[0] + " " + form
                rules.append("Shared initial subject repeated for coordinated alternative predicates.")
            forms.append(nfc(form))
            previous = form
    return forms, list(dict.fromkeys(rules))


def grammar_gloss(unit):
    # These parentheticals describe the already structured paradigm, not a sense.
    gloss = unit["gloss"]
    for label in [" (past participle)", " (present participle)", " (conjunctive participle)", " (agent)", " (oblique)", " (genitive)", " (feminine)", " (plural)", " (plural oblique)", " (oblique singular)", " (present)", " (including addressee)"]:
        gloss = gloss.removesuffix(label)
    return gloss


def grammar_notes(unit):
    section = unit["section"]
    if section.startswith("noun-"):
        return "Source says plural nasalization is commonly omitted." if "pl" in unit["tags"] else ""
    if section.startswith("pronoun-"):
        return "Source says nasals in this paradigm are frequently omitted."
    if section.startswith("simple-pres-"):
        return "Source gives this paradigm simple-present, present-conjunctive and future uses."
    if section.startswith("aux-pres-"):
        return "Source states that the third-person plural is not nasalized." if section.endswith("3pl") else ""
    if section.startswith("aux-past-"):
        return "Source says this tense does not change for person."
    if section in {"irregular-do-past", "irregular-take-past", "irregular-give-past"}:
        return "Source also compares the dh alternative with Gujarati."
    if section == "irregular-go-past":
        return ""
    if section in {"what", "no-one", "strike-agency", "conj-example-got", "conj-example-gone"}:
        return ""
    if section in {"children", "dog"}:
        return "Source illustrates its diminutive or contemptuous suffix class with this example; it does not select one of those functions here."
    if section == "reflexive-example":
        return "Source illustrates possessive use of the reflexive."
    if section == "ordinary-genitive-example":
        return "Source illustrates an ordinary possessive replacing the reflexive."
    if section.startswith("past-"):
        return "Source describes past-participle constructions of transitive verbs as passive." if "struck" in section else ""
    return unit.get("note", "")


def reuse_exact_attestations(rows, audit, legacy_keys):
    """Merge only complete-analysis equality, retaining all occurrences in audit."""
    groups = {}
    for row in rows:
        fingerprint = tuple(" ".join(sorted(value.split())) if i == 14 else value
                            for i, value in enumerate(row) if i not in {7, 10})
        groups.setdefault(fingerprint, []).append(row)
    representatives, alias_map = [], {}
    for group in groups.values():
        legacy = [r for r in group if r[10] in legacy_keys]
        # Multiple durable pilot IDs never collapse, even for identical analysis.
        if len(legacy) > 1:
            for row in group:
                representatives.append(row)
                alias_map[row[10]] = row[10]
            continue
        representative = (legacy or group)[0].copy()
        citations = []
        for row in group:
            alias_map[row[10]] = representative[10]
            for citation in row[7].split("; "):
                if citation not in citations:
                    citations.append(citation)
        representative[7] = "; ".join(citations)
        representatives.append(representative)
    for unit in audit:
        original = unit["entry_keys"][:]
        unit["pre_reuse_entry_keys"] = original
        unit["entry_keys"] = list(dict.fromkeys(alias_map.get(k, k) for k in original))
        changes = {k: alias_map[k] for k in original if alias_map.get(k, k) != k}
        if changes:
            unit["exact_attestation_reuse"] = changes
            unit["reuse_basis"] = "Identical language, literal form, gloss, native, phonemic, residual Notes, graph fields and complete grammatical/dialect tag set; all citations merged."
            if unit["status"] == "ingested" and len(changes) == len(original):
                unit["status"] = "reused_exact_attestation"
    assert {r[10] for r in representatives} >= legacy_keys
    return representatives


def generate():
    rows, audit, expanded = [], [], []

    def emit(unit, forms, gloss, tags, notes, key, locator, native="", extra_citations=None):
        keys = []
        for index, form in enumerate(forms, 1):
            child = key if index == 1 else key + f":alternate:{index}"
            row_tags = list(tags)
            if " " in form:
                row_tags.append("multiword-expression")
            row_tags.append(DIALECT)
            rows.append(["Malw", "", nfc(form), nfc(gloss), nfc(native), "", nfc(notes),
                         "; ".join([f"{SOURCE}[{locator}]"] + (extra_citations or [])), "", "", child, "", "", "",
                         " ".join(dict.fromkeys(row_tags))])
            keys.append(child)
        return keys

    grammar = read_jsonl("shared-grammar-first-reading.jsonl")
    assert len(grammar) == 280
    for unit in grammar:
        assert unit["review"] == "visual-second-reading-sharper-original"
        key = unit["source_unit_key"].replace("grierson1908rangri:", SOURCE + ":", 1)
        locator = f"p. {unit['printed_page']}, grammatical sketch, {unit['section']}"
        bound_morphology = bool({"suffix", "prefix"} & set(unit["tags"]))
        target = unit["scope"] in TARGET_SCOPES and not bound_morphology
        notes = " ".join(x for x in [grammar_notes(unit), unit.get("source_commentary", "")] if x)
        if unit["scope"] == "shared-rangri-malvi":
            notes = "Source applies this shared grammatical example to Rangri and Malvi proper under its explicit standing rule on p.54. " + notes
        keys = []
        analyses = []
        if target:
            section = unit["section"]
            tags = list(unit["tags"])
            if section in {"past-struck", "past-have-struck", "past-had-struck"}:
                tags.append("pass")
            if section == "impersonal":
                tags.append("impersonal")
            if section in {"then-when", "there-where"}:
                tags += ["relative", "demonstrative"]
            if section.startswith("simple-pres-"):
                for function, gloss in [("pres", "strike"), ("subjunctive", "may strike"), ("fut", "shall strike")]:
                    sense_tags = [t for t in tags if t != "pres"] + [function]
                    sense_key = key if function == "pres" else key + ":function:" + function
                    keys += emit(unit, unit["forms"], gloss, sense_tags, notes.strip(), sense_key, locator)
                    analyses.append({"sense_key": sense_key, "function": function, "gloss": gloss})
            else:
                keys = emit(unit, unit["forms"], grammar_gloss(unit), tags, notes.strip(), key, locator)
        audit.append({**unit, "function_analyses": analyses, "entry_keys": keys, "status": "ingested" if target else "bound_morphology_evidence" if bound_morphology and unit["scope"] in TARGET_SCOPES else "excluded_control",
                      "citation_locator": locator, "reason": "Explicit shared/Rangri source scope" if target else "Isolated source grammatical affix; retained as bound morphology evidence, not standalone lexical entry" if bound_morphology else unit["scope"]})

    table = read_jsonl("full-table-reconciled-reading.jsonl")
    assert [u["prompt_number"] for u in table] == list(range(1, 242))
    for unit in table:
        assert unit["status"] in {"full_second_reading", "source_blank"}
        n, page = unit["prompt_number"], unit["printed_page"]
        forms, rules = expand_table(unit)
        key = f"{SOURCE}:p{page}:rangri:item:{n}"
        locator = f"p. {page}, Mālvī (Rāngrī) column, standard-list item {n}"
        tags = table_tags(n)
        notes = ""
        if n in {92, 93}:
            notes = "The English prompt does not specify interrogative versus relative use."
        if n == 133:
            notes = "Source parenthetically glosses Waṇī-sū̃ as ‘than that’."
        if n in {134, 137}:
            notes = "Source prompt identifies the superlative construction."
        keys = emit(unit, forms, unit["english_prompt"], tags, notes, key, locator)
        audit.append({**unit, "raw_source_cell": unit["provisional_visual_reading"],
                      "expanded_forms": forms, "expansion_rules": rules, "entry_keys": keys,
                      "status": "ingested" if forms else "source_blank", "citation_locator": locator,
                      "grammar_basis": "English survey prompt; no subtype invented for who/what; degree follows better/best/higher/highest.",
                      "control_columns_excluded": ["Malvi when different from Rangri", "Nimadi"]})
        expanded.append({"source_unit_key": unit["source_unit_key"], "prompt_number": n,
                         "printed_page": page, "raw_source_cell": unit["provisional_visual_reading"],
                         "forms": forms, "english_prompt": unit["english_prompt"], "expansion_rules": rules,
                         "status": "reviewed_literal_expansion" if forms else "source_blank"})
    if (P / "full-stage-inputs.json").exists():
        configuration = json.loads((P / "full-stage-inputs.json").read_text())
        settings = configuration["specimens"]
        raw = (P / settings["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == settings["sha256"], "Frozen specimen input changed"
        specimens = [json.loads(line) for line in raw.decode().splitlines()]
        assert len(specimens) == settings["physical_units"]
        spec = importlib.util.spec_from_file_location("rangri_specimen_grammar", P / "specimen_grammar.py")
        grammar_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grammar_module)
        specimen_keys = {u["source_unit_key"] for u in specimens}
        specimens_by_key = {u["source_unit_key"]: u for u in specimens}
        native_map = {}
        native_holds = []
        if "native" in configuration:
            native_settings = configuration["native"]
            native_bytes = (P / native_settings["path"]).read_bytes()
            assert hashlib.sha256(native_bytes).hexdigest() == native_settings["sha256"], "Frozen native input changed"
            native_records = [json.loads(line) for line in native_bytes.decode().splitlines()]
            assert len(native_records) == native_settings["alignment_records"]
            for record in native_records:
                if record["alignment_status"] == "aligned":
                    assert record["source_unit_key"] not in native_map
                    native_map[record["source_unit_key"]] = record
                else:
                    native_holds.append(record)
            assert set(native_map) == specimen_keys
        for unit in specimens:
            assert unit["review"] == "complete-six-page-grayscale-rereading-20260926"
            key = unit["source_unit_key"]
            locator = f"p. {unit['printed_page']}, {unit['section']}, Roman interlinear line {unit['line']}, cell {unit['word']}"
            grouped = unit.get("emission_group_keys", [])
            if grouped:
                assert set(grouped) <= specimen_keys
                for other_key in grouped[1:]:
                    other = specimens_by_key[other_key]
                    locator += f", continued p. {other['printed_page']}, line {other['line']}, cell {other['word']}"
            continuation = re.search(r"end of line (\d+) continues.*beginning of line (\d+)", unit.get("note", ""))
            if continuation:
                locator += f", continued line {continuation[2]}, cell 1"
            tags, observations = grammar_module.classify(unit)
            notes = "Source locality: " + unit["source_locality"] + ". Aligned specimen expression; translation is local to this occurrence."
            residual = unit.get("note", "")
            if "source footnote" in residual:
                notes += " " + residual
            elif grouped:
                notes += " The source interlinear glosses for the two adjacent words are transposed; the complete expression preserves their combined meaning."
            elif "title Sarᵃdār-kā" in residual:
                notes += " This name is followed by the title Sarᵃdār-kā on the next page."
            elif residual.startswith("Title following Paḍiyār"):
                notes += " This title follows the name Paḍiyār on the preceding page."
            native_form = ""
            extra_citations = []
            native_evidence = []
            if native_map:
                for atomic_key in grouped or [key]:
                    alignment = native_map[atomic_key]
                    assert alignment["roman_forms_reviewed"] == specimens_by_key[atomic_key]["forms"]
                    native_evidence.append(alignment)
                    extra_citations.append(f"{SOURCE}[p. {alignment['printed_native_page']}, native witness, {alignment['native_locator'].replace(';', ' and ')}]")
                    if alignment.get("native_uncertainty"):
                        tags.append("uncertain")
                        notes += " Native transcription uncertainty: " + alignment["native_uncertainty"]
                native_form = " ".join(x["native_form"] for x in native_evidence)
            if unit.get("reuse_entry_key"):
                assert unit["reuse_entry_key"] in {row[10] for row in rows}
                keys = [unit["reuse_entry_key"]]
                status = "reused_grouped_expression"
            else:
                keys = emit(unit, unit.get("emission_forms", unit["forms"]), unit.get("emission_gloss", unit["gloss"]), tags, notes, key, locator, native_form, extra_citations)
                status = "ingested"
            audit.append({**unit, "entry_keys": keys, "status": status,
                          "citation_locator": locator, "grammatical_interpretation_notes": observations,
                          "grammar_basis": "Explicit local English inflection/function labels; ambiguous by/to case readings remain unassigned.",
                          "native_alignment": native_evidence})
        for hold in native_holds:
            audit.append({**hold, "source_unit_key": hold["native_evidence_key"],
                          "related_grammar_unit_key": hold["source_unit_key"], "entry_keys": [],
                          "status": "native_alignment_hold"})
    assert len({row[10] for row in rows}) == len(rows)
    old = list(csv.reader((DATA / "data/other/forms/20260925-grierson-malvi-rangri.csv").open()))
    legacy_keys = {row[10] for row in old}
    assert legacy_keys <= {row[10] for row in rows}
    if (P / "full-stage-inputs.json").exists() and "native" in json.loads((P / "full-stage-inputs.json").read_text()):
        rows = reuse_exact_attestations(rows, audit, legacy_keys)
    return rows, audit, expanded


def main():
    rows, audit, expanded = generate()
    with (P / "proposal.csv").open("w", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    for name, records in [("proposal-audit.jsonl", audit), ("full-table-expanded.jsonl", expanded)]:
        (P / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
    clusters = set()
    for row in rows:
        current = []
        for char in row[2]:
            if unicodedata.combining(char) and current:
                current[-1] += char
            else:
                current.append(char)
        clusters.update(current)
    (P / "proposal-profile.txt").write_text("Grapheme\tIPA\n" + "".join(c + "\t" + ("#" if c == " " else c.lower().replace("ṅ", "ŋ")) + "\n" for c in sorted(clusters)))
    progress = {"status": "confirmed_partial_not_installable", "rows": len(rows), "audit_units": len(audit),
                "statuses": dict(Counter(x["status"] for x in audit)),
                "pending": ["whole-source independent audit", "metadata/profile and focused checks"],
                "pre_reuse_rows": sum(len(u.get("pre_reuse_entry_keys", u["entry_keys"])) for u in audit if u["status"] != "reused_grouped_expression"),
                "roman_editorial_aligned_units": 823, "roman_physical_fragments": 827,
                "native_witness_lines": 65, "native_aligned_atoms": 823, "unpaired_native_alternatives": 2,
                "canonical_pilot_unchanged": True}
    (P / "full-stage-progress-20260926.json").write_text(json.dumps(progress, indent=2) + "\n")
    print(json.dumps(progress))


if __name__ == "__main__":
    main()
