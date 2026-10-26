"""Prepare a reviewable whole-source Juang proposal; never install or build CLDF."""
import csv
import hashlib
import importlib.util
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "whole-source-recovery"
SOURCE = "dalton1872haldarjuang"
spec = importlib.util.spec_from_file_location("historical_juang", ROOT / "import_source.py")
historical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(historical)

RECOVERED = {
    "p235:row:04": ["gandá-ni"],
    "p236:row:01": ["ijing"], "p236:row:02": ["muttái"],
    "p236:row:07": ["jhuntá"], "p236:row:16": ["boy-i"],
    "p236:row:19": ["juáng", "mánush", "kamaholo"],
    "p236:row:21": ["kutonge"], "p236:row:22": ["wériki"],
    "p236:row:23": ["kano", "kan-da"], "p236:row:32": ["kujindá"],
    "p236:row:35": ["in-yá"], "p237:row:14": ["hassen-yá"],
    "p237:row:16": ["onúku"], "p237:row:17": ["kinomitá"],
    "p237:row:20": ["mándu"], "p241:numerals:01": ["moiñ"],
    "p242:left-column:31": ["kánáh", "bagi"],
    "p242:right-column:08": ["diunu"], "p242:right-column:30": ["hindrá"],
}
SOURCE_QUESTION = {"p236:row:21", "p236:row:22", "p237:row:14", "p237:row:16", "p237:row:17", "p237:row:20"}
TYPOGRAPHY = {
    "p235:row:04": "Post-hyphen n is the preferred original reading; cramped m/n typography remains uncertain. Separate-list ganda mi is not substituted.",
    "p241:numerals:01": "Final n bears a short curved/horizontal upper mark, provisionally preserved as ñ; precise mark shape and phonetic interpretation remain uncertain.",
    "p242:right-column:08": "Middle letters are cramped; two original-image readings support diunu, retained with typography uncertainty.",
}


def tags_for(unit):
    """Only source headings, explicit labels, and contrastive paradigm prompts."""
    key = unit["entry_key"].removeprefix(SOURCE + ":")
    page, row = unit["printed_page"], unit["row"]
    gloss = unit["english_prompt"].lower()
    tags = [historical.DIALECT]
    if (page == 235 and row <= 13) or ":numerals:" in key:
        tags += ["num"]
    if page == 235 and 14 <= row <= 34:
        tags += ["pron"]
        person = {14:"1sg",15:"1sg",18:"1pl",20:"2sg",21:"2sg",24:"2pl",27:"3sg"}.get(row)
        if person:
            tags.append(person)
        if row in (15,21):
            tags += ["gen", "poss"]
    if ":pronouns:" in key:
        tags += ["pron", {1:"1sg",2:"1sg",3:"1pl",4:"2sg",5:"2pl",6:"3sg",7:"3pl"}[row]]
        if row == 2:
            tags += ["gen", "poss"]
    if page == 237 and row in (3,4,5,6,7,8,9,10,11):
        tags += ["verb"]
    if page == 238 and row in (34,36):
        tags += ["noun", "sg" if row == 34 else "pl"]
        if row == 34:
            tags += ["m"]
    if page == 239 and row == 3:
        tags += ["verb", "copula", "pres", "1sg"]
    if page == 240 and row in (4,7):
        tags += ["verb", "1sg", "pres" if row == 4 else "pret"]
    if ", v." in gloss:
        tags += ["verb"]
    if key in ("p242:left-column:46", "p242:left-column:47"):
        tags += ["noun", "sg" if row == 46 else "pl"]
    if key in ("p242:right-column:47", "p242:right-column:48"):
        tags += ["noun", "m" if row == 47 else "f"]
    if gloss in ("who", "what", "why"):
        tags += ["interr", "adv" if gloss == "why" else "pron"]
    return list(dict.fromkeys(tags))


def build():
    old_rows, old_units = historical.build()
    assert len(old_rows) == 196 and len(old_units) == 361
    old_keys = {r[10] for r in old_rows}
    correction_map = {}
    owner = json.loads((OUT / "owner-table-second-reading-20260927.json").read_text())
    for item in owner["retained_cell_corrections_pending_independent_review"]:
        correction_map[item["source_unit_key"]] = item
    separate = json.loads((OUT / "owner-separate-list-corrections-20260927.json").read_text())
    for item in separate["corrections"]:
        correction_map[f"{SOURCE}:{item['inventory_key']}"] = item
    rows, audit = [], []
    for original in old_units:
        unit = dict(original)
        key = unit["entry_key"]
        short = key.removeprefix(SOURCE + ":")
        unit["historical_decision"] = original
        unit["source_unit_key"] = key
        forms = list(original["selected_forms"])
        if short in RECOVERED:
            assert original["status"] == "held"
            forms = RECOVERED[short]
        if key in correction_map:
            fix = correction_map[key]
            assert forms == [fix["old_form"]], (key, forms, fix)
            forms = [fix["form"]]
            unit["literal_correction"] = fix
        unit["selected_forms"] = forms
        unit["status"] = "reviewed-target" if forms else "printed-blank"
        unit["entry_keys"] = []
        tags = tags_for(unit)
        base_notes = []
        if short in SOURCE_QUESTION:
            tags += ["uncertain"]
            base_notes += ["The source prints a question mark for this response."]
        if short in TYPOGRAPHY:
            tags += ["uncertain"]
            base_notes += [TYPOGRAPHY[short]]
        if "separate Juang list" in unit.get("section", ""):
            locator = f"p. {unit['printed_page']}, {unit['section']}, row {unit['row']}"
        else:
            locator = f"p. {unit['printed_page']}, Haldar table, Juanga column, row {unit['row']}"
        for n, form in enumerate(forms, 1):
            row_key = f"{key}:v{n}" if len(forms) > 1 else key
            notes = list(base_notes)
            if "(s.)" in original["juanga_printed"] and (short != "p236:row:19" or n == 2):
                notes.append("Source annotation: (s.); no donor-language interpretation or borrowing edge is inferred.")
            if len(forms) > 1:
                notes.append(f"Printed alternative {n}/{len(forms)} for the same prompt.")
            grouped_cases = short in ("p241:pronouns:04", "p241:pronouns:06")
            if grouped_cases:
                notes.append("Source groups nominative and possessive/case glosses without pairing each response; no variant or morphological relation is inferred between these forms.")
            if short in ("p242:left-column:46", "p242:left-column:47"):
                notes.append("Source explicitly says the plural is formed by adding loko; plain a in these examples differs from the earlier ká head.")
            if short in ("p242:right-column:47", "p242:right-column:48"):
                notes.append("Source explicitly says sex is shown by adding andesh and petah respectively.")
            gloss = unit["english_prompt"].lower().replace(", v.", "")
            row = ["ju", "", form, gloss, "", "", " ".join(notes), f"{SOURCE}[{locator}]", "", "", row_key,
                   f"{key}:v1" if n > 1 and not grouped_cases else "", "", "", " ".join(dict.fromkeys(tags))]
            rows.append(row)
            unit["entry_keys"].append(row_key)
        audit.append(unit)
    for line in (OUT / "chapter-prose-first-reading.jsonl").read_text().splitlines():
        unit = json.loads(line)
        unit["historical_review"] = unit["review"]
        unit["historical_status"] = unit["status"]
        if unit["status"] == "documentary-attribution-review":
            unit["status"] = "documentary-control"
        unit["review"] = "Owner and root original-context review completed; final whole-output sample separate."
        unit["independent_review"] = "root-prose-attribution-review-20260927.json"
        unit["entry_keys"] = []
        if unit["status"].startswith("target"):
            key = unit["source_unit_key"]
            rows.append(["ju", "", unit["form"], unit["gloss"], "", "", unit["note"],
                         f"{SOURCE}[p. {unit['printed_page']}, Juang chapter prose, {key.rsplit(':',1)[-1]}]",
                         "", "", key, "", "", "", " ".join(unit["tags"])])
            unit["entry_keys"] = [key]
        audit.append(unit)
    for item in separate["additional_explicit_morphology"]:
        key = f"{SOURCE}:p242:morphology:{item['form']}"
        tag = "pl" if item["form"] == "loko" else "m" if item["form"] == "andesh" else "f"
        rows.append(["ju", "", item["form"], item["gloss"], "", "", item["note"],
                     f"{SOURCE}[p. 242, explicit grammatical statement, {item['form']}]", "", "", key, "", "", "",
                     f"{historical.DIALECT} {tag}"])
        audit.append(dict(item, source_unit_key=key, entry_keys=[key], status="explicit-morphology", printed_page=242, pdf_page=257))
    keys = {r[10] for r in rows}
    assert len(keys) == len(rows) and old_keys <= keys
    # Preserve historical physical order as well as immutable keys. Newly recovered
    # attestations append, so legacy stem/append-order compatibility is not shifted.
    by_key = {r[10]: r for r in rows}
    rows = [by_key[r[10]] for r in old_rows] + [r for r in rows if r[10] not in old_keys]
    assert all(not r[11] or r[11] in keys for r in rows)
    assert all(len(r) == 15 and all(unicodedata.normalize("NFC", v) == v for v in r) for r in rows)
    return rows, audit, old_keys


def main():
    rows, audit, old_keys = build()
    with (OUT / "proposal.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with (OUT / "proposal-audit.jsonl").open("w") as handle:
        for unit in audit:
            handle.write(json.dumps(unit, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"status":"whole-source proposal; not independently approved or installed", "rows":len(rows), "audit_units":len(audit),
               "preserved_legacy_keys":len(old_keys), "legacy_row_order_preserved":True, "recovered_historical_held_cells":len(RECOVERED),
               "printed_blank_cells":sum(u["status"] == "printed-blank" for u in audit),
               "literal_correction_cells":sum("literal_correction" in u for u in audit),
               "uncertain_rows":sum("uncertain" in r[14].split() for r in rows),
               "variant_links":sum(bool(r[11]) for r in rows),
               "hashes":{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ("proposal.csv","proposal-audit.jsonl")}}
    (OUT / "proposal-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
