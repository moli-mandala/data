"""Reproducible source-local proposal for the complete Adiya chapter.

This stage emits every attributed Adiya/Malayalam/Kannada witness independently.
It makes no canonical writes and does not decide variant or etymology edges.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / "dialects.py").exists())
sys.path.insert(0, str(DATA))
from dialects import dialect_tag  # noqa: E402

CSV_FIELDS = (
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
)
LECT = dialect_tag("Ravula", "nair2013-adiya", "Adiya (Wayanad)")


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tags_for(section, witness, index):
    form = witness["form"]
    label = witness.get("note", "")
    tags = []
    if section.endswith("table") and section not in {"pronoun-paradigm"}:
        if section == "plural-table":
            tags.append("pl")
        if section == "l-to-v-table" and witness["gloss"] == "seven":
            tags.append("num")
    elif section == "plural-morphology":
        tags += (["suffix", "pl"] if form.startswith("-") else ["pron", "pl"])
        if form in {"ava", "iva"}:
            tags.append("demonstrative")
    elif section == "pronoun-paradigm":
        tags.append("pron")
        if "nominative" in label:
            tags.append("nom")
        if "oblique" in label:
            tags.append("obl")
        if "stem" in label:
            tags.append("stem")
        if label.startswith("I "):
            tags += ["1pl" if "plural" in label else "1sg"]
            if "exclusive" in label:
                tags.append("exclusive")
            if "inclusive" in label:
                tags.append("inclusive")
        elif label.startswith("II "):
            tags += ["2pl" if "plural" in label else "2sg"]
        elif label.startswith("III "):
            tags += ["3pl" if "plural" in label else "3sg"]
            if "feminine" in label and "non-feminine" not in label:
                tags.append("f")
            if "proximal" in label or "distal" in label:
                tags.append("demonstrative")
            if "proximal" in label:
                tags.append("prox")
            if "distal" in label:
                tags.append("dist")
        elif "honorific" in label:
            tags += ["honorific", "sg" if "singular" in label else "pl"]
    elif section in {"first-person-variation", "first-person-oblique"}:
        tags += ["1sg", "sentential"] if " " in form else ["pron", "1sg"]
        if section == "first-person-oblique" and " " not in form:
            tags += ["obl", "stem"]
    elif section == "second-person-distribution":
        if form == "okka":
            tags.append("postp")
        else:
            tags += ["2pl" if form in {"niiŋka", "niinŋka"} else "2sg", "pron"]
            if form in {"niiŋka", "niinŋka"}:
                tags.append("honorific")
    elif section == "all-of-you-expression":
        tags += ["2pl", "sentential"]
    elif section == "kin-address-pronouns":
        tags += ["pron", "2pl" if form == "niiŋka" else "2sg"]
        if form == "niiŋka":
            tags.append("honorific")
    elif section == "third-person-gender-examples":
        if witness["language_label"] == "Adiya":
            tags += ["3sg", "sentential"] if " " in form else ["pron", "3sg"]
        if form in {"ooḷu", "avooḷu"}:
            tags += ["f", "demonstrative"]
    elif section in {"elder-kin-demonstratives", "younger-kin-demonstratives"}:
        tags += ["pron", "demonstrative", "honorific"] if " " not in form else ["sentential", "kinship"]
    elif section == "three-tense-examples":
        tags += ["sentential", "1sg", ("pret", "pres", "fut")[index - 1]]
    elif section == "personal-termination-examples":
        tags += ["sentential", ("1sg", "1pl", "2sg", "2pl", "3sg", "3pl")[index - 1]]
        tags += ["fut" if index == 2 else "pret"] if index <= 6 else []
    elif section in {"addressee-sex-markers", "affinal-address-mi", "affinal-address-i"}:
        tags += ["suffix"] if form.startswith("-") else ["sentential"]
        if section == "addressee-sex-markers" and not form.startswith("-"):
            tags.append("impv")
    elif section == "negative-kaani":
        tags += ["negator", "neg"] if form.startswith("-") else ["sentential", "neg"]
    elif section == "negative-a":
        tags += ["suffix", "neg"] if form.startswith("-") else ["sentential", "neg"]
    elif section == "archaic-ula":
        tags += ["verb"] if form == "uḷa" else ["sentential"]
    elif section == "kinship-terms":
        tags.append("kinship")
    if witness["role"] != "target":
        tags = [x for x in tags if x not in {"sentential", "1sg", "1pl", "2sg", "2pl", "3sg", "3pl"}]
    if witness["role"] == "target":
        tags.append(LECT)
    return " ".join(dict.fromkeys(tags))


def main():
    first = HERE / "first-reading.jsonl"
    units = [json.loads(line) for line in first.read_text().splitlines()]
    rows, audit = [], []
    for unit in units:
        section = unit["unit_key"].split(":", 2)[2]
        keys = []
        evidence = []
        for index, witness in enumerate(unit["witnesses"], 1):
            physical_key = f'{unit["unit_key"]}:slot{index}'
            emitted = witness["role"] in {"target", "comparison control", "parenthetical comparison control"}
            key = physical_key if emitted else None
            evidence.append({"physical_key": physical_key, "witness": witness,
                             "disposition": "emitted" if emitted else "documentary_only",
                             "entry_key": key})
            if not emitted:
                continue
            keys.append(key)
            label = witness["language_label"]
            lang = {"Adiya": "Ravula", "Malayalam": "Malayalam", "Kannada": "Kannada"}[label]
            note = witness.get("note", "")
            if label != "Adiya":
                note = (f"Nair attributes this comparative form to {label}. " + note).strip()
            if section == "vowel-a-to-e-table" and index == 6:
                note += " Source visibly spaces ka and ṭeelu; whether one word is typographically split remains uncertain."
            if section == "v-to-b-table" and " " in witness["form"]:
                note += " Source visibly spaces the ṭ segment; a typesetting split versus a word boundary is uncertain."
            if section == "rounded-u-table" and index == 3:
                note += " Source spelling includes both breve and final u; retain literal typography."
            if section == "kinship-terms" and witness["gloss"]:
                note = (note + "; source kinship scope: " + witness["gloss"]).strip("; ")
            row = [lang, "", witness["form"], witness["gloss"], "", "", note,
                   f'nair2013adiya[p. {unit["printed_page"]}, {section}, slot {index}]',
                   "", "", key, "", "", "", tags_for(section, witness, index)]
            assert len(row) == len(CSV_FIELDS)
            rows.append(row)
        audit.append({"source_unit_key": unit["unit_key"], "printed_page": unit["printed_page"],
                      "pdf_page": unit["pdf_page"], "source_claim": unit["source_claim"],
                      "note": unit["note"], "physical_witnesses": evidence,
                      "entry_keys": keys, "held_or_excluded_count": len(evidence)-len(keys),
                      "review": "owner provisional; independent full review and output audit pending"})
    assert len(units) == 29 and sum(len(x["physical_witnesses"]) for x in audit) == 174
    assert len(rows) == 171 and len({r[10] for r in rows}) == 171
    assert sum(a["held_or_excluded_count"] for a in audit) == 3
    out = HERE / "proposed.csv"
    with out.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    audit_path = HERE / "proposal-audit.jsonl"
    audit_path.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audit))
    (HERE / "proposal-status.json").write_text(json.dumps({
        "status": "complete first candidate; literal, tag, reference, graph and independent output reviews pending",
        "source_pdf_sha256": hash_file(HERE.parents[5] / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"),
        "first_reading_sha256": hash_file(first), "csv_sha256": hash_file(out),
        "audit_sha256": hash_file(audit_path), "physical_units": len(units),
        "physical_witnesses": 174, "emitted_rows": len(rows), "documentary_only": 3,
        "source_lect_tag": LECT, "canonical_changes": False,
    }, ensure_ascii=False, indent=2) + "\n")
    print(len(rows), "candidate rows,", len(audit), "audited blocks")


if __name__ == "__main__":
    main()
