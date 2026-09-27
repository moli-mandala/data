"""Assemble the complete reviewed Koda source readings; never install or build a DB."""

import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = "grierson1906lsi4"
FILES = {
    "grammar": "grammar-second-reading-20260926.jsonl",
    "birbhum": "birbhum-specimen-third-reading-20260926.jsonl",
    "bankura": "bankura-specimen-third-reading-20260926.jsonl",
    "prose": "dhangar-bankura-prose-second-reading-20260926.jsonl",
    "table": "dhangar-table-final-reading-20260926.jsonl",
}
LECTS = {
    "Birbhum": "dialect:Koda:koda_birbhum_lsi1906:Birbhum",
    "Dhangar": "dialect:Koda:koda_dhangar_lsi1906:Dhangar",
    "Bankura": "dialect:Koda:koda_bankura_lsi1906:Bankura",
}
BANKURA_NOTE = (
    "The source describes this Bankura witness as corrupt and says its beginning "
    "was editorially restored, without consistently restoring semi-consonants. "
    "Printed parentheses and segmentation are retained."
)


def read_jsonl(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines()]


def nfc(text):
    return unicodedata.normalize("NFC", text)


def build():
    inputs = {section: read_jsonl(name) for section, name in FILES.items()}
    table_metadata = {
        item['prompt_number']: item
        for item in json.loads((ROOT / 'table-grammatical-metadata-staged.json').read_text())['items']
    }
    prose_metadata = json.loads((ROOT / 'prose-grammatical-metadata-staged.json').read_text())['items']
    assert {k: len(v) for k, v in inputs.items()} == {
        "grammar": 41, "birbhum": 333, "bankura": 93, "prose": 27, "table": 241
    }
    legacy = json.loads((ROOT / "whole-source-assembly-plan-20260926.json").read_text())["legacy_key_mapping"]
    wraps = json.loads((ROOT / "specimen-line-wrap-review-20260926.json").read_text())["items"]
    starts = {item["physical_cells"][0]: item["physical_cells"][1] for item in wraps}
    tails = set(starts.values())
    specimen = {r["source_cell_key"]: r for r in inputs["birbhum"] + inputs["bankura"]}
    rows, audit, identities, by_key = [], [], {}, {}

    def emit(key, lect, form, gloss, citation, notes="", tags=""):
        form, gloss = nfc(form), nfc(gloss)
        assert form and gloss and lect in LECTS
        identity = (lect, form, gloss)
        if identity in identities:
            existing = identities[identity]
            if citation not in existing[7].split(";"):
                existing[7] += ";" + citation
            if notes and notes not in existing[6]:
                existing[6] = (existing[6] + " " + notes).strip()
            existing[14] = ' '.join(dict.fromkeys(existing[14].split() + tags.split()))
            return existing[10], True
        row = ["Koda", "", form, gloss, "", "", notes, citation, "", "", key, "", "", "", (LECTS[lect] + " " + tags).strip()]
        rows.append(row)
        identities[identity] = row
        by_key[key] = row
        return key, False

    for section, records in inputs.items():
        for raw in records:
            key = raw.get("source_cell_key", raw.get("source_key", raw.get("entry_key")))
            page = raw["printed_page"]
            lect = raw.get("lect", "Birbhum" if section == "grammar" else "Dhangar")
            record = {"source_cell_key": key, "section": section, "printed_page": page, "lect": lect, "source_reading": raw, "entry_keys": [], "reuse_entry_keys": []}
            audit.append(record)
            status = raw["status"]
            if status in {"bound_morphology_inventory", "bound-morphology-inventory"}:
                record.update(status="inventory_bound_morphology", reason="Explicit grammatical suffix or categorical element; no independent lexical meaning invented.")
                continue
            if status == "excluded-comparison-control":
                record.update(status="inventory_other_language_control", reason="Explicit Mundari comparison, not a Bankura lexical attestation.")
                continue
            if status == "source_reuse_pending_alignment":
                target = "grierson1906koda:grammar:p109:possessive02"
                assert target in by_key
                record.update(status="inventory_repeated_fragment", reuse_entry_keys=[target], reason="The source repeats this fragment of the preceding phrase to discuss grammar, without a separate lexical gloss.")
                continue
            if key in tails:
                head = next(k for k, v in starts.items() if v == key)
                head_record = next(r for r in audit if r["source_cell_key"] == head)
                record.update(status="physical_continuation", reuse_entry_keys=head_record["entry_keys"] + head_record["reuse_entry_keys"], reason="Explicit terminal-hyphen continuation emitted with its preceding physical cell.")
                continue
            form = raw.get("form_candidate", raw.get("form"))
            gloss = raw.get("printed_gloss", raw.get("gloss"))
            notes = BANKURA_NOTE if lect == "Bankura" else ""
            tags = ' '.join(prose_metadata.get(key, []))
            if raw.get("uncertainty"):
                notes = (notes + " Source transcription uncertainty: " + raw["uncertainty"]).strip()
                tags += " uncertain"
            if section in {"birbhum", "bankura"}:
                locator = f"{lect} specimen, line {raw['line']}, word {raw['word']}"
            elif section == "table":
                locator = f"Dhangar standard list, item {raw['prompt_number']}"
            else:
                locator = key.split(":", 2)[-1]
            citation = f"{SOURCE}[p. {page}, {locator}]"
            if key in starts:
                tail = specimen[starts[key]]
                assert form.endswith("-")
                form += tail["form_candidate"]
                gloss = gloss.rstrip("-") + "-" + tail["printed_gloss"]
                citation += f";{SOURCE}[p. {tail['printed_page']}, {lect} specimen, line {tail['line']}, word {tail['word']}]"
                notes = (notes + " Joined explicit physical line continuation; both original locators retained.").strip()
            alternatives = [form]
            glosses = [gloss]
            if section == "table":
                number = raw["prompt_number"]
                metadata = table_metadata[number]
                assert metadata['printed_gloss'] == gloss
                tags += ' ' + ' '.join(metadata['tags'])
                if 156 <= number <= 219:
                    notes = (notes + ' Grammatical tags describe the printed English paradigm; they do not assert an independently analysed target-language morphological segmentation.').strip()
                if number in {133, 136, 134, 137}:
                    degree = 'comparative' if number in {133, 136} else 'superlative'
                    tags += ' degree'
                    notes = (notes + f' The printed English prompt is {degree}; the degree tag describes that source contrast, without asserting target morphology.').strip()
                if number in {103, 104}:
                    notes = (notes + " Complete printed table answer retained as one alternative pattern: semicolon-separated abbreviated relational endings have no independently stated lexical gloss. No missing phrase text supplied.").strip()
                    tags += " multiword-expression"
                else:
                    alternatives = [x.strip() for x in form.split(";")]
                    glosses = [gloss] * len(alternatives)
                if number == 1:
                    glosses = ["one", "one", "one only"]
                elif number == 47:
                    glosses = ["father", "father", "my father", "thy father", "his father"]
                elif number == 49:
                    glosses = ["brother", "brother", "elder brother"]
                elif number == 50:
                    glosses = ["elder sister", "my younger sister"]
                if any(x.startswith("-") for x in alternatives):
                    notes = (notes + " Leading hyphen is printed in the table answer; no omitted stem is supplied.").strip()
                if number >= 220:
                    tags += " multiword-expression"
            if section == "grammar" and raw.get("note") and key.endswith(("number06", "number07", "number08", "number09", "number10")):
                reading_note = 'Read as ãṭ after the second reading and independent alternate-witness check; tilde and underdot retained.' if key.endswith('number08') else raw['note']
                notes = (notes + " " + reading_note).strip()
                notes += ' Grierson explicitly classifies numerals six and following as Aryan loan-words (p. 110); attribution retained without inventing a donor lemma or borrowing edge.'
                tags += ' num loanword'
            if section == 'grammar' and ':number' in key:
                tags += ' num'
            if section == 'bankura' and '(sic)' in raw.get('note', ''):
                notes += ' The author prints (sic) immediately after this form; the source warning is retained without treating it as lexical letters.'
            if section == "prose" and key.endswith("bankura:10"):
                notes += " This expanded spelling is supplied explicitly by the author."
            assert len(alternatives) == len(glosses)
            for i, (alternative, meaning) in enumerate(zip(alternatives, glosses)):
                proposed_key = legacy.get(key, key) if i == 0 else f"{key}:answer{i + 1}"
                target, reused = emit(proposed_key, lect, alternative, meaning, citation, notes.strip(), ' '.join(dict.fromkeys(tags.split())))
                record["reuse_entry_keys" if reused else "entry_keys"].append(target)
            record["status"] = "ingested" if record["entry_keys"] else "exact_reuse"
    assert len(audit) == 735
    assert set(legacy.values()) <= {r[10] for r in rows}
    assert len({r[10] for r in rows}) == len(rows)
    assert all(len(r) == 15 for r in rows)
    assert all(not r[2].endswith("-") for r in rows)
    keys = {r[10] for r in rows}
    assert all(set(r["entry_keys"] + r["reuse_entry_keys"]) <= keys for r in audit)
    return rows, audit


def main():
    rows, audit = build()
    output = ROOT / "full-preview.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    with (ROOT / "full-preview-audit.jsonl").open("w", encoding="utf-8") as f:
        for record in audit:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "status": "whole_source_proposal_pending_independent_audit_not_installed",
        "forms": len(rows), "physical_units": len(audit),
        "statuses": dict(Counter(r["status"] for r in audit)),
        "forms_by_lect": dict(Counter(r[14].split()[0] for r in rows)),
        "legacy_keys_preserved": 5,
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256((ROOT / "full-preview-audit.jsonl").read_bytes()).hexdigest(),
        "remaining": "Independent original-page audit, complete preservation profile, lect/source metadata, focused parser checks, then source-stage installation. DB and full-build/browser gates deferred per user.",
    }
    (ROOT / "full-preview-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
