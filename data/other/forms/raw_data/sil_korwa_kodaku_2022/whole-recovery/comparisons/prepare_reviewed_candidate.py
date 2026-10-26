"""Owner-reviewed comparison witnesses; no canonical writer or identity inference."""
import copy
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

B = Path(__file__).resolve().parent


def read(name):
    return [json.loads(x) for x in (B / name).read_text().splitlines()]


def write(name, rows):
    (B / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def tags(item):
    if 151 <= item <= 164:
        return ["num"]
    if 165 <= item <= 170:
        return ["interr"]
    if 171 <= item <= 174:
        return ["demonstrative"] + (["pl"] if item >= 173 else [])
    if 182 <= item <= 201:
        return ["verb"]
    if 202 <= item <= 210:
        person = 1 if item in (202, 207, 208) else 2 if item in (203, 204, 209) else 3
        number = "sg" if item <= 206 else "pl"
        out = ["pron", f"{person}{number}", number]
        out += {203: ["informal"], 204: ["formal"], 205: ["m"], 206: ["f"],
                207: ["inclusive"], 208: ["exclusive"]}.get(item, [])
        return out
    return []


def main():
    data_root = next(p for p in B.parents if (p / "tags.py").exists())
    baseline = json.loads((B.parent / "legacy-review/reconciliation.json").read_text())
    canonical = data_root / "data/other/forms/20260828-sil-korwa-kodaku.csv"
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == baseline["pins"]["data/other/forms/20260828-sil-korwa-kodaku.csv"]
    registered_tags = set()
    for node in ast.parse((data_root / "tags.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"GENDER_TAGS", "GRAMMATICAL_TAGS"} for t in node.targets):
            registered_tags.update(ast.literal_eval(node.value))
    source = read("comparison-first-inventory.jsonl")
    physical = read("comparison-physical-occurrences.jsonl")
    progress = json.loads((B / "original-review-progress.json").read_text())
    assert sorted(progress["completed_original_pages"]) == list(range(66, 91))
    candidates, accounting = [], []
    moved = {"behera2022korwakodaku:p80:wordlist-line79:siteB",
             "behera2022korwakodaku:p80:wordlist-line80:siteV",
             "behera2022korwakodaku:p80:wordlist-line81:sitek"}
    for original in source:
        r = copy.deepcopy(original)
        r["reviewed_against_original"] = True
        r["status"] = "owner-reviewed proposal; independent acceptance pending"
        r["typed_uncertainties"] = []
        r["source_prompt"] = r["gloss"]
        r["tags"] = tags(r["item"])
        r["phonemic"] = ""
        r["transcription_policy"] = "Printed phonetic notation preserved; no additional IPA inference."
        emitted = []
        if not r["form"]:
            r["disposition"] = "historical blank superseded by physical response" if (r["item"], r["source_code"]) == (208, "V") else "source blank"
            accounting.append(r)
            continue
        groups = [(False, [k for k in r["physical_keys"] if k not in moved]),
                  (True, [k for k in r["physical_keys"] if k in moved])]
        for relocated, keys in groups:
            if not keys:
                continue
            n = copy.deepcopy(r)
            n["physical_keys"] = keys
            if relocated:
                n["source_unit_key"] += ":physical208"
                n["item"] = 208
                n["gloss"] = n["source_prompt"] = "we (1st pl, exclusive)"
                n["tags"] = tags(208) + ["uncertain"]
                n["typed_uncertainties"] = [{"type": "source-layout-attribution", "note": "Printed first-column cells lie below item208 before its second-column continuation. Historical text-order parser assigned210. Duplicate similarity-group numbering makes the source layout itself unusual; retain this physical attribution for independent adjudication, without semantic inference."}]
            if n["item"] == 71 and n["source_code"] == "U" and "(" in n["form"]:
                assert n["form"] == "khud̪i (broken rice)"
                n["form"], n["gloss"] = "khud̪i", "broken rice"
                n["field_decision"] = "English parenthetical is source gloss qualification, not a target form segment."
            if n["item"] == 195 and n["source_code"] in ("B", "V"):
                assert n["form"] == "tʃɐl   ["
                n["form"] = "tʃɐl"
                n["field_decision"] = "Opening bracket belongs to source code list [BV], not target form."
            alternatives = [n["form"]]
            if n["item"] in (200, 201) and n["source_code"] in ("B", "V"):
                alternatives = [f.strip() for f in n["form"].split(",")]
                assert len(alternatives) == 2
                n["field_decision"] = "Two comma-separated target responses; no unsupported response-specific imperative/past assignment."
            for i, form in enumerate(alternatives, 1):
                a = copy.deepcopy(n)
                a["form"] = form
                if len(alternatives) > 1:
                    a["source_unit_key"] += f":v{i}"
                if " " in form:
                    a["tags"].append("multiword-expression")
                a["historical_source_unit_key"] = original["source_unit_key"]
                a["disposition"] = "candidate, language routing pending independent identity review"
                candidates.append(a)
                emitted.append(a["source_unit_key"])
        r["candidate_keys"] = emitted
        r["disposition"] = "represented by owner-reviewed candidate"
        accounting.append(r)
    assert len(source) == len(accounting) == 1628
    assert len({x["source_unit_key"] for x in candidates}) == len(candidates)
    assert set(k for r in candidates for k in r["physical_keys"]) == {r["physical_key"] for r in physical}
    assert all(r["language_id"] is None for r in candidates)
    assert all(set(r["tags"]) <= registered_tags for r in candidates)
    assert sum(r["source_code"] == "J" for r in candidates) == 209
    assert sum(r["source_code"] == "J" and r["disposition"] == "source blank" for r in accounting) == 2
    write("comparison-reviewed-candidate.jsonl", candidates)
    write("comparison-review-accounting.jsonl", accounting)
    summary = {
        "status": "complete owner original review; independent source/output and identity approval pending",
        "historical_response_units": len(source), "candidate_rows": len(candidates),
        "candidate_counts_by_source_code": dict(Counter(r["source_code"] for r in candidates)),
        "physical_occurrences": len(physical),
        "confirmed_blank_units": sum(r["disposition"] == "source blank" for r in accounting),
        "historical_blank_superseded": 1,
        "coverage_pdf_pages": list(range(66, 91)),
        "baseline_installed_keys": 4458,
        "canonical_changes": 0,
        "identity_scope": "All seven labels retained verbatim; Tanmai explicitly unresolved. No new base assignments.",
        "grammar_scope": "Numerals, interrogative/demonstrative prompts, explicit person/number/gender/formality/inclusion structured. Combined imperative/past prompts remain whole source glosses without guessing each response's TAM. Other English lexical prompts do not automatically impose POS.",
        "original_review_findings": ["item71U parenthetical gloss", "item195B/V code-list bracket", "item200/201B/V comma alternatives", "PDF90 lines79–81 physical208 attribution, with scope uncertainty"],
        "fine_glyph_controls": ["PDF67 item17 k ɳõh retained: visible retroflex hook", "PDF69 item33 AJ kuʈasːi retained: colon immediately after s", "PDF86 item185 A nuʔd̪e retained: dental mark beneath d"],
        "inputs": {n: hashlib.sha256((B/n).read_bytes()).hexdigest() for n in ["comparison-first-inventory.jsonl", "comparison-physical-occurrences.jsonl", "original-review-progress.json", "prepare_reviewed_candidate.py"]},
        "outputs": {n: hashlib.sha256((B/n).read_bytes()).hexdigest() for n in ["comparison-reviewed-candidate.jsonl", "comparison-review-accounting.jsonl"]},
    }
    (B / "reviewed-candidate-status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["candidate_rows", "physical_occurrences", "confirmed_blank_units", "candidate_counts_by_source_code"]}))


if __name__ == "__main__":
    main()
