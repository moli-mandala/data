"""Key-preserving source-local legacy repair proposal; never installs."""
import ast
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

B = Path(__file__).resolve().parent
D = next(p for p in B.parents if (p / "tags.py").exists())
P = D / "data/other/forms/20260828-sil-korwa-kodaku.csv"
PRON = {
    202: ("I", ["pron", "1sg", "sg"]),
    203: ("you", ["pron", "2sg", "sg", "informal"]),
    204: ("you", ["pron", "2sg", "sg", "formal"]),
    205: ("he", ["pron", "3sg", "sg", "m"]),
    206: ("she", ["pron", "3sg", "sg", "f"]),
    207: ("we", ["pron", "1pl", "pl", "inclusive"]),
    208: ("we", ["pron", "1pl", "pl", "exclusive"]),
    209: ("you", ["pron", "2pl", "pl"]),
    210: ("they", ["pron", "3pl", "pl"]),
}


def main():
    baseline = json.loads((B / "reconciliation.json").read_text())
    assert hashlib.sha256(P.read_bytes()).hexdigest() == baseline["pins"]["data/other/forms/20260828-sil-korwa-kodaku.csv"]
    rows = list(csv.reader(P.open()))
    registered = set()
    for node in ast.parse((D / "tags.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"GENDER_TAGS", "GRAMMATICAL_TAGS"} for t in node.targets):
            registered.update(ast.literal_eval(node.value))
    output, audit = [], []
    fieldnames = ["Language", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes", "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags"]
    for old in rows:
        row = old.copy()
        item = int(re.search(r":i(\d+):", row[10])[1])
        decisions, extra = [], []
        if item in PRON:
            row[3], extra = PRON[item]
            decisions.append("Move the explicit source prompt's person/number/gender/register/inclusion analysis to structured tags; keep lexical pronoun gloss.")
        elif 151 <= item <= 164:
            extra = ["num"]
            decisions.append("Explicit numbered numeral elicitation sequence.")
        elif 165 <= item <= 170:
            extra = ["interr"]
            decisions.append("Printed interrogative elicitation prompts; no inferred target case or morphological decomposition.")
        elif 171 <= item <= 174:
            extra = ["demonstrative"] + (["pl"] if item >= 173 else [])
            decisions.append("Printed demonstrative contrasts; these/those explicitly plural, no unprinted case classification.")
        # Mixed imperative/past prompts do not establish which analysis each
        # response instantiates. Keep the whole prompt and avoid guessed TAM.
        row[14] = " ".join(dict.fromkeys([*row[14].split(), *extra]))
        prefix = "manually verified against typeset source image; "
        assert row[6].startswith(prefix)
        row[6] = row[6][len(prefix):]
        decisions.append("Move procedural review status to audit, preserving the non-etymological similarity-group information in Notes.")
        if item == 11:
            row[6] += "; Behera p. 49 says prompt 11 ‘breast’ was replaced with ‘chest’; the printed p. 56 wordlist heading nevertheless reads ‘breast’."
            decisions.append("Preserve contradictory printed heading; do not silently replace breast with chest.")
        if item == 99 and row[2] == "gɛɳɖel (small)":
            row[2] = row[5] = "gɛɳɖel"
            row[3] = "small ant"
            row[6] += "; Source response is qualified ‘small’ under the prompt ‘ant’."
            decisions.append("Separate explicit English parenthetical qualification from source phonetic response (printed65).")
        if item == 107 and row[2] == "wagʌt̪ (eldest)":
            row[2] = row[5] = "wagʌt̪"
            row[3] = "eldest brother"
            row[6] += "; Source response is qualified ‘eldest’ under the prompt ‘older brother’."
            decisions.append("Separate explicit English parenthetical qualification from source phonetic response (printed66).")
        emitted = [row]
        if row[10] == "silkorwakodaku2022:i183:K:f2":
            assert row[2] == "hʊwa,hʊʔ ʌkʌrɪja"
            row[2] = row[5] = "hʊwa"
            second = row.copy()
            second[2] = second[5] = "hʊʔ ʌkʌrɪja"
            second[10] = old[10] + ":v2"
            second[14] += " multiword-expression"
            for part in (row, second):
                part[6] += "; Two comma-separated responses share the printed elicitation prompt; their individual imperative/past analyses are not specified."
            emitted.append(second)
            decisions.append("Split the two printed comma responses without guessing a morphological relation; original key retained for first, new stable child key for second (printed76).")
        output.extend(emitted)
        changes = {fieldnames[i]: {"old": old[i], "new": row[i]} for i in range(15) if old[i] != row[i]}
        audit.append({"entry_key": old[10], "item": item, "printed_prompt": old[3], "raw_source_cell": old[2], "baseline_row": old,
                      "emitted_keys": [r[10] for r in emitted], "changes": changes, "decisions": decisions,
                      "source_citation": old[7], "original_evidence": "Existing immutable wordlist-200dpi original pages66–90; explicit pronouns89–90, numerals82–83, interrogatives83–84, demonstratives84; qualifications75–76 and comma86."})
    assert len(rows) == 4458 and len(output) == 4459
    assert len({r[10] for r in output}) == len(output)
    assert {r[10] for r in rows} < {r[10] for r in output}
    assert sum("(small)" in r[2] or "(eldest)" in r[2] for r in rows) == 8
    for r in output:
        assert len(r) == 15
        assert all(t.startswith("dialect:") or t in registered for t in r[14].split())
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(output)
    (B / "legacy-repaired-candidate.csv").write_text(stream.getvalue())
    (B / "legacy-metadata-repair.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in audit))
    status = {"status": "source-backed proposal; independent approval pending", "baseline_sha256": hashlib.sha256(P.read_bytes()).hexdigest(),
              "baseline_rows": 4458, "proposed_rows": 4459, "all_existing_keys_preserved": True, "new_keys": ["silkorwakodaku2022:i183:K:f2:v2"],
              "changed_field_counts": dict(Counter(k for r in audit for k in r["changes"])),
              "pronoun_rows": sum(r["item"] in PRON for r in audit), "parenthetical_rows_repaired": 8,
              "source_scope": "All4458 baseline rows examined for prompt/field classes; not a fresh literal rereading of all18 target lists.",
              "deferred_or_withheld": ["No noun/adjective/case morphology inferred solely from English lexical gloss.", "No individual imperative/past mapping imposed on combined prompts182–201.", "No etymology inferred from similarity groups.", "Full row/output independent audit and canonical installation remain pending."],
              "canonical_writes": 0,
              "outputs": {n: hashlib.sha256((B/n).read_bytes()).hexdigest() for n in ["legacy-repaired-candidate.csv", "legacy-metadata-repair.jsonl"]}}
    (B / "legacy-metadata-repair-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
