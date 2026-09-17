#!/usr/bin/env python3
"""Install the frozen Ho, Bhumij and Dhurwa manual packages into shared input CSVs.

No OCR or PDF text supplies readings. Frozen source-local files remain unchanged;
this adapter only maps ISO staging IDs to existing Jambu languages/dialect tags,
removes audit-only prose, and records those changes by immutable entry key.
Run without --install for a dry validation; --output-dir writes review proposals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
SOURCES = {
    "ho": {
        "package": "sil_ho_2024", "staged": "staged_forms.csv", "count": 2900,
        "source": "varenkamp2024ho", "parent": "ho", "staging_parent": "ho",
        "sha256": "14726341266f1769100f6d0d37986cb55ee3aa4f9257ec5a517cdee65b570480",
        "output": "20260914-sil-ho.csv",
    },
    "bhumij": {
        "package": "sil_bhumij_2015", "staged": "staged_forms.tsv", "count": 2100,
        "source": "baileymaggard2015bhumij", "parent": "mu", "staging_parent": "unr",
        "sha256": "1196571ed479289507cbebe89467097a3d44f4e05ced96069e9cc4baff0ae4e9",
        "output": "20260914-sil-bhumij.csv",
    },
    "dhurwa": {
        "package": "sil_dhurwa_2021", "staged": "checkpoint_forms.csv", "count": 809,
        "source": "josephmichael2021dhurwa", "parent": "Parji", "staging_parent": "pci",
        "sha256": "20150f3058da6a8c55f854d63d163ed8b374abb10ff03348347bcdf4baf8cf9a",
        "output": "20260914-sil-dhurwa.csv",
    },
}


def read_staged(name: str) -> list[dict[str, str]]:
    spec = SOURCES[name]
    path = HERE / spec["package"] / spec["staged"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"{name}: frozen source file changed; review before installation")
    with path.open(encoding="utf-8", newline="") as stream:
        if path.suffix == ".tsv":
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames != FIELDS:
                raise ValueError(f"{name}: unexpected staged header")
            rows = list(reader)
        else:
            raw = list(csv.reader(stream))
            if any(len(row) != len(FIELDS) for row in raw):
                raise ValueError(f"{name}: expected rich 15-column rows")
            rows = [dict(zip(FIELDS, row)) for row in raw]
    keys = {row["Entry_Key"] for row in rows}
    if len(rows) != spec["count"] or len(keys) != len(rows) or "" in keys:
        raise ValueError(f"{name}: unexpected counts or duplicate/missing keys")
    for row in rows:
        if row["Language_ID"] != spec["staging_parent"]:
            raise ValueError(f"{name}: unexpected staging language")
        if row["Source"].split("[", 1)[0] != spec["source"]:
            raise ValueError(f"{name}: unexpected source")
        if any(row[field] for field in ["Parameter_ID", "Borrowed_From_Key", "Derivation_Parent_Keys", "Cognateset", "Etymology"]):
            raise ValueError(f"{name}: unexpected etymology claim")
        if row["Variant_Of_Key"] and row["Variant_Of_Key"] not in keys:
            raise ValueError(f"{name}: missing variant parent")
    return rows


def prepare(name: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    spec = SOURCES[name]
    installed, audit = [], []
    cells = {}
    if name == "bhumij":
        with (HERE / spec["package"] / "staged_audit.tsv").open() as stream:
            cells = {r["Stable_Cell_ID"]: r for r in csv.DictReader(stream, delimiter="\t")}
    for original in read_staged(name):
        row = original.copy()
        row["Language_ID"] = spec["parent"]
        prefix = f"dialect:{spec['staging_parent']}:"
        tags = row["Tags"].split()
        dialects = [tag for tag in tags if tag.startswith(prefix)]
        if len(dialects) != 1:
            raise ValueError(f"{name}: expected one qualified source dialect")
        row["Tags"] = " ".join(
            f"dialect:{spec['parent']}:" + tag[len(prefix):]
            if tag.startswith(prefix) else tag for tag in tags
        )
        reasons = []
        correction = ""
        if name == "ho" and row["Entry_Key"].endswith("-i093"):
            if row["Gloss"] != "tall":
                raise ValueError("Ho item 93 correction expects frozen gloss 'tall'")
            row["Gloss"] = "tail"
            correction = "gloss: tall -> tail; visually verified in Varenkamp 2024, physical p. 102 / printed p. 93, item 93"
        if name == "bhumij" and "bhumij-mundari1989-udala" in row["Entry_Key"]:
            row["Tags"] += " uncertain"
            reasons.append("dialect-mapping: source labels Udala 'Mundari? Bhumij?'; retained under Mundari without resolving the label")
        # Extraction method, group labels and redundant site/page prose remain
        # in the frozen audit and this integration ledger, not lexical notes.
        row["Notes"] = ""
        if name == "bhumij":
            cell = cells[row["Entry_Key"].rsplit("-a", 1)[0]]
            qualifier = cell["Uncertainty"]
            if qualifier.startswith("source parenthetical qualifier:"):
                row["Notes"] = qualifier.removeprefix("source parenthetical qualifier:").strip() + " (source qualifier)"
            elif "source appends '(?)'" in qualifier:
                row["Tags"] = " ".join(dict.fromkeys(row["Tags"].split() + ["uncertain"]))
                row["Notes"] = "Source qualifier: (?)"
                reasons.append("source-qualification: " + qualifier)
        audit.append({
            "Entry_Key": row["Entry_Key"], "Source": row["Source"],
            "Staged_Language_ID": original["Language_ID"],
            "Language_ID": row["Language_ID"], "Staged_Tags": original["Tags"],
            "Tags": row["Tags"], "Form": row["Form"], "Gloss": row["Gloss"],
            "Staged_Gloss": original["Gloss"], "Correction": correction,
            "Phonemic": row["Phonemic"], "Variant_Of_Key": row["Variant_Of_Key"],
            "Audit_Only_Notes": original["Notes"], "Installed_Notes": row["Notes"],
            "Uncertainty": "; ".join(reasons), "Disposition": "installed-input",
        })
        installed.append(row)
    return installed, audit


def validate_registry(rows: list[dict[str, str]]) -> None:
    with (ROOT / "cldf/languages.csv").open() as stream:
        languages = {row["ID"] for row in csv.DictReader(stream)}
    with (ROOT / "cldf/dialects.csv").open() as stream:
        dialects = {row["Tag"]: row["Language_ID"] for row in csv.DictReader(stream)}
    for row in rows:
        if row["Language_ID"] not in languages:
            raise ValueError(f"Missing language: {row['Language_ID']}")
        for tag in row["Tags"].split():
            if tag.startswith("dialect:") and dialects.get(tag) != row["Language_ID"]:
                raise ValueError(f"Missing or mismatched dialect: {tag}")


def sample_audit(seed: int = 20260914) -> dict:
    """Reproducible frozen-cell-ledger vs installed-row sample, twenty per source."""
    result = {"seed": seed, "scope": "frozen manual cell ledgers versus installed inputs; primary page spot-checks are recorded separately", "sources": {}}
    for name, spec in SOURCES.items():
        audit_file = "checkpoint_audit.tsv" if name == "dhurwa" else "staged_audit.tsv"
        with (HERE / spec["package"] / audit_file).open() as stream:
            cells = list(csv.DictReader(stream, delimiter="\t"))
        candidates = [r for r in cells if r["Review_Status"] == "attested" and (r.get("Scope") in {"target", "known_dhurwa_target"})]
        with (HERE.parent / spec["output"]).open(newline="") as stream:
            installed = [dict(zip(FIELDS, r)) for r in csv.reader(stream)]
        sampled = random.Random(f"{seed}:{name}").sample(candidates, 20)
        records = []
        for cell in sampled:
            site, item = cell["Site_Code"], cell["Item"]
            matching = [r for r in installed if f"item {item}, list {site}]" in r["Source"]]
            literal = cell["Manual_Transcription"]
            if name == "ho":
                # Printed comparison group labels are separate from lexical text.
                expected = [re.sub(r"(^|,\s*)[0-9]+(?:,\s*[0-9]+)*\s+", r"\1", literal).strip()]
            elif name == "bhumij":
                expected = literal.split(" | ")
            else:
                expected = [part.strip() for part in literal.split("/") if part.strip()]
            reviewed_gloss = "tail" if name == "ho" and item == "93" else cell["Gloss"]
            checks = {
                "forms": [r["Form"] for r in matching] == expected,
                "gloss": all(r["Gloss"] == reviewed_gloss for r in matching),
                "canonical_parent": all(r["Language_ID"] == spec["parent"] for r in matching),
                "phonemic_preserved": all(r["Phonemic"] == r["Form"] for r in matching),
                "no_ancestry": all(not r["Parameter_ID"] and not r["Borrowed_From_Key"] for r in matching),
            }
            records.append({
                "item": item, "site": site, "pdf_page": cell["PDF_Page"],
                "printed_page": cell["Printed_Page"], "column": cell["Column"],
                "raw_manual_transcription": literal, "gloss": reviewed_gloss,
                "staged_gloss": cell["Gloss"],
                "installed_forms": [r["Form"] for r in matching],
                "entry_keys": [r["Entry_Key"] for r in matching],
                "checks": checks, "material_error": not all(checks.values()),
            })
        result["sources"][name] = {"sample_size": 20, "material_errors": sum(r["material_error"] for r in records), "records": records}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--audit-seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.install and args.output_dir:
        parser.error("Use --install or --output-dir, not both")
    proposals = {name: prepare(name) for name in SOURCES}
    if args.install:
        for rows, _ in proposals.values():
            validate_registry(rows)
    target = HERE.parent if args.install else args.output_dir
    if target:
        target.mkdir(parents=True, exist_ok=True)
        for name, (rows, audit) in proposals.items():
            output = target / SOURCES[name]["output"]
            with output.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows([[row[k] for k in FIELDS] for row in rows])
            audit_path = (HERE if args.install else target) / f"20260914-sil-{name}-integration-audit.csv"
            with audit_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
                writer.writeheader()
                writer.writerows(audit)
    if args.audit_output:
        result = sample_audit(args.audit_seed)
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if any(r["material_errors"] for r in result["sources"].values()):
            raise ValueError("Seeded audit found material mismatches; inspect the audit output")
    print(json.dumps({name: {"rows": len(rows), "variants": sum(bool(r['Variant_Of_Key']) for r in rows), "dialects": len({r['Tags'].split()[0] for r in rows})} for name, (rows, _) in proposals.items()}, indent=2))


if __name__ == "__main__":
    main()
