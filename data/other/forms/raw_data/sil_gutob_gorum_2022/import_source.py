"""Prepare or install Gutob/Gorum source files; never run database builds."""
import argparse
import csv
import json
import re
import shutil
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = "20260921-sil-gutob-gorum"
SOURCE = "mathew-chamberlain2022bonda-didayi"
SITES = {"GUT": ("gu", "Tikrapada Gutob"), "PAR": ("go", "Kinumun Parenga Parja")}
DIALECTS = {r["site_code"]: r["tag"] for r in json.loads((ROOT / "site-metadata.json").read_text())}
PRONOUNS = {
    202: ("I", "1sg"), 203: ("you", "2sg informal"),
    204: ("you", "2sg formal"), 205: ("he", "3sg m"),
    206: ("she", "3sg f"), 207: ("we", "1pl inclusive"),
    208: ("we", "1pl exclusive"), 209: ("you", "2pl"), 210: ("they", "3pl"),
}


def segments(raw):
    """Preserve segment positions and group labels, including empty markers."""
    result, group = [], None
    for position, piece in enumerate(raw.rstrip(",").split(","), 1):
        piece = unicodedata.normalize("NFC", piece.strip())
        match = re.match(r"^(\d+)\s*(.*)$", piece)
        if match:
            group, form = match.groups()
        else:
            form = piece
        compact = "".join(c for c in unicodedata.normalize("NFD", form).lower() if c.isalnum())
        missing = group == "0" or compact == "noentry" or form in {"", "---", "DISQUALIFIED"}
        if not missing and group is None:
            raise ValueError(f"Missing similarity code: {raw!r}")
        result.append({"position": position, "group": group, "form": form,
                       "status": "missing" if missing else "response"})
    return result


def prepare():
    with (ROOT / "source-cells.tsv").open(newline="") as handle:
        source = list(csv.DictReader(handle, delimiter="\t"))
    expected = {(i, site) for i in range(1, 211) for site in SITES}
    if len(source) != 420 or {(int(r["Item"]), r["Site_Code"]) for r in source} != expected:
        raise ValueError("Source topology is not 210 prompts by two sites")
    rows, audit = [], []
    for cell in sorted(source, key=lambda r: (int(r["Item"]), r["Site_Code"])):
        item, site = int(cell["Item"]), cell["Site_Code"]
        language, label = SITES[site]
        if cell["Site_Name"] != label:
            raise ValueError("Source site-label mismatch")
        entry = {"source_cell": cell, "segments": segments(cell["Raw_Response"]),
                 "emitted_rows": [], "uncertainty_types": [], "status": "draft"}
        if cell["Extraction_Status"] == "disqualified":
            entry["status"] = "source-disqualified"
            audit.append(entry)
            continue
        if (item, site) == (83, "GUT"):
            entry.update(status="withheld-source-corruption",
                         reason="PDF32/printed27 visibly prints b18 ti̪; no defensible reading for embedded digits. Retain source, do not infer from neighbouring lists.")
            entry["uncertainty_types"].append("transcription")
            audit.append(entry)
            continue
        if 182 <= item <= 201:
            entry["uncertainty_types"].append("grammatical-scope")
        seen = {}
        gloss, grammar = PRONOUNS.get(item, (cell["Gloss"], ""))
        for segment in entry["segments"]:
            if segment["status"] == "missing":
                continue
            form = segment["form"]
            if form in seen:
                segment.update(status="same-cell-repeat", target_entry_key=seen[form])
                continue
            key = f"silbondadidayi1997:{site.lower()}:i{item:03d}:r{segment['position']}"
            seen[form] = key
            segment["entry_key"] = key
            tags = grammar.split()
            if item in PRONOUNS:
                tags.insert(0, "pron")
            if entry["uncertainty_types"]:
                tags.append("uncertain")
            tags.append(DIALECTS[site])
            citation = f"{SOURCE}[Appendix B, printed p. {cell['Printed_Page']}, item {item}, {label}]"
            row = [language, "", form, gloss, "", "", "", citation, "", "", key,
                   "", "", "", " ".join(tags)]
            rows.append(row)
            entry["emitted_rows"].append(row)
        if not entry["emitted_rows"]:
            entry["status"] = "source-no-response"
        audit.append(entry)
    return rows, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--install", action="store_true", help="Copy reviewed CSV/YAML source files only")
    args = parser.parse_args()
    if not args.output and not args.install:
        parser.error("provide --output or --install")
    if args.output is None:
        args.output = ROOT
    rows, audit = prepare()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "draft.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    (args.output / "audit.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audit))
    summary = {"source_cells": len(audit), "draft_rows": len(rows),
               "disqualified": sum(a["status"] == "source-disqualified" for a in audit),
               "no_response": sum(a["status"] == "source-no-response" for a in audit),
               "withheld_corrupt": sum(a["status"] == "withheld-source-corruption" for a in audit),
               "same_cell_repetitions": sum(s["status"] == "same-cell-repeat" for a in audit for s in a["segments"]),
               "uncertain_rows": sum("uncertain" in r[14].split() for r in rows), "installed_rows": 0}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.install:
        shutil.copyfile(args.output / "draft.csv", ROOT.parent.parent / f"{STEM}.csv")
        shutil.copyfile(ROOT / f"{STEM}.yaml", ROOT.parent.parent / f"{STEM}.yaml")
        summary["installed_rows"] = len(rows)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
