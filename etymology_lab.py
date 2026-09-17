#!/usr/bin/env python3
"""Save an etymology-lab research pass into the per-source etymology sidecars.

Every research pass under ``curation/etymology-lab/`` used to ship its own ``*_save.py`` copying
one template: validate the accepted decisions against the compiled graph, apply them to a scratch
copy of ``cldf/edges.csv`` to prove they are idempotent and touch nothing else, hash the inputs
before and after, back the whole overlay up, append the rows, and write per-language batch
manifests.  This module is that template, once, with two changes:

* rows are written through :mod:`etymology_assignments`, so each lands in the sidecar of the
  source that owns its form instead of one central file;
* instead of copying every overlay file before writing (the ``backups/`` directories grew to
  gigabytes), the pass ledger records the sha256 of every sidecar before and after plus the exact
  rows written, which is enough to audit or revert the pass.

Decision files keep the established shape::

    {"accepted": [{"record": {"ID": "f_…", "Language_ID": …, "Form": …, "Gloss": …,
                              "Source": …, "Tags": …},
                   "parent": "<etymon node id>", "kind": "reflex" | "borrowed" | …,
                   "citation": "<evidence citation>", "evidence": "<prose>"}, …],
     "held": [...]}

Usage::

    uv run python etymology_lab.py save PASS_DIR/decisions.json --pass alternant \\
        --note "Joint SIL review 2026-09-14." [--authorization "..."] [--dry-run]

The pass name prefixes the ledger files written next to the decisions file
(``<pass>-saved-assignments.json``, ``<pass>-validation.json``, ``<pass>-manifest-paths.json``).
A pass refuses to run twice: any accepted target that already has an overlay row aborts the save.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import etymology_assignments as overlay  # noqa: E402
from assign_form_ids import apply_assignments, validate_assignments  # noqa: E402

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
LAB = ROOT / "curation/etymology-lab"
FORMS = ROOT / "cldf/forms.csv"
EDGES = ROOT / "cldf/edges.csv"
REGISTRY = ROOT / "data/form-identities.csv"
RECORD_FIELDS = ("Language_ID", "Form", "Gloss", "Source", "Tags")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def rows_from_decisions(accepted: list[dict], note: str) -> list[dict[str, str]]:
    suffix = f" {note}" if note else ""
    return [
        dict(
            Form_ID=item["record"]["ID"], Etymon_ID=item["parent"],
            Kind=item.get("kind", "reflex"), Rank=str(item.get("rank", "1")),
            Status="accepted", Source=item.get("citation", ""),
            Notes=(item.get("evidence", "") + suffix).strip(), Pos=str(item.get("pos", "")),
        )
        for item in accepted
    ]


def validate_against_graph(accepted: list[dict], rows: list[dict[str, str]]) -> dict:
    """Every check the historical save scripts made, against the compiled graph on disk."""
    targets = {item["record"]["ID"] for item in accepted}
    if len(targets) != len(accepted):
        raise ValueError("a record is accepted twice in this pass")
    existing = [r for r in overlay.read_assignments() if r["Form_ID"] in targets]
    if existing:
        raise ValueError(
            f"{len(existing)} target(s) already have overlay rows; reconcile explicitly "
            f"(e.g. {existing[0]['Form_ID']} in {existing[0].path})"
        )
    needed = targets | {item["parent"] for item in accepted}
    forms, selected = [], {}
    for row in csv.DictReader(FORMS.open(encoding="utf-8", newline="")):
        forms.append({"ID": row["ID"], "Status": row["Status"]})
        if row["ID"] in needed:
            selected[row["ID"]] = row
    missing = needed - set(selected)
    if missing:
        raise ValueError(f"nodes absent from cldf/forms.csv: {sorted(missing)[:5]}")
    for item in accepted:
        record = selected[item["record"]["ID"]]
        if record["Redirect"]:
            raise ValueError(f"{record['ID']} is a redirect")
        drift = [k for k in RECORD_FIELDS if k in item["record"] and record[k] != item["record"][k]]
        if drift:
            raise ValueError(f"{record['ID']} changed since the decision was made: {drift}")
        parent = selected[item["parent"]]
        if parent["Redirect"]:
            raise ValueError(f"parent {parent['ID']} is a redirect")
        if parent["Status"] == "unlinked":
            raise ValueError(f"parent {parent['ID']} is an unlinked node")
    validate_assignments(forms, rows)
    registered = {
        r["Form_ID"] for r in csv.DictReader(REGISTRY.open(encoding="utf-8", newline=""))
        if r["Form_ID"] in targets and r["Status"] == "active"
    }
    if registered != targets:
        raise ValueError(f"targets without an active durable identity: {sorted(targets - registered)[:5]}")
    # Apply to a scratch copy of the graph: idempotent, and no unrelated edge moves.
    with tempfile.TemporaryDirectory(prefix="etymology-lab-") as scratch:
        graph = Path(scratch) / "edges.csv"
        shutil.copyfile(EDGES, graph)
        first = apply_assignments(graph, forms, rows)
        second = apply_assignments(graph, forms, rows)
        if second != 0:
            raise ValueError("pass is not idempotent against the compiled graph")
        _, original = read_rows(EDGES)
        _, result = read_rows(graph)
        if [e for e in original if e["Child_ID"] in targets and e["Rank"] == "1"]:
            raise ValueError("a target already has a rank-1 edge in the compiled graph")
        untouched = lambda edges: [e for e in edges if e["Child_ID"] not in targets]  # noqa: E731
        if untouched(result) != untouched(original):
            raise ValueError("applying the pass moved an unrelated edge")
        expected = {(r["Form_ID"], r["Etymon_ID"], r["Kind"], r["Rank"], r["Pos"]) for r in rows}
        actual = {
            (e["Child_ID"], e["Parent_ID"], e["Kind"], e["Rank"], e["Pos"])
            for e in result if e["Child_ID"] in targets and e["Rank"] == "1"
        }
        if actual != expected:
            raise ValueError("compiled edges after the pass do not match the accepted rows")
    return {"forms": selected, "firstApplicationChanges": first, "secondApplicationChanges": second}


def write_language_manifests(
    accepted: list[dict], rows: list[dict[str, str]], selected: dict, *,
    pass_dir: Path, validation_path: Path, authorization: str, saved_at: str,
) -> list[str]:
    by_language: dict[str, list[dict]] = defaultdict(list)
    for item in accepted:
        by_language[item["record"]["Language_ID"]].append(item)
    paths = []
    for language, items in sorted(by_language.items()):
        dest = LAB / language
        dest.mkdir(parents=True, exist_ok=True)
        numbers = [int(p.stem.split("-")[1]) for p in dest.glob("batch-*.json") if p.stem.split("-")[1].isdigit()]
        number = max(numbers, default=0) + 1
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for item in items:
            grouped[(item["parent"], item.get("citation", ""), item.get("evidence", ""), item.get("kind", "reflex"))].append(item)
        proposals = []
        for index, ((parent, citation, evidence, kind), members) in enumerate(grouped.items(), 1):
            ids = {m["record"]["ID"] for m in members}
            proposals.append(dict(
                number=index, status="saved", parentId=parent, parentForm=selected[parent]["Form"],
                kind=kind, citation=citation, evidence=evidence, formIds=sorted(ids),
                records=[m["record"] for m in members],
                assignments=[r for r in rows if r["Form_ID"] in ids],
            ))
        path = dest / f"batch-{number:03d}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(dict(
            language=language, batch=number, status="saved", savedAt=saved_at,
            authorization=authorization, researchDirectory=str(pass_dir),
            validation=str(validation_path), proposals=proposals,
        ), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        paths.append(str(path))
    return paths


def save(decisions: Path, *, pass_name: str, note: str, authorization: str, dry_run: bool = False) -> dict:
    pass_dir = decisions.resolve().parent
    data = json.loads(decisions.read_text(encoding="utf-8"))
    accepted = data["accepted"]
    if not accepted:
        raise ValueError("nothing accepted in this pass")
    rows = rows_from_decisions(accepted, note)
    watched = [REGISTRY, FORMS, EDGES, *overlay.assignment_files()]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in watched}
    checks = validate_against_graph(accepted, rows)
    if {str(p.relative_to(ROOT)): sha256(p) for p in watched} != before:
        raise RuntimeError("inputs changed during validation; retry from fresh input")
    report = dict(
        pass_name=pass_name, decisions=str(decisions), assignmentRows=len(rows),
        affectedRecords=len({r["Form_ID"] for r in rows}),
        parentNodes=len({r["Etymon_ID"] for r in rows}),
        firstApplicationChanges=checks["firstApplicationChanges"],
        secondApplicationChanges=checks["secondApplicationChanges"],
        hashesBefore=before, dryRun=dry_run,
    )
    if dry_run:
        print(json.dumps(report, indent=2))
        return report
    resolver = overlay.SidecarResolver()
    written = overlay.write_assignments(overlay.read_assignments() + rows, resolver)
    touched = sorted(str(p.relative_to(ROOT)) for p in written if p in written)
    after = {str(p.relative_to(ROOT)): sha256(p) for p in [REGISTRY, FORMS, EDGES, *overlay.assignment_files()]}
    if any(after[k] != before[k] for k in (str(p.relative_to(ROOT)) for p in (REGISTRY, FORMS, EDGES))):
        raise RuntimeError("a compiled input changed during the save")
    saved_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    report.update(savedAt=saved_at, hashesAfter=after,
                  sidecarsChanged=sorted(k for k in after if before.get(k) != after[k]),
                  sidecarsWritten=touched)
    (pass_dir / f"{pass_name}-saved-assignments.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    validation_path = pass_dir / f"{pass_name}-validation.json"
    validation_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifests = write_language_manifests(
        accepted, rows, checks["forms"], pass_dir=pass_dir, validation_path=validation_path,
        authorization=authorization, saved_at=saved_at,
    )
    (pass_dir / f"{pass_name}-manifest-paths.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("hashesBefore", "hashesAfter")}, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("save", help="validate and save one research pass into the sidecars")
    s.add_argument("decisions", type=Path)
    s.add_argument("--pass", dest="pass_name", required=True, help="pass name; prefixes the ledger files")
    s.add_argument("--note", default="", help="appended to every row's Notes (e.g. review date)")
    s.add_argument("--authorization", default="", help="recorded in the language batch manifests")
    s.add_argument("--dry-run", action="store_true", help="validate only; write nothing")
    args = parser.parse_args(argv)
    save(args.decisions, pass_name=args.pass_name, note=args.note,
         authorization=args.authorization, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
