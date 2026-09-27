"""Read-only verification of the 2026-09-14 surveys in existing shared CLDF.

Run with --output PATH to save evidence. This never builds or rewrites CLDF/DBs.
Only source-owned rows are retained in memory while shared tables are streamed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path

import make_cldf

ROOT = Path(__file__).resolve().parent
ADAPTER_PATH = ROOT / "data/other/forms/raw_data/integrate_manual_surveys_2026.py"
spec = importlib.util.spec_from_file_location("manual_surveys_adapter", ADAPTER_PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def rows(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def sha256(path):
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify():
    expected, source_keys, inputs = {}, {}, []
    for name, source in adapter.SOURCES.items():
        prepared, _ = adapter.prepare(name)
        adapter.validate_registry(prepared)
        path = f"data/other/forms/{source['output']}"
        inputs.extend([path, path.removesuffix(".csv") + ".yaml"])
        errors = io.StringIO()
        parsed, _ = make_cldf.parse_file(path, errors, file_num=source["output"].removesuffix(".csv"))
        assert not errors.getvalue(), errors.getvalue()
        assert len(parsed) == source["count"]
        expected.update({row.entry_key: (name, row) for row in parsed})
        source_keys[name] = {row["Entry_Key"] for row in prepared}
    legacy = {r["Legacy_ID"]: r["Source_Key"] for r in rows("cldf/form-source-keys.csv")
              if r["Source_Key"] in expected}
    assert set(legacy.values()) == set(expected), "Missing compiled source keys"
    key_ids = {legacy[r["Legacy_ID"]]: r["Form_ID"] for r in rows("cldf/form-id-aliases.csv")
               if r["Legacy_ID"] in legacy}
    assert set(key_ids) == set(expected), "Missing persistent ID aliases"
    assert len(set(key_ids.values())) == len(expected), "Distinct survey records collapsed"
    ids = set(key_ids.values())
    citations = {s["source"] for s in adapter.SOURCES.values()}
    compiled = {}
    for row in rows("cldf/forms.csv"):
        cited = {s.split("[", 1)[0] for s in row["Source"].split(";")}
        if row["ID"] in ids or cited & citations:
            assert row["ID"] not in compiled, "Duplicate compiled ID"
            compiled[row["ID"]] = row
    assert set(compiled) == ids, "Missing or unexpected source-bearing nodes"
    registry = {(r["Source_Key"], r["Form_ID"]) for r in rows("data/form-identities.csv")
                if r["Source_Key"] in expected and r["Status"] == "active"}
    assert registry == set(key_ids.items()), "Persistent identity registry mismatch"
    edges = [r for r in rows("cldf/edges.csv") if r["Child_ID"] in ids]
    expected_edges = set()
    for key, (name, raw) in expected.items():
        row = compiled[key_ids[key]]
        for column, value in {"Language_ID": raw.lang, "Form": raw.form,
                              "Original": raw.old_form, "Phonemic": raw.ipa,
                              "Gloss": raw.gloss, "Source": raw.source}.items():
            assert row[column] == value, (key, column, row[column], value)
        assert set(raw.tags.split()) <= set(row["Tags"].split()), (key, "tags")
        assert "�" not in row["Form"] and row["Form"].strip()
        if raw.variant_of_key:
            expected_edges.add((row["ID"], key_ids[raw.variant_of_key], "variant", "1"))
        assert row["Status"] == ("" if raw.variant_of_key else "unlinked"), (key, "status")
    actual_edges = {(e["Child_ID"], e["Parent_ID"], e["Kind"], e["Rank"]) for e in edges}
    assert len(edges) == len(actual_edges) == 46
    assert actual_edges == expected_edges, "Variant or ancestry graph mismatch"
    concepts = Counter(r["Form_ID"] for r in rows("cldf/form_concepts.csv") if r["Form_ID"] in ids)
    alignments = Counter(r["Form_ID"] for r in rows("cldf/alignments.csv") if r["Form_ID"] in ids)
    paths = inputs + ["cldf/" + p for p in ["forms.csv", "edges.csv", "form-source-keys.csv",
        "form-id-aliases.csv", "references.csv", "languages.csv", "dialects.csv",
        "form_concepts.csv", "concepts.csv", "alignments.csv", "sources.bib"]]
    paths += ["data/form-identities.csv", "errors.txt", "verify_manual_surveys.py",
              str(ADAPTER_PATH.relative_to(ROOT))]
    paths += [f"conversion/{p}.txt" for p in ("sil-ho", "sil-bhumij", "sil-dhurwa-2021")]
    assert not (ROOT / "errors.txt").read_text().strip(), "Current conversion errors are nonempty"
    return {
        "scope": "Read-only verification of existing shared CLDF; no database rebuild",
        "sources": {name: {
            "installed_rows": len(keys), "compiled_nodes": len(keys),
            "statuses": dict(Counter(compiled[key_ids[k]]["Status"] for k in keys)),
            "nodes_with_concepts": sum(key_ids[k] in concepts for k in keys),
            "nodes_with_alignments": sum(key_ids[k] in alignments for k in keys),
            "representative": {"entry_key": min(keys), "form_id": key_ids[min(keys)]},
        } for name, keys in source_keys.items()},
        "variant_edges": len(edges), "conversion_errors": 0,
        "sha256": {p: sha256(p) for p in paths},
        "deferred": ["New full build prohibited by user", "Full suite requires suitable authorized runner",
                     "Global retrospective manifest regeneration", "Browser refresh prohibited; browser QA not run"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(verify(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    print(result)
