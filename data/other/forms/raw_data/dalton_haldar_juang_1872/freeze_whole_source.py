"""Pin local Juang proposal/evidence and a deterministic whole-output review sample."""
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "whole-source-recovery"
DATA = ROOT.parents[4]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = [
        "import_source.py", "prepare_whole_source.py", "install_whole_source.py", "freeze_whole_source.py",
        "reviewed_inventory.tsv", "comparative-extension.tsv", "separate-list-inventory.tsv",
        "20260925-dalton-haldar-juang.csv",
    ]
    names += ["whole-source-recovery/" + n for n in (
        "proposal.csv", "proposal-audit.jsonl", "proposal-profile.txt", "proposal-sources.bib",
        "20260925-dalton-haldar-juang.yaml", "proposal-summary.json", "profile-review-20260927.json",
        "chapter-prose-first-reading.jsonl", "chapter-page-census-first-reading.json",
        "outside-chapter-index-review-20260927.json", "scope-and-ownership-20260927.json",
        "historical-held-19-review-packet.json", "owner-table-second-reading-20260927.json",
        "owner-separate-list-corrections-20260927.json", "root-held-original-review-20260927.json",
        "root-retained-corrections-review-20260927.json", "root-prose-attribution-review-20260927.json",
        "root-cow-p241-review-20260927.json",
    )]
    # Test evidence is pinned by its actual repository path in addition to input files.
    tests = DATA / "tests/test_dalton_juang_whole_recovery.py"
    canonical = DATA / "data/other/forms/20260925-dalton-haldar-juang.csv"
    manifest = {"status":"frozen for independent whole-output review; not installation approval",
                "files":{name:sha(ROOT/name) for name in names},
                "test_file":"tests/test_dalton_juang_whole_recovery.py", "test_sha256":sha(tests),
                "canonical_before_sha256":sha(canonical),
                "rows":226,"source_units":373,"historical_keys":196,
                "source_original_sha256":"38dda30a06f7c56df34311328c36d3fdf8cb9835a5420aea4453fbb1b76008b1",
                "scope":"Juang chapter152–158, all361 table cells235–242, explicit grammar, and source-wide documentary census.",
                "deferred":"Full CLDF/database builds, full suite and browser QA not authorized."}
    (OUT/"frozen-manifest-20260927.json").write_text(json.dumps(manifest,indent=2)+"\n")
    rows = list(csv.reader((OUT/"proposal.csv").open()))
    by_key = {r[10]:r for r in rows}
    units = [json.loads(x) for x in (OUT/"proposal-audit.jsonl").read_text().splitlines()]
    rng = random.Random(2026092727)
    groups = {
        "retained-table":([u for u in units if u.get("historical_decision",{}).get("status", "").startswith("selected")],12),
        "recovered-hold":([u for u in units if u.get("historical_decision",{}).get("status") == "held"],5),
        "chapter-prose":([u for u in units if ":prose:" in u["source_unit_key"] and u["entry_keys"]],2),
        "explicit-morphology":([u for u in units if u["status"] == "explicit-morphology"],1),
    }
    selected = []
    for group,(candidates,n) in groups.items():
        for unit in rng.sample(candidates,n):
            selected.append(dict(unit, sample_stratum=group, actual_rows=[by_key[k] for k in unit["entry_keys"]]))
    blank = [u for u in units if u["status"] == "printed-blank"]
    documentary = [u for u in units if ":prose:" in u["source_unit_key"] and not u["entry_keys"]]
    packet = {"seed":2026092727,"status":"selected; independent original/output review pending",
              "frozen_manifest_sha256":sha(OUT/"frozen-manifest-20260927.json"),
              "sample_units":selected,"controls":rng.sample(blank,3)+rng.sample(documentary,2),
              "additional_morphology_edges":[dict(u,actual_rows=[by_key[k] for k in u["entry_keys"]]) for u in units if u["status"]=="explicit-morphology"],
              "source_locator":"printed_page/pdf_page; table row/section or prose source_quote in each unit; PDF=/private/tmp/dalton-1872.pdf"}
    (OUT/"independent-sample-selection-2026092727.json").write_text(json.dumps(packet,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"manifest_sha256":packet["frozen_manifest_sha256"],"sample_units":len(selected),"sample_rows":sum(len(u["actual_rows"]) for u in selected),"controls":5},indent=2))


if __name__ == "__main__":
    main()
