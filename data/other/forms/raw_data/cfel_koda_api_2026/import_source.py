"""Install reviewed CFEL Koda online-dictionary records; never build data/DB."""

import argparse
import csv
import json
import shutil
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = "20260925-cfel-koda-adornments"
SOURCE = "cfel2026koda"
DOMAIN = "Adornments and Costumes"
# These publisher IPA strings have ambiguous source glyphs, an internal slash, or
# an unpositioned aspiration mark. Keep them in the audit pending visual review.
UNRESOLVED = set("?յͪͻᴂ/")


def build():
    queries = json.loads((ROOT / "queries.json").read_text())["queries"]
    proposals = [json.loads(line) for line in (ROOT / "proposals.jsonl").read_text().splitlines()]
    assert len(queries) == len(proposals) == 60
    assert [p["query"] for p in proposals] == queries
    rows, audit = [], []
    ids = set()
    for proposal in proposals:
        assert proposal["reported_count"] == len(proposal["candidates"])
        selected = [c for c in proposal["candidates"] if c["domain_name"] == DOMAIN]
        assert len(selected) == 1, proposal["query"]
        for item in proposal["candidates"]:
            item_id = item["id"]
            assert isinstance(item_id, int) and item_id not in ids
            ids.add(item_id)
            record = {
                "query": proposal["query"],
                "publisher_id": item_id,
                "publisher_concept_id": item["abs_ref"],
                "publisher_domain": item["domain_name"],
                "raw_ipa": item["ipa"],
                "raw_native": item["word"],
                "raw_native_alternates": [value for value in (item.get("word2"), item.get("word3")) if value],
                "raw_grammar": item["grammatical_category"],
                "response_sha256": proposal["response_sha256"],
                "retrieved_utc": proposal["retrieved_utc"],
                "entry_key": f"cfel-koda-api:record:{item_id}",
            }
            if item["domain_name"] != DOMAIN:
                record["status"] = "excluded_other_domain"
            else:
                ipa = item["ipa"].strip()
                assert ipa.startswith("/") and ipa.endswith("/"), item_id
                raw = ipa[1:-1]
                record["source_ipa_inner"] = raw
                if not raw or any(char in raw for char in UNRESOLVED) or proposal["query"] == "Lingerie":
                    record["status"] = "withheld_transcription"
                    record["issue"] = (
                        "transcription:IPA-does-not-cover-entire-native-phrase"
                        if proposal["query"] == "Lingerie" else
                        "transcription:ambiguous-publisher-IPA-glyph-or-boundary"
                    )
                else:
                    assert item["word"] and item["grammatical_category"] in {"Noun", "Verb"}
                    record["status"] = "installed"
                    record["source_locator"] = f"online dictionary, Koda record {item_id}, concept {item['abs_ref']}"
                    row = [
                        "Koda", "", unicodedata.normalize("NFC", raw), proposal["query"],
                        unicodedata.normalize("NFC", item["word"].strip()), "", "",
                        f"{SOURCE}[{record['source_locator']}]", "", "", record["entry_key"],
                        "", "", "", item["grammatical_category"].lower(),
                    ]
                    rows.append(row)
            audit.append(record)
    assert len(audit) == 62 and len(rows) == 47
    assert len({row[10] for row in rows}) == len(rows)
    return rows, audit


def write(output: Path, install: bool = False):
    rows, audit = build()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{STEM}.csv"
    with csv_path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    (output / "audit.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in audit)
    )
    if install:
        forms = ROOT.parents[1]
        shutil.copyfile(csv_path, forms / csv_path.name)
    print(f"{len(rows)} installed candidates; {len(audit)} source records audited; database not built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    write(args.output, args.install)
