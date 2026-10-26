"""Real parser, isolated source-only fixture; never writes canonical outputs."""
import csv
import hashlib
import io
import json
import shutil
import sys
import unicodedata as ud
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "data/SOURCE_INGESTION_CHECKLIST.md").exists())
DATA = ROOT / "data"
sys.path.insert(0,str(DATA))
import make_cldf
import source_meta
from tags import GRAMMATICAL_TAGS, GENDER_TAGS

FIELDS = ("lang","param","old_form","gloss","native","ipa","notes","source","cognateset","etymology","entry_key","variant_of_key","borrowed_from_key","derivation_parent_keys","tags")
FIXTURE = HERE / "parser-fixture/data/other/forms"
STEM = "20260927-nair-mullu-kurumba"


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    paths = [HERE / name for name in ("proposed.csv","proposal-audit.jsonl","proposed-settings.yaml","proposed-references.bib","metadata-proposal.json","graph-proposal.json")]
    pins = {p.name:sha(p) for p in paths}
    rows = list(csv.reader((HERE / "proposed.csv").open(newline="")))
    assert len(rows)==135
    registry = {r[0] for r in csv.reader((DATA / "cldf/languages.csv").open(newline=""))}
    for row in rows:
        assert len(row)==15 and row[0] in registry
        assert set(row[14].split()) <= GRAMMATICAL_TAGS | GENDER_TAGS
    audit = [json.loads(s) for s in (HERE / "proposal-audit.jsonl").read_text().splitlines()]
    assert len(audit)==65
    keys = {r[10] for r in rows}
    assert keys == {k for a in audit for k in a["entry_keys"]}
    assert len(keys)==len(rows)
    assert sum(bool(r[11]) for r in rows)==11
    assert all(not r[11] or r[11] in keys for r in rows)
    FIXTURE.mkdir(parents=True,exist_ok=True)
    settings = FIXTURE / f"{STEM}.yaml"
    shutil.copyfile(HERE / "proposed-settings.yaml",settings)
    original_yaml_files = source_meta.yaml_files
    checks = {}
    try:
        source_meta.yaml_files = lambda:[settings]
        source_meta.load.cache_clear()
        for normalization in ("NFC","NFD"):
            input_rows = [[ud.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rows]
            path = FIXTURE / f"{STEM}.csv"
            with path.open("w",newline="") as out: csv.writer(out).writerows(input_rows)
            errors = io.StringIO()
            parsed,stats = make_cldf.parse_file(str(path),errors,name=STEM)
            assert not errors.getvalue(),errors.getvalue()
            assert len(parsed)==len(rows) and stats=={"converted":0,"for_conversion":0}
            for actual,expected in zip(parsed,input_rows):
                assert [getattr(actual,f) for f in FIELDS]==expected,actual.entry_key
                assert ud.normalize("NFC",actual.form)==ud.normalize("NFC",expected[2]),actual.entry_key
            meta = source_meta.load()
            for key in ('nair2013mullukurumba', 'robertsathyajoseph1982mullukurumba'):
                assert meta.flag(key, 'reference', 'editor') == 'OpenAI Codex'
                assert meta.flag(key, 'reference', 'ocr') is False
                assert meta.flag(key, 'reference', 'etymology_provenance') == ('source' if key == 'nair2013mullukurumba' else 'none')
            assert len({(r.lang, r.param, r.form, make_cldf.source_entry_dedupe_key(r,meta)) for r in parsed})==135
            checks[normalization] = dict(initial_dedupe_keys=135,rows=len(parsed),all_15_fields_exact=True,ordered_keys_exact=True,literal_display=True)
    finally:
        source_meta.yaml_files = original_yaml_files
        source_meta.load.cache_clear()
    assert pins == {p.name:sha(p) for p in paths}
    report = dict(status="passed source-local actual parser; canonical install not performed",inputs=pins,
        checks=checks,registered_languages=sorted({r[0] for r in rows}),registered_tags=True,
        closed_variant_edges=11,audit_key_closure=True,excluded_witness_records=23,commentary_reuses=6,
        fixture_isolation="Only source-local YAML discovery patched; actual generic parse_file is unchanged; this is not a canonical SourceMeta/build acceptance claim.")
    (HERE / "actual-parser-review.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__": main()
