"""Source-grounded Mannan proposal checks; no canonical installation/build."""
import csv
import importlib.util
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "data/other/forms/raw_data/nair_mannan_2013"


def rows():
    return list(csv.reader((PACKAGE / "proposed.csv").open(newline="")))


def test_complete_scope_and_closed_source_keys():
    source = [json.loads(s) for s in (PACKAGE / "physical-units-first-reading.jsonl").read_text().splitlines()]
    audit = [json.loads(s) for s in (PACKAGE / "proposal-audit.jsonl").read_text().splitlines()]
    candidate = rows()
    assert len(source)==len(audit)==140
    assert [u["source_unit_key"] for u in source]==[a["source_unit_key"] for a in audit]
    witnesses = [w for u in source for w in u["source_witnesses"]]
    assert len(witnesses)==208
    assert len(candidate)==154 and sum(r[0]=="Mannan" for r in candidate)==123
    assert len({r[10] for r in candidate})==154
    assert {k for a in audit for k in a["proposed_entry_keys"]}=={r[10] for r in candidate}
    names = next(u for u in source if u["source_unit_key"].endswith("traditional-names:1"))
    assert len(names["source_witnesses"])==30
    assert not {w["witness_key"] for w in names["source_witnesses"]} & {r[10] for r in candidate}
    assert any(u["printed_page"]==50 and u["disposition"]=="context" for u in source)


def test_source_shapes_and_occurrence_distinctions_are_not_normalized():
    by_key = {r[10]:r for r in rows()}
    assert by_key["nair2013mannan:p47:past-negative:0:w1"][2]=="aa"
    assert by_key["nair2013mannan:p46:other-plural:0:w1"][2]=="kaaṭ-u"
    assert by_key["nair2013mannan:p46:kin-plural:0:w1"][2]=="–laam"
    assert by_key["nair2013mannan:p46:nasal-clusters:1:w1"][2]=="pan̲n̲i"
    assert by_key["nair2013mannan:p47:present:0:w1"][2]=="–nnu"
    assert by_key["nair2013mannan:p45:l-to-c:4:w2"][2]=="kooci"
    assert by_key["nair2013mannan:p48:no-personal-endings:3:w1"][2]=="koocci vantu"
    assert by_key["nair2013mannan:p48:vocabulary:14:w1"][2]=="uḷantaari"
    assert by_key["nair2013mannan:p48:vocabulary:15:w1"][2]=="ulantaaricci"
    assert by_key["nair2013mannan:p48:vocabulary:2:w1"][3]=="calling a bor"
    assert "uncertain" in by_key["nair2013mannan:p48:vocabulary:2:w1"][14].split()


def test_bound_morphology_whole_sentences_and_comparisons():
    by_key = {r[10]:r for r in rows()}
    assert {"affix","suffix","pl"} <= set(by_key["nair2013mannan:p46:kin-plural:0:w1"][14].split())
    assert {"affix","suffix","acc","dat"} <= set(by_key["nair2013mannan:p47:case-suffix:0:w3"][14].split())
    assert {"pron","first-person","pl","exclusive"} <= set(by_key["nair2013mannan:p47:personal-pronouns:2:w1"][14].split())
    assert {"pron","first-person","pl","inclusive"} <= set(by_key["nair2013mannan:p47:personal-pronouns:3:w1"][14].split())
    sentences = [r for r in by_key.values() if "sentential" in r[14].split()]
    assert len(sentences)==5 and all(" " in r[2] for r in sentences)
    assert all(not r[11] and not r[12] and not r[13] for r in by_key.values())
    assert by_key["nair2013mannan:p46:other-plural:0:w1"][9]
    assert by_key["nair2013mannan:p47:case-suffix:0:w1"][0]=="Mannan"
    assert by_key["nair2013mannan:p47:case-suffix:0:w2"][0]=="Malayalam"


def test_actual_parser_nfc_and_nfd_retains_all_fields():
    path = PACKAGE / "validate_actual_parser.py"
    spec = importlib.util.spec_from_file_location("mannan_source_parser_check",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    report = json.loads((PACKAGE / "actual-parser-review.json").read_text())
    assert report["checks"]["NFC"]["all_15_fields_exact"]
    assert report["checks"]["NFD"]["ordered_keys_exact"]


def test_independent_metadata_repair_preserves_literals_and_all_prior_tags():
    old = list(csv.reader((PACKAGE / "revision1/proposed.csv").open(newline="")))
    new = rows()
    assert [r[10] for r in old]==[r[10] for r in new]
    changed = []
    for before,after in zip(old,new):
        assert before[:14]==after[:14]
        assert set(before[14].split()) <= set(after[14].split())
        if before[14]!=after[14]: changed.append(after[10])
    assert len(changed)==22
    for row in new:
        key = row[10]
        if ":nonpast-negative:" in key: assert "non-past" in row[14].split()
        if ":kin-plural:" in key or ":kinship:" in key: assert "kinship" in row[14].split()
        if key=="nair2013mannan:p48:vocabulary:23:w1": assert "verb" in row[14].split()
    continuity = json.loads((PACKAGE / "metadata-repair-continuity.json").read_text())
    assert continuity["independent_decisions"]==19 and continuity["owner_class_extensions"]==3
