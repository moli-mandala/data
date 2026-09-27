"""Check whole responses and identity preservation without building the database."""
import csv
import importlib.util
import io
from pathlib import Path

from segments import Tokenizer

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / "data/other/forms/raw_data/crooke_korwa_1892"
spec = importlib.util.spec_from_file_location("crooke_full", PACKAGE / "prepare_full_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def test_recovered_responses_and_legacy_order():
    rows, audit = source.build()
    old = list(csv.reader((PACKAGE / "20260925-crooke-korwa-mirzapur.csv").open()))
    assert len(rows) == len(audit) == 123
    assert [r[10] for r in rows[:93]] == [r[10] for r in old]
    assert len({r[10] for r in rows}) == 123
    assert len([r for r in audit if r["prior_status"] == "held"]) == 30
    by_gloss = {r[3]: r for r in rows}
    assert by_gloss["daughter"][2] == "kori hopûnu"
    assert by_gloss["the rice is cooking"][2] == "leti dova senidâ"
    assert by_gloss["to be bald"][2] == "koânâ uptido"
    assert by_gloss["to cook"][2] == "badelangi sînmâ"
    assert by_gloss["cheek"][2] == "johâtu"
    assert by_gloss["beard or moustache"][2] == "ḍaṛhît"
    assert by_gloss["boiled rice"][2] == "leṭî"
    assert by_gloss["morning"][2] == "jhâtkarîti"
    assert "uncertain" not in by_gloss["partridge"][14].split()
    assert "uncertain" in by_gloss["wrist"][14].split()
    assert "diacritic" in by_gloss["wrist"][6]
    assert sum("uncertain" in r[14].split() for r in rows) == 1
    assert all(not r[1] and not any(r[8:10]) and not any(r[11:14]) for r in rows)


def test_actual_parser_preserves_whole_forms_and_keys(monkeypatch):
    import make_cldf
    rows, _ = source.build()
    profile = Tokenizer(str(PACKAGE / "full-proposed-profile.txt"))
    monkeypatch.setitem(make_cldf.convertors, "crooke-korwa-1892", profile)
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(
        str(PACKAGE / "full-proposed.csv"), errors, name="20260925-crooke-korwa-mirzapur")
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 123
    for raw, final in zip(rows, parsed):
        assert final.entry_key == raw[10]
        assert final.old_form == raw[2]
        assert final.source == raw[7] and final.notes == raw[6]
        assert final.form.count(" ") == raw[2].count(" ")
        assert "\ufffd" not in final.form
        assert source.DIALECT in final.tags.split()
    assert len({r.entry_key for r in parsed}) == 123


def test_source_comparisons_and_complete_profile_policy():
    import profile_policy
    rows, audit = source.build()
    by_key = {r[10]: r for r in rows}
    assert sum(r["comparison_evidence"] is not None for r in audit) == 41
    tentative = by_key["crooke1892korwa:mirzapur:p126:item04"]
    assert "tentatively" in tentative[6] and "sâmnê" in tentative[6]
    assert "kaṇika" in by_key["crooke1892korwa:mirzapur:p125:item19"][6]
    assert all(not r[12] and not r[9] for r in rows)
    profile = PACKAGE / "full-proposed-profile.txt"
    rules = dict(list(csv.reader(profile.open(), delimiter="\t"))[1:])
    for grapheme, output in rules.items():
        assert profile_policy.house_output(grapheme, output, rules, "crooke-korwa-1892") == output
    tokenizer = Tokenizer(str(profile))
    assert all("\ufffd" not in tokenizer(r[2], column="IPA") for r in rows)
