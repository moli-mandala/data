import json
from pathlib import Path
import unicodedata as ud

from segments.tokenizer import Tokenizer
import prepare_fixture as fixture

HERE = Path(__file__).resolve().parent


def convert(tokenizer, form):
    return ud.normalize("NFC", tokenizer(form, column="IPA").replace(" ", "").replace("#", " "))


def test_every_available_form_nfc_and_nfd():
    tokenizer = Tokenizer(str(HERE / "grierson-kolami-basim-1906-preliminary.txt"))
    with (HERE / "coverage-corpus.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            for normalization in ("NFC", "NFD"):
                actual = convert(tokenizer, ud.normalize(normalization, row["original"]))
                assert actual == row["expected_display"], (row["key"], normalization, row["original"], actual)
                assert "�" not in actual


def test_only_authorized_display_changes():
    tokenizer = Tokenizer(str(HERE / "grierson-kolami-basim-1906-preliminary.txt"))
    for source, expected in [("Ṅ Ṅ ṅ ṅ", "ŋ ŋ ŋ ŋ"), ("Ñ ñ", "ñ ñ"), ("A ū", "a ū"), ("CH W ch w", "ch w ch w"), ("annu(ṅg)", "annu(ŋg)"), ("Bāp- phōkān", "bāp- phōkān"), ("āṇḍā(t)", "āṇḍā(t)"), ("A,B;C.D? ‘E’ (F)!", "a,b;c.d? ‘e’ (f)!")]:
        assert convert(tokenizer, source) == expected


def test_pinned_originals_and_coverage():
    status = json.loads((HERE / "fixture-status.json").read_text())
    assert status["counts"] == {"specimen_physical_atom":561, "grammar_child":224, "outside_candidate":5, "outside_alternative":1}
    for path, digest in status["input_hashes"].items():
        assert fixture.sha(fixture.ROOT / path) == digest
    for name, digest in status["output_hashes"].items():
        assert fixture.sha(HERE / name) == digest
    corpus = {}
    with (HERE / "coverage-corpus.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row["key"] not in corpus
            corpus[row["key"]] = row
    originals = {}
    for page in (564, 565, 568, 569):
        with (fixture.PACKAGE / "specimens" / f"p{page}-first-reading.jsonl").open() as stream:
            for line in stream:
                for atom in json.loads(line)["target_atoms"]:
                    originals[atom["source_unit_key"]] = atom["raw_source_form"]
    with (fixture.PACKAGE / "independent-grammar-review/independent-child-witnesses.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            originals[row["witness_key"]] = row["form"]
    outside = json.loads((fixture.PACKAGE / "outside-review/scope-decisions.json").read_text())
    for row in outside["physical_lexical_witnesses_outside_chapter"]:
        originals[row["key"]] = row["form_candidate"]
    assert len(originals) == 790
    assert {key: row["original"] for key, row in corpus.items() if row["category"] in {"specimen_physical_atom", "grammar_child", "outside_candidate"}} == originals
    assert status["canonical_changes"] is False
