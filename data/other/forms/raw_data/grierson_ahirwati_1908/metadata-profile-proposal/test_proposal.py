"""Tests of preliminary configuration, not acceptance of source readings."""
import unicodedata as ud
from segments.tokenizer import Tokenizer
import prepare_proposal as proposal
import validate_proposal


def conversion(value):
    tokenizer = Tokenizer(str(proposal.HERE / f"{proposal.PROFILE}.txt"))
    return ud.normalize("NFC", tokenizer(value, column="IPA").replace(" ", "").replace("#", " "))


def test_complete_pinned_corpus_actual_metadata_and_formatter():
    validate_proposal.validate()


def test_difficult_source_marks_preserved_independently_of_case():
    for original, expected in [
        ("Ṅ Ṅ ṅ ṅ", "ŋ ŋ ŋ ŋ"),
        ("Ñ ñ", "ñ ñ"),
        ("Ā̃ Ī̃ ā̃ ī̃", "ā̃ ī̃ ā̃ ī̃"),
        ("karᵃṇō apᵃṇū aĩh", "karᵃṇō apᵃṇū aĩh"),
        ("-ā ā- –ā", "-ā ā- –ā"),
        ("kai-ṭaraḥ", "kai-ṭaraḥ"),
        ("‘Bōlō,’ (ē) ?", "‘bōlō,’ (ē) ?"),
        ("Jai-kaḷi- kai", "jai-kaḷi- kai"),
        ("पीलने बाँट-दियो", "पीलने बाँट-दियो"),
        ("ایک بیرےنے", "ایک بیرےنے"),
    ]:
        for normalization in ("NFC", "NFD"):
            assert conversion(ud.normalize(normalization, original)) == expected


def test_exact_original_inventory_no_silent_display_rewrite():
    corpus = list(proposal.records(proposal.HERE / "coverage-corpus.jsonl"))
    assert len(corpus) == 736
    assert len({r["key"] for r in corpus}) == 736
    assert sum(r["category"] == "no37" for r in corpus) == 390
    assert sum(r["category"] == "outside" for r in corpus) == 62
    for row in corpus:
        assert row["expected_display"] == proposal.display(row["original"])
    assert any(r["original"] == "Jai-kaḷi-" for r in corpus)
    assert any("ḥ" in r["original"] for r in corpus)
