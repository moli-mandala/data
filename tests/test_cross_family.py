import csv
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from data.cross_family import build, classify_claim


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "data"
CLDF = ROOT / "cldf"


def dict_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_claim_classifier_preserves_source_direction_and_hedging():
    assert classify_claim(
        "Probably < IA; Turner CDIAL no. 1347.", "dedr"
    ) == ("loan", "entry-from-compared", "medium")
    assert classify_claim(
        "The direction of borrowing is uncertain; Turner CDIAL no. 955.", "dedr"
    ) == ("loan", "undetermined", "low")
    assert classify_claim(
        "Tel. tokka, but DED 2937 Drav. rather ← Sk. tvák.", "cdial"
    ) == ("loan", "compared-from-entry", "medium")
    assert classify_claim(
        "Poss. ← or infl. by Drav. Prj. muṭka, DED 4041.", "cdial"
    ) == ("influence", "entry-from-compared", "medium")
    assert classify_claim(
        "Br. sil rind (prob. < Panj. chill; Turner CDIAL no. 5052).", "dedr"
    ) == ("loan", "entry-from-compared", "medium")
    assert classify_claim(
        "The IA words are probably < Dr.; Turner CDIAL no. 14174.", "dedr"
    ) == ("loan", "compared-from-entry", "medium")


def test_full_cached_extraction_is_reproducible_and_audited():
    comparisons, audits = build()
    installed = dict_rows(SOURCE / "cross-family-comparisons.csv")
    checked_audit = dict_rows(SOURCE / "cross-family-comparisons-audit.csv")

    assert [asdict(row) for row in comparisons] == installed
    assert [asdict(row) for row in audits] == checked_audit
    assert len(comparisons) == 604
    assert Counter((row.Source_Dictionary, row.Status) for row in audits) == {
        ("cdial", "installed"): 122,
        ("cdial", "unresolved"): 428,
        ("dedr", "installed"): 475,
        ("dedr", "unresolved"): 486,
        ("dedr", "excluded"): 10,
        ("southworth", "installed"): 7,
    }


def test_ambiguous_legacy_ded_numbers_have_reviewed_overrides():
    rows = {row["ID"]: row for row in dict_rows(SOURCE / "cross-family-comparisons.csv")}
    assert rows["cdial:4477:dedr:d1633"]["Compared_Entry_ID"] == "d1633"
    assert rows["cdial:10146:dedr:d4893"]["Compared_Entry_ID"] == "d4893"
    assert rows["cdial:10970:dedr:d3705"]["Compared_Entry_ID"] == "d3705"


def test_dedr_comparison_sources_include_printed_old_edition_locators():
    rows = {row["ID"]: row for row in dict_rows(SOURCE / "cross-family-comparisons.csv")}
    assert rows["dedr:d64:cdial:1"]["Source"] == "dedr[entry 64, DED 57]"


def test_southworth_supplies_only_printed_cross_table_pairs():
    rows = [
        row for row in dict_rows(SOURCE / "cross-family-comparisons.csv")
        if row["Source"].startswith("southworth2005m[")
    ]
    assert len(rows) == 7
    assert {
        (row["Entry_ID"], row["Compared_Entry_ID"])
        for row in rows
    } == {
        ("d1494", "3083"),
        ("d4004", "9051"),
        ("d449", "5539"),
        ("d364", "997"),
        ("d1109", "2639"),
        ("d4673", "9734"),
        ("d1651", "5566"),
    }
    assert {row["Relation"] for row in rows} == {"loan"}
    assert {row["Direction"] for row in rows} == {"compared-from-entry"}
    assert {
        row["Confidence"] for row in rows if row["Entry_ID"] == "d4004"
    } == {"low"}
    assert {
        row["Confidence"] for row in rows if row["Entry_ID"] != "d4004"
    } == {"high"}
    assert all(
        "Table 1 row" in row["Evidence"] and "Table 2" in row["Evidence"]
        for row in rows
    )

    audits = [
        row for row in dict_rows(SOURCE / "cross-family-comparisons-audit.csv")
        if row["Source_Dictionary"] == "southworth"
    ]
    assert len(audits) == 7
    assert {row["Status"] for row in audits} == {"installed"}
    assert {row["Resolution"] for row in audits} == {
        "source-image-verified-cross-table-pair"
    }


def test_cross_family_forms_are_not_emitted_as_ordinary_reflexes():
    with (SOURCE / "dedr" / "dedr_new.csv").open(encoding="utf-8", newline="") as handle:
        dedr_rows = list(csv.reader(handle))
    assert not any(row[0] == "OIA" for row in dedr_rows)
    assert not any(
        re.search(r"(?:\bCDIAL\b|cf\. Turner|; Turner)", " ".join(row[:4]), re.I)
        for row in dedr_rows
    )
    assert not any(row[1] == "d104" and row[2] == "addha-" for row in dedr_rows)
    assert not any(
        row[1] == "d1110" and row[2] in {"kaḍayaḍ-", "kaṭkaṭī"}
        for row in dedr_rows
    )
    assert any(
        row[0] == "Kurux" and row[1] == "d50" and row[2] == "ajjī"
        and row[3] == "grandmother"
        for row in dedr_rows
    )
    assert any(row[0] == "Kannada" and row[1] == "d1110" for row in dedr_rows)
    assert {
        row[2] for row in dedr_rows if row[0] == "Kota" and row[1] == "d1767"
    } == {"ku·g im", "ku·g et"}

    dravidian = {"Brah", "Drav", "Ga", "Go", "Kan", "Kol", "Kur", "Mal", "Nk", "Prj", "Tam", "Tel", "Tu"}
    with (SOURCE / "cdial" / "cdial.csv").open(encoding="utf-8", newline="") as handle:
        assert not any(row[0] in dravidian for row in csv.reader(handle))


def test_review_sample_is_fixed_and_signed_off():
    sample = dict_rows(SOURCE / "cross-family-comparisons-sample.csv")
    assert len(sample) == 20
    assert {row["Review"] for row in sample} == {"ok"}


def test_compiled_comparison_sidecar_has_cross_family_entries_and_valid_references():
    comparisons = dict_rows(CLDF / "comparisons.csv")
    extracted = dict_rows(SOURCE / "cross-family-comparisons.csv")
    manual = dict_rows(SOURCE / "manual-cross-family-comparisons.csv")
    dbia = dict_rows(SOURCE / "dbia" / "comparisons.csv")
    forms = {row["ID"]: row for row in dict_rows(CLDF / "forms.csv")}
    references = {row["ID"] for row in dict_rows(CLDF / "references.csv")}

    assert len(comparisons) > 604
    assert sum(not row["ID"].startswith("burushaski:") for row in comparisons) == (
        len(extracted) + len(manual) + len(dbia)
    )
    assert len(manual) == 91
    assert {row["ID"] for row in manual} <= {row["ID"] for row in comparisons}
    assert len(dbia) == 328
    assert {row["ID"] for row in dbia} <= {row["ID"] for row in comparisons}
    assert len({row["ID"] for row in comparisons}) == len(comparisons)
    for row in comparisons:
        left = forms[row["Entry_ID"]]
        right = forms[row["Compared_Entry_ID"]]
        assert {left["Language_ID"], right["Language_ID"]} in (
            {"PDr", "Indo-Aryan"},
            {"PBr", "Indo-Aryan"},
        )
        assert row["Source"].split("[", 1)[0] in references
        assert row["Evidence"].strip()
