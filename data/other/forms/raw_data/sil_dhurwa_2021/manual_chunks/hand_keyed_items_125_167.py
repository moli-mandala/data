#!/usr/bin/env python3
"""Write the OCR-blind manual ledger for Dhurwa Appendix B, physical p. 20.

Every lexical value below was independently keyed by visual inspection of the
600-dpi page render and rechecked against targeted 1200-dpi renders. No PDF
text extraction or OCR value is an input to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_125_167_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Review_Status", "Confidence",
    "Uncertainty", "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]
SITES = [
    ("TIR", "Tiriya", "1"),
    ("NET", "Nethanar", "2"),
    ("DHA", "Dharba", "3"),
    ("KUK", "Kukanar", "4"),
    ("U5", "Unlabeled fifth printed column", "5"),
]

# Explicit hand-entered page decisions, in printed row and column order.
# None denotes a source cell printed as a double hyphen. A slash remains the
# literal printed alternative separator; ordinary spaces stay within a form.
PAGE_DECISIONS = [
    (125, "month", ("nɛlɪŋ", "nɛliŋ", "nɛlɪŋ", "nɛlɪŋ ba:r", "nɛliñ")),
    (126, "year", ("bɛrcikar", "vɛrcikar", "vɛrcikar", "bɛrcikar", "bɛrcikar")),
    (127, "cold season", ("pʌñyil", "pʌñyil", "pʌñyil", "pʌñyil", "pʌñdʒil")),
    (128, "warm season", ("dʒeʈa", "jeta", "jeta", "jeta", "dʒeʈa/neɳɖ")),
    (129, "rainy season", ("borca", "borca", "bʌrca", "bʌrca", "vañdʒi")),
    (130, "good", ("rec:a", "rec:a", "rec:a", "nɪ:a", "ʌc:al")),
    (131, "bad", ("kiyal", "kiyal", "kiyal", "ʌr:a", "kareŋ")),
    (132, "wet", ("po:ʈi", "po:ɖurano", "po:ɖurano", "po:yʌʈʌ", "po:yʌʈa")),
    (133, "dry", ("veʈu", "veʈrano", "veʈʌraro", "veʈʌʈʌ", "elaʈ:ʌ")),
    (134, "long", ("laʈi", "laʈi", "laʈi", "laʈi", "laʈi")),
    (135, "short", ("moɳɖi", "moɳɖi", "moɳɖi", "moɳɖi", "moɳɖi")),
    (136, "hot", ("ʈʌɖu", "ʈʌɖu", "ʈʌrmo", "ʈʌɖiyʌ", "ʈʌɖiyʌ")),
    (137, "cold", ("elu", "elu", "eɖano", "ɛliyʌ", "ɛlʌʈ:ʌ")),
    (138, "right", ("vela", "vela", "vela", "ʈinʈa kɛy", "ʈinʈakɛy")),
    (139, "left", ("ɖɛbra", "ɖɛbra", "ɖɛbri", "rʌɖʌ kɛy", "ɖɛbra")),
    (140, "near", ("lɛg:e", "lɛg:e", "lɛg:e", "lɛg:ʌ", "lʌk:nɖi")),
    (141, "far", ("ko:maɖ", "ko:maɖ", "ko:maɖ", "lapi", "komaɖ")),
    (142, "big", ("bɛɖʈo", "bɛrʈo", "bɛrʈuʈ", "bɛrʈo", "bɛrʈo")),
    (143, "small", ("pɪʈiʈ", "pɪʈiʈo", "pɪʈiʈʌ", "pɪʈiʈ", "pɪʈiʈʌ")),
    (144, "above", ("puɖi", "poɖi", "poɖi", "poɖɪ", "poɖi")),
    (145, "below", ("kiɖi", "kiɖi", "kiɖi", "kidi", "kiɖi")),
    (146, "white", ("bɪl:oʈ", "vɪl", "vɪl", "vɪl:ʌʈ", "vɪliyaʈ")),
    (147, "black", ("mañdʒoʈ", "mañji", "koyli", "koyle", "cu:ɖʌʈʌ")),
    (148, "red", ("nɛʈ raʈ", "nɛʈraʈ", "rʌgrʌgaʈ", "sɛŋg", "lal")),
    (149, "green", ("pʌyo", "pʌyo", None, "nili", "nili")),
    (150, "blue", ("nili", "nili", "nili", "nili", "nili")),
    (151, "one", ("ok:uʈ", "ok:uʈ", "ok:uʈ", "ok:uʈ", "ok:uʈ")),
    (152, "two", ("ɪrɛɖuk", "ɪrɖu:k", "ɪrɖu:k", "urɖu:k", "urɖu:k")),
    (153, "three", ("munɖuk", "mu:nɖu:k", "mu:ɳɖu:k", "mu:ɳɖu:k", "mu:ɳɖu:k")),
    (154, "four", ("naluk", "naluk", "naluk", "nalu:k", "nalu:k")),
    (155, "five", ("cenɖuk", "cenɖu:k", "ceɳɖu:k", "ceɳɖu:k", "ceɳɖu:k")),
    (156, "six", ("coy kota", "coy kota", "cokota", "coy ʈan", "coy")),
    (157, "seven", ("caʈ kota", "caʈ kota", "caʈ kota", "caʈ ʈan", "caʈ")),
    (158, "eight", ("aʈ kota", "aʈ kota", "aʈ kota", "aʈ ʈan", "aʈ")),
    (159, "nine", ("nov kota", "no kota", "nov kota", "no: ʈan", "nov")),
    (160, "ten", ("ɖec kota", "ɖec kota", "ɖʌc kota", "ɖʌs ʈan", "ɖec")),
    (161, "twenty", ("koɖek", "koɖek", "koɖek", "hie", "koɖek")),
    (162, "half", ("aɖa", "aɖa", "aɖa", None, None)),
    (163, "hundred", ("cenɖ ko:l", "cenɖ ko:ɖek", "ceɳɖ ko:ɖu", "cʌv", "cenɖu ko:l")),
    (164, "yes", ("o:", "o:", "o:", "o:", "o:")),
    (165, "no", ("cɪl:a", "cɪl:a", "cɪl:a/ɛra", "cɪl:a/ɛra", "cɪlɛgaʈ")),
    (166, "same", ("okʈi", "okʈi", "okʈi", "okʈi", "okʈi")),
    (167, "different", ("binɛy", "avur", "ɛleg ɛleg", "ɛlge ɛlge", "bɪlag bɪlag")),
]


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item, gloss, forms in PAGE_DECISIONS:
        assert len(forms) == len(SITES) == 5
        for (site_code, site_name, column), form in zip(SITES, forms, strict=True):
            is_blank = form is None
            row = {
                "Item": str(item),
                "Gloss": gloss,
                "Site_Code": site_code,
                "Site_Name": site_name,
                "PDF_Page": "20",
                "Printed_Page": "15",
                "Column": column,
                "Manual_Transcription": "" if is_blank else form,
                "Review_Status": "source_blank" if is_blank else "attested",
                "Confidence": "high",
                "Uncertainty": "source prints double hyphen" if is_blank else "",
                "Reviewer_Method": (
                    "manual visual inspection of 600-dpi rendered page; "
                    "difficult glyphs rechecked at 1200 dpi"
                ),
                "Reviewed_At": "2026-08-28",
                "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            out.append(row)
    return out


def main() -> None:
    output_rows = rows()
    assert len(output_rows) == 215
    assert len({(row["Item"], row["Site_Code"]) for row in output_rows}) == 215
    assert sum(row["Review_Status"] == "source_blank" for row in output_rows) == 3
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-keyed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
