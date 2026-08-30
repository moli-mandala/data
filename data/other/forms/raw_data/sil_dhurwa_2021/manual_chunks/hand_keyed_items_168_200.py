#!/usr/bin/env python3
"""Write the OCR-blind manual ledger for Dhurwa Appendix B, physical p. 21.

Every lexical value below was independently keyed by visual inspection of the
600-dpi page render and rechecked against targeted 1200-dpi renders. No PDF
text extraction or OCR value is an input to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_168_200_hand_keyed.tsv"
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
# Slashes are literal printed alternative separators; ordinary spaces remain
# inside a single response. No cell on this page is blank or unresolved.
PAGE_DECISIONS = [
    (168, "clay", ("ʈu:k", "ʈu:k", "ʈu:k", "ʈu:k", "ʈu:k")),
    (169, "soil", ("mañ", "maŋ", "maŋ", "mʌŋ:u", "kuy mʌn")),
    (170, "seed", ("bɪʈiɖ", "vɪʈiɖ", "vɪʈiɖ", "vɪʈiɖ", "vɪʈiɖ")),
    (171, "bark", ("poc:iɖ", "bʌŋdaŋ/ʈʌl", "bʌŋdaŋ", "ʈo:lu", "ʈo:l")),
    (172, "star", ("cuk:a", "cuk:a", "cuk:a", "cuk:a", "cuk:a")),
    (173, "branch", ("cɛl:a", "cɛl:a", "cɛla", "jɛl:a", "jɛl:a")),
    (174, "dew", ("mʌn", "mʌn", "meɳɖir", "mʌɳ", "mañdʒu")),
    (175, "lightning", ("ɖabumo", "midu", "jagurano", "vɪliɖ", "marupkumo")),
    (176, "thunder", ("kuɖriyano", "guduru", "guɖiemo", "guduru", "uɖen puyɪl")),
    (177, "hole", ("palka", "boʈ:a", "boʈ:a", "boʈ:a", "boʈ:a")),
    (178, "pond", ("mu:nɖa", "mu:nɖa", "mu:nɖa", "munɖa", "munɖa")),
    (179, "hair", ("ʈɛlʈa", "ʈɛlʈa", "ʈɛlʈa kull/kadrel", "veɳɖrel", "veɳɖrel")),
    (180, "forehead", ("kʌpar", "kʌpar", "kʌpar", "mɛɖer", "mɛɖek")),
    (181, "tooth", ("pɛl", "pɛl", "pɛl", "pɛl:u", "pɛl")),
    (182, "car", ("keko:l", "keko:l", "keko:l", "keko:l", "keko:l")),
    (183, "boy", ("cepal", "cepal", "padir", "pʌdir", "pʌc:u")),
    (184, "girl", ("mal", "mal", "mal", "mal", "malu")),
    (185, "beard", ("mecel", "gʌd:om", "gʌd:om", "gʌd:om", "gʌd:al")),
    (186, "mustache", ("mecel", "gʌd:om", "mec gʌd:om", "gʌd:om", "gʌd:al")),
    (187, "pig", ("pɛɳɖ", "pɛɳɖu", "pɛɳɖ", "pɛɳɖu", "pɛɳɖu")),
    (188, "feather", ("keɳɖiɖ", "keɳɖiɖ", "keɳɖiɖ", "veɳɖrel", "kaʈuk")),
    (189, "earth", ("nɛɳɖɪl", "nɛɳɖɪl/neli", "nɛɳɖɪl", "nɛɳɖɪl", "nɛɳɖɪl")),
    (190, "butterfly", ("kok:al", "kok:al", "pilpili", "gog:a vala", "gog:a vala")),
    (191, "tiger", ("ɖurki", "ɖurki", "ɖu:v", "ɖu:vu", "ɖu:v")),
    (192, "bear", ("ɪli", "ɪli", "ɪli", "ɪli", "ɪli")),
    (193, "monkey", ("kov:a", "kov:a", "kov:a", "kov:a", "kov:a")),
    (194, "snake", ("bam", "bamb", "bamb", "bamb", "bamb")),
    (195, "worm", ("puɖuʈ", "puɖuʈ", "puɖu:ʈ", "puɖu:ʈ", "puɖuʈ")),
    (196, "mat", ("cʌʈ:a", "cʌʈ:a", "cʌʈ:a", "cʌʈ:a", "cʌʈ:a")),
    (197, "bat", ("vagu:r", "vagu:r", "vagu:r", "vagu:r", "vagu:r")),
    (198, "bird", ("ʈiʈa", "ʈiʈa", "ʈiʈa", "ʈiʈa", "ʈiʈa")),
    (199, "umbrella", ("kɛridʒ", "kɛrid", "kɛrid", "kɛrid", "kɛridʒ")),
    (200, "literate", ("paykel", "poral", "poral", "poral", "porel")),
]


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item, gloss, forms in PAGE_DECISIONS:
        assert len(forms) == len(SITES) == 5
        for (site_code, site_name, column), form in zip(SITES, forms, strict=True):
            row = {
                "Item": str(item),
                "Gloss": gloss,
                "Site_Code": site_code,
                "Site_Name": site_name,
                "PDF_Page": "21",
                "Printed_Page": "16",
                "Column": column,
                "Manual_Transcription": form,
                "Review_Status": "attested",
                "Confidence": "high",
                "Uncertainty": "",
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
    assert len(output_rows) == 165
    assert len({(row["Item"], row["Site_Code"]) for row in output_rows}) == 165
    assert all(row["Review_Status"] == "attested" for row in output_rows)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-keyed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
