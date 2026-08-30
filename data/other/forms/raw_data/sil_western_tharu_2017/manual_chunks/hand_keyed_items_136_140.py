#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 136-140."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_136_140_hand_keyed.tsv")
DECLARATION = "hand-keyed-from-rendered-source; PDF-text-OCR-legacy-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF page; 900/1200-dpi "
    "crops used for every cell; PDF text/OCR/legacy not accepted"
)
SITES = (
    "BNM", "BNT", "RNK", "RNS_Sisaikhara", "RNS_Sisana", "RKM", "RKB",
    "TkN", "KkP", "SkP", "DKS", "DDK", "DGC", "DkR", "CCC", "HIN",
)
SOURCE_CODES = {
    "BNM": "BNM", "BNT": "BNT", "RNK": "RNK", "RNS_Sisaikhara": "RNS",
    "RNS_Sisana": "RNS", "RKM": "RkM", "RKB": "RKB", "TkN": "TkN",
    "KkP": "KkP", "SkP": "SkP", "DKS": "DKS", "DDK": "DDK",
    "DGC": "DGC", "DkR": "DkR", "CCC": "CCC", "HIN": "HIN",
}
OCCURRENCE = {site: "" for site in SITES}
OCCURRENCE.update({"RNS_Sisaikhara": "1", "RNS_Sisana": "2"})
FIELDS = [
    "Item", "Gloss", "Site_Key", "Source_Code", "Source_Code_Occurrence", "Scope",
    "PDF_Page", "Printed_Page", "Column", "Source_Group_Labels",
    "Manual_Transcription", "Manual_Form_Count", "Source_Qualifier", "Review_Status",
    "Confidence", "Site_Assignment_Confidence", "Uncertainty", "Reviewer_Method",
    "Reviewed_At", "Reviewer_Declaration",
]


def cell(form, labels="1", column="left", qualifier=""):
    return form, labels, "56", "51", column, qualifier


# Independently keyed by eye from 900/1200/1600/2400-dpi rendered-page crops
# before any comparison with the legacy CSV.
ITEMS = {
    136: ("hot", {
        "HIN": cell("gʌɾʌm"),
        "DKS": cell("gʰam"),
        "BNT": cell("gʰam"),
        "BNM": cell("gaɾʌm / tʌʈːi", "1 / 2", qualifier="first response: (weather)"),
        "RNS_Sisaikhara": cell("tʌlo", "2"),
        "TkN": cell("tʌtːo", "2"),
        "RNS_Sisana": cell("tʌtːo", "2"),
        "RKM": cell("tʌʈːo", "2"),
        "SkP": cell("tatʌl", "2"),
        "DDK": cell("ʈaʈul", "2"),
        "DkR": cell("tatɾʌl", "2"),
        "RKB": cell("tʌto", "2"),
        "RNK": cell("lʌtːo", "2"),
        "CCC": cell("dʰikʌl", "3"),
        "DGC": cell("dʰikʌl", "3"),
        "KkP": cell("dʰikʌl", "3"),
    }),
    137: ("cold", {
        "HIN": cell("ʈʰʌɳɖa"),
        "BNM": cell("ʈʰʌɳɖa"),
        "RNK": cell("ʈʰʌɳɖo"),
        "RNS_Sisaikhara": cell("ʈʰʌɳɖo"),
        "TkN": cell("ʈʰʌɳɖo"),
        "BNT": cell("ʈʰʌɳɖo"),
        "RNS_Sisana": cell("ʈʰʌɳɖo"),
        "SkP": cell("ʈʰʌɳɖʌ"),
        "CCC": cell("tʌɳɖʰa / dʒaɖ", "1 / 2"),
        "DGC": cell("dʒuɾ", "2"),
        "DkR": cell("dʒuɾ", "2"),
        "DDK": cell("dʒuɾ", "2"),
        "KkP": cell("dʒuɾ", "2"),
        "RKB": cell("dʒudo", "2"),
        "DKS": cell("dʒaɾ", "2"),
        "RKM": cell("dʒuɾo", "2"),
    }),
    138: ("right", {
        "HIN": cell("dahina"),
        "BNM": cell("dahina / kʰana", "1 / 2", column="left / right"),
        "SkP": cell("dahina"),
        "RKB": cell("dahina"),
        "DKS": cell("dahina"),
        "KkP": cell("dahina"),
        "RNK": cell("dahino"),
        "RNS_Sisaikhara": cell("dahino"),
        "DGC": cell("dʌhinʌ", column="right"),
        "DkR": cell("dʌhin", column="right"),
        "TkN": cell("dʌhino", column="right"),
        "RNS_Sisana": cell("dʌhino", column="right"),
        "RKM": cell("dãhino", column="right"),
        "CCC": cell("dahin / dʌhini", "1 / 1", column="right"),
        "DDK": cell("dahija", column="right"),
        "BNT": cell("kʰana", "2", column="right"),
    }),
    139: ("left", {
        "HIN": cell("bãja", column="right"),
        "BNM": cell("bãja", column="right"),
        "DGC": cell("bãja / lebʌɾi", "1 / 3", column="right"),
        "SkP": cell("bãja", column="right"),
        "DkR": cell("bãjo", column="right"),
        "TkN": cell("baja", column="right"),
        "DKS": cell("bawʊ", column="right"),
        "BNT": cell("baj", column="right"),
        "RKM": cell("bao", column="right"),
        "RNS_Sisaikhara": cell("bão / ɖibʌno", "1 / 2", column="right"),
        "RNS_Sisana": cell(None, "", column="right"),
        "CCC": cell("bajaː / lʌdʌɖi", "1 / 3", column="right"),
        "RNK": cell("ɖibʌno", "2", column="right"),
        "RKB": cell("dibʌɾa", "2", column="right"),
        "KkP": cell("dibʌɾa", "2", column="right"),
        "DDK": cell("ɖabɾi / lebʌɾi", "2 / 3", column="right"),
    }),
    140: ("near", {
        "HIN": cell("nʌdʒik", column="right"),
        "CCC": cell("ladʒike", column="right"),
        "RNK": cell("dʒʰɔno", "2", column="right"),
        "RNS_Sisaikhara": cell("dʒʰɔno", "2", column="right"),
        "RKM": cell("dʒʰɔne", "2", column="right"),
        "RNS_Sisana": cell("dʒʰɔne", "2", column="right"),
        "BNM": cell("ʈʰoɾæ", "3", column="right"),
        "SkP": cell("ʈʰɔɽe", "3", column="right"),
        "BNT": cell("ʈʰoɾi", "3", column="right"),
        "DGC": cell("lʌgːe", "4", column="right"),
        "DKS": cell("lʌgːe", "4", column="right"),
        "DDK": cell("lʌgːe / tʰʌn", "4 / 7", column="right"),
        "DkR": cell("lʌgːʌva", "4", column="right"),
        "KkP": cell("ligʰe", "4", column="right"),
        "RKB": cell("ɖʰiŋgai", "5", column="right"),
        "TkN": cell("hin", "6", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(136, 141):
        gloss, cells = ITEMS[item]
        assert set(cells) == set(SITES)
        for site in SITES:
            form, labels, page, printed, column, qualifier = cells[site]
            blank = form is None
            uncertainty = ""
            site_confidence = "high"
            if site.startswith("RNS_"):
                site_confidence = "medium"
                uncertainty = (
                    "duplicate source code RNS; within-group occurrence order assigned to "
                    "metadata row order; unmatched extra group occurrence assigned to "
                    "metadata row 1 (Sisaikhara)"
                )
            if blank:
                uncertainty = (
                    "duplicate source code RNS; only one RNS response per printed group; "
                    "both group responses assigned to metadata row 1, leaving row 2 blank"
                ) if site == "RNS_Sisana" else "site code absent from the complete printed item block"
            rows.append({
                "Item": str(item), "Gloss": gloss, "Site_Key": site,
                "Source_Code": SOURCE_CODES[site],
                "Source_Code_Occurrence": OCCURRENCE[site],
                "Scope": "control" if site == "HIN" else "target",
                "PDF_Page": page, "Printed_Page": printed, "Column": column,
                "Source_Group_Labels": labels, "Manual_Transcription": form or "",
                "Manual_Form_Count": str(len(form.split(" / "))) if form else "0",
                "Source_Qualifier": qualifier,
                "Review_Status": "source_blank" if blank else "attested",
                "Confidence": "high", "Site_Assignment_Confidence": site_confidence,
                "Uncertainty": uncertainty, "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-29", "Reviewer_Declaration": DECLARATION,
            })
    assert len(rows) == 80
    assert sum(row["Review_Status"] == "attested" for row in rows) == 79
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 1
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 88
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 83
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
