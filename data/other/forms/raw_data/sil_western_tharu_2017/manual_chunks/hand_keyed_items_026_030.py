#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 26-30."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_026_030_hand_keyed.tsv")
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


def cell(form, labels, page=36, printed=31, column="left", qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 1600-dpi crops before legacy comparison.
ITEMS = {
    26: ("house", {
        "HIN": cell("ɡʰʌɾ", "1", page=35, printed=30, column="right"),
        "RNK": cell("ɡʰʌɾ / tʃʰala", "1 / 2", page="35 / 36", printed="30 / 31", column="right / left"),
        "RNS_Sisaikhara": cell("ɡʰʌɾ", "1", page=35, printed=30, column="right"),
        "RNS_Sisana": cell("ɡʰʌɾ", "1"),
        "BNM": cell("ɡʰʌɾ / mʌkan", "1 / 3", qualifier="second response: (pukka house)"),
        "DGC": cell("ɡʰʌɾ", "1"),
        "DkR": cell("ɡʰʌɾ", "1"),
        "SkP": cell("ɡʰʌɾ", "1"),
        "TkN": cell("ɡʰʌɾ", "1"),
        "DKS": cell("ɡʰʌɾ / mʌkan", "1 / 3", qualifier="second response: (pukka house)"),
        "BNT": cell("ɡʰʌɾ / mʌkan", "1 / 3", qualifier="second response: (pukka house)"),
        "RKM": cell("ɡʰʌɾ", "1"),
        "DDK": cell("ɡʰʌɾ", "1"),
        "RKB": cell("ɡʰʌɾ / mʌkʌn", "1 / 3"),
        "CCC": cell("ɡʰaɾ", "1"),
        "KkP": cell("ɡʰaɾe", "1"),
    }),
    27: ("roof", {
        "HIN": cell("tʃʰʌt", "1"),
        "DKS": cell("tʃʰʌt / tʃʰʌpʌɾa", "1 / 2"),
        "DGC": cell("tʃʰʌt / tʃʰʌpʌɽa", "1 / 2"),
        "RNK": cell("tʃʌpːʌɾ", "2"),
        "RNS_Sisaikhara": cell("tʃʌpːʌɾ / lintʌɾ", "2 / 3"),
        "DDK": cell("tʃʌpːʌɾ / pʌʈʌŋ", "2 / 5"),
        "DkR": cell("tʃʰʌpːʌɾa", "2"),
        "BNM": cell("tʃʰʌpːʌɾ / lendʌɾ", "2 / 3", qualifier="second response: (pukka roof)"),
        "RKM": cell("tʃʰʌpːʌɾ", "2"),
        "RNS_Sisana": cell("tʃʰʌpːʌɾ", "2"),
        "RKB": cell("tʃʰʌpːʌɾ / tʃani", "2 / 4"),
        "SkP": cell("tʃʰʌpʌɾa", "2"),
        "TkN": cell("tʃʌpːʌɾa", "2"),
        "BNT": cell("tʃʰʌpʌɾ / lʌɳɖʌɾ", "2 / 3"),
        "KkP": cell("tʃʌpaɾa", "2"),
        "CCC": cell("tʃʌnhi", "4"),
    }),
    28: ("door", {
        "HIN": cell("dʌɾvaza / pʰaʈʌk", "1 / 2"),
        "RNK": cell("pʰaʈʌk", "2"),
        "RNS_Sisaikhara": cell("pʰaʈʌk", "2"),
        "RKB": cell("pʰaʈʌk", "2"),
        "RKM": cell("pʰaʈʌk", "2"),
        "RNS_Sisana": cell("pʰaʈʌk", "2"),
        "SkP": cell("kaʈʌk", "2"),
        "BNM": cell("kowaɾ / moɖe", "3 / 4", column="left / right"),
        "TkN": cell("kɪwaɽ", "3", column="right"),
        "BNT": cell("kɪwaɖ", "3", column="right"),
        "CCC": cell("keːwaɾi", "3", column="right"),
        "KkP": cell("kibaɾa", "3", column="right"),
        "DkR": cell("duwaɾ", "5", column="right"),
        "DKS": cell("duwaɾ", "5", column="right"),
        "DGC": cell("duwaɾ", "5", column="right"),
        "DDK": cell("dʌwaɾ", "5", column="right"),
    }),
    29: ("firewood", {
        "HIN": cell("lʌkʌɽi / ĩdʰʌn", "1 / 3", column="right"),
        "BNM": cell("lʌkʌɽi", "1", column="right"),
        "BNT": cell("lʌkʌɽi", "1", column="right"),
        "RNK": cell("kaɖʰɪja", "2", column="right"),
        "RNS_Sisaikhara": cell("kaɖʰɪja", "2", column="right"),
        "DkR": cell("kʌʈʰua", "2", column="right"),
        "SkP": cell("kaʈʰi", "2", column="right"),
        "DKS": cell("kaʈʰi", "2", column="right"),
        "DDK": cell("kaʈʰi", "2", column="right"),
        "CCC": cell("katʰ", "2", column="right", qualifier="(wood)"),
        "RKB": cell("kʌʈɪja", "2", column="right"),
        "TkN": cell("kʌtʰɪja", "2", column="right"),
        "RNS_Sisana": cell("kʌtʰɪja", "2", column="right"),
        "RKM": cell("kaʈʰɪja", "2", column="right"),
        "DGC": cell("kaʈʰɪ", "2", column="right"),
        "KkP": cell("kaʈʰa", "2", column="right"),
    }),
    30: ("broom", {
        "HIN": cell("dʒʰaɽu", "1", column="right"),
        "RNK": cell("bʌɖnɪ", "2", column="right"),
        "RKB": cell("bʌɖni / kʰʌɾajo", "2 / 3", column="right", qualifier="first response: (small); second response: (big)"),
        "RNS_Sisaikhara": cell("bʌɽhʌni", "2", column="right"),
        "RKM": cell("bʌɽhʌni", "2", column="right"),
        "RNS_Sisana": cell("bʌɽhʌni", "2", column="right"),
        "DDK": cell("bʌɽhʌni", "2", column="right"),
        "BNT": cell("bʌnni", "2", column="right"),
        "BNM": cell("bʌni", "2", column="right"),
        "DGC": cell("bʌɾʌni", "2", column="right"),
        "DKS": cell("bʌɾʌni / sɪʈa", "2 / 4", column="right", qualifier="second response: (for outside)"),
        "DkR": cell("bʌɖʌhʌni", "2", column="right"),
        "SkP": cell("baɽhʌni", "2", column="right"),
        "KkP": cell("baɽhʌni", "2", column="right"),
        "TkN": cell("bʌɖʰʌhʌnɪ", "2", column="right"),
        "CCC": cell("bʌɾhʌni / kutʃo", "2 / 5", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(26, 31):
        gloss, cells = ITEMS[item]
        assert set(cells) == set(SITES)
        for site in SITES:
            form, labels, page, printed, column, qualifier = cells[site]
            uncertainty = ""
            site_confidence = "high"
            if site.startswith("RNS_"):
                site_confidence = "medium"
                uncertainty = (
                    "duplicate source code RNS; within-group occurrence order assigned to "
                    "metadata row order; unmatched extra group occurrence assigned to "
                    "metadata row 1 (Sisaikhara)"
                )
            rows.append({
                "Item": str(item), "Gloss": gloss, "Site_Key": site,
                "Source_Code": SOURCE_CODES[site],
                "Source_Code_Occurrence": OCCURRENCE[site],
                "Scope": "control" if site == "HIN" else "target",
                "PDF_Page": page, "Printed_Page": printed, "Column": column,
                "Source_Group_Labels": labels, "Manual_Transcription": form,
                "Manual_Form_Count": str(len(form.split(" / "))),
                "Source_Qualifier": qualifier, "Review_Status": "attested",
                "Confidence": "high", "Site_Assignment_Confidence": site_confidence,
                "Uncertainty": uncertainty, "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-29", "Reviewer_Declaration": DECLARATION,
            })
    assert len(rows) == 80
    assert all(row["Review_Status"] == "attested" for row in rows)
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 98
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 91
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
