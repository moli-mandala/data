#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 31-35."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_031_035_hand_keyed.tsv")
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


def cell(form, labels, page=37, printed=32, column="left", qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 1600-dpi crops before legacy comparison.
ITEMS = {
    31: ("mortar", {
        "HIN": cell("okʰli / kʰʌɾʌl", "1 / 7", page="36 / 37", printed="31 / 32", column="right / left"),
        "BNM": cell("okʰʌɾi", "1", page=36, printed=31, column="right"),
        "BNT": cell("okʰʌɾi", "1", page=36, printed=31, column="right"),
        "CCC": cell("okʰʌɾi", "1"),
        "DkR": cell("dokni", "1"),
        "DKS": cell("dukʌni", "1"),
        "DGC": cell("dokʌni", "1", qualifier="(wood)"),
        "DDK": cell("dokʌni / loɖʰa", "1 / 6", qualifier="first response: (wood); second response: (stone)"),
        "KkP": cell("õŋkʰʌɾi", "1"),
        "RNK": cell("pʌtɪja / ɪmandʌsta", "2 / 3"),
        "RNS_Sisaikhara": cell("pʌtɪja", "2"),
        "SkP": cell("pʌtɪja", "2"),
        "RNS_Sisana": cell("pʌtɪja", "2"),
        "TkN": cell("pʌtija", "2"),
        "RKM": cell("patɪja", "2"),
        "RKB": cell("ɖʊkia", "4"),
    }),
    32: ("pestle", {
        "HIN": cell("musʌl", "1"),
        "BNT": cell("musʌl", "1"),
        "BNM": cell("musʌɾ", "1"),
        "RNS_Sisaikhara": cell("musʌɾa / kʊʈʌna", "1 / 2"),
        "RNS_Sisana": cell("mʊsʌla", "1"),
        "CCC": cell("muːsuɾa", "1"),
        "KkP": cell("musaɾa", "1"),
        "RNK": cell("kʊɖʰʌna", "2"),
        "RKM": cell("kʊʈʌna / kʊɽi", "2 / 7"),
        "DkR": cell("pʌtʰʌɾa", "3"),
        "RKB": cell("pʌtʰʌɾ", "3"),
        "DKS": cell("belʌna", "4"),
        "DGC": cell("loɖʌha", "5"),
        "SkP": cell("lʊɖwa", "5"),
        "TkN": cell("mõʈa", "6"),
        "DDK": cell("dhanki", "8"),
    }),
    33: ("hammer", {
        "HIN": cell("hʌtʰoɾi / hʌtʰoɾa", "1 / 1"),
        "RNK": cell("hʌtʰoɽi / hʌtʰoɽija", "1 / 1"),
        "RNS_Sisaikhara": cell("hʌtʰoɽija", "1"),
        "BNM": cell("hʌtʰaɽa", "1"),
        "SkP": cell("hʌtʰɔɽi", "1"),
        "RKB": cell("hʌtʰɔɖi", "1"),
        "TkN": cell("hʌtɔɾɪja", "1"),
        "BNT": cell("hʌtːɔɖa", "1"),
        "RKM": cell("hʌtʰoɾa", "1"),
        "RNS_Sisana": cell("hʌtoɾa", "1"),
        "DGC": cell("dokɪja", "2"),
        "DkR": cell("ʈʰʌkɪja", "2"),
        "DKS": cell("ʈʰokɪja", "2"),
        "DDK": cell("ʈʰokɪja", "2"),
        "KkP": cell("tʰuŋkia", "2", column="right"),
        "CCC": cell(None, "", column="left / right"),
    }),
    34: ("knife", {
        "HIN": cell("tʃaku / tʃʌkːu / tʃaku / tʃʌkːu / tʃʰuɾi", "1 / 1 / 2 / 2 / 4", column="right"),
        "RNS_Sisaikhara": cell("tʃaku / tʃaku / hʌsija", "1 / 2 / 3", column="right"),
        "BNM": cell("tʃuk", "1", column="right"),
        "DGC": cell("tʃʌkːu / tʃʌkːu / hʌsɪja / ɡʰuɾi", "1 / 2 / 3 / 4", column="right"),
        "SkP": cell("tʃʌkːu / tʃʌkːu", "1 / 2", column="right"),
        "TkN": cell("tʃʌkːu / tʃʌkːu", "1 / 2", column="right"),
        "RNS_Sisana": cell("tʃʌkːu / tʃʌkːu / hʌsija", "1 / 2 / 3", column="right"),
        "DKS": cell("tʃʌkː / hʌsɪja", "1 / 3", column="right"),
        "BNT": cell("tʃʌku / tʃʌku", "1 / 2", column="right"),
        "KkP": cell("tʃʰʌkʰua", "1", column="right"),
        "CCC": cell("tʃʰuɾi / tʃʰuɾi", "2 / 4", column="right"),
        "RNK": cell("hʌsija", "3", column="right"),
        "DkR": cell("hʌsija", "3", column="right"),
        "RKM": cell("hʌsija", "3", column="right"),
        "DDK": cell("hʌsija / tʃʰuɾija", "3 / 4", column="right"),
        "RKB": cell("hʌsɪja", "3", column="right"),
    }),
    35: ("axe", {
        "HIN": cell("kulhaɖi", "1", column="right"),
        "RNK": cell("kʊɽhaɾi / tukla", "1 / 2", column="right", qualifier="second response: (small)"),
        "RNS_Sisaikhara": cell("kʊɽhaɾi", "1", column="right"),
        "SkP": cell("kʊɽhaɾi", "1", column="right"),
        "BNM": cell("kuhaɖi", "1", column="right"),
        "DkR": cell("kuɾhaɾ", "1", column="right"),
        "DDK": cell("kuɾhaɾ", "1", column="right"),
        "RKB": cell("kʊdahari", "1", column="right"),
        "TkN": cell("kʊɽʌhʌɾija", "1", column="right"),
        "DKS": cell("kʊɾʌhaɾ", "1", column="right"),
        "BNT": cell("kohaɖi", "1", column="right"),
        "RKM": cell("kʊɽʌtʃʰʌɾi", "1", column="right"),
        "RNS_Sisana": cell("kʊɽʌhʌni", "1", column="right"),
        "KkP": cell("kohaɾi", "1", column="right"),
        "DGC": cell("bʌntʃeɾi / tegaɾi", "3 / 5", page="37 / 38", printed="32 / 33", column="right / left", qualifier="first response: (small); second response: (big)"),
        "CCC": cell("taŋi", "4", page=38, printed=33, column="left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(31, 36):
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
                uncertainty = "site code absent from the complete printed item block"
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 105
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 94
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
