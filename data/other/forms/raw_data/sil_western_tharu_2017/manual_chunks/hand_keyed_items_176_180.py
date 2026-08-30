#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 176-180."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_176_180_hand_keyed.tsv")
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


def cell(form, labels="1", page="63", printed="58", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/2400-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    176: ("different", {
        "HIN": cell("ʌlʌgʌlʌg / fʌɾʌk / bʰinː", labels="1 / 6 / 7", page="62", printed="57", column="right"),
        "BNM": cell("ʌlʌgʌlʌg", page="62", printed="57", column="right"),
        "DkR": cell("ʌlʌgʌlʌg", page="62", printed="57", column="right"),
        "BNT": cell("ʌlʌgʌlʌg", page="62", printed="57", column="right"),
        "TkN": cell("ʌlʌgælʌg", page="62", printed="57", column="right"),
        "SkP": cell("ʌlʌgʌlʌgoʈ", page="62", printed="57", column="right"),
        "RKM": cell("ʌlʌgæʌlʌgæ", page="62", printed="57", column="right"),
        "DDK": cell("ʌligẽʌligẽ", page="62", printed="57", column="right"),
        "DGC": cell("ʌlʌgeʌlʌge / dusʌɾdusʌɾ", labels="1 / 5", page="62", printed="57", column="right"),
        "RKB": cell("ʌlʌgoʈ / ɔɾeɔɾe", labels="2 / 3", page="62", printed="57", column="right"),
        "RNK": cell("ɔɾeɔɾe", labels="3", page="62", printed="57", column="right"),
        "RNS_Sisaikhara": cell("ɔɾeɔɾe", labels="3", page="62", printed="57", column="right"),
        "RNS_Sisana": cell("ɔɾeɔɾe", labels="3", page="62", printed="57", column="right"),
        "DKS": cell("ɔɾeɔɾe", labels="3", page="62", printed="57", column="right"),
        "KkP": cell("ɔɾeʈaɾɔɾeʈaɾ / ɔɾeʈʰaɾɔɾeʈʰaɾ", labels="3 / 8", page="62", printed="57", column="right"),
        "CCC": cell("weera", labels="4", page="62", printed="57", column="right"),
    }),
    177: ("whole", {
        "HIN": cell("pʊɾa", page="62", printed="57", column="right"),
        "RNK": cell("pʊɾa", page="62", printed="57", column="right"),
        "BNM": cell("pʊɾa", page="62", printed="57", column="right"),
        "SkP": cell("pʊɾa", page="62", printed="57", column="right"),
        "DGC": cell("pʊɾa", page="62", printed="57", column="right"),
        "DKS": cell("pʊɾa / sʌkːu", labels="1 / 2", page="62 / 63", printed="57 / 58", column="right / left"),
        "RNS_Sisaikhara": cell("puɾo", page="62", printed="57", column="right"),
        "RKB": cell("puɾo / sʌb", labels="1 / 3", page="62 / 63", printed="57 / 58", column="right / left"),
        "BNT": cell("pʊɾa", page="62", printed="57", column="right"),
        "DkR": cell("sʌbku", labels="2"),
        "DDK": cell("sʌkːu", labels="2"),
        "TkN": cell("sʌb", labels="3"),
        "RKM": cell("sʌb", labels="3"),
        "RNS_Sisana": cell("sʌb", labels="3"),
        "CCC": cell("ond̪i", labels="4"),
        "KkP": cell("dʒʌmma", labels="5"),
    }),
    178: ("broken", {
        "HIN": cell("tʊʈa / ʌdʰura", labels="1 / 4"),
        "BNM": cell("tʊʈa"),
        "RNK": cell("tʊʈo"),
        "RNS_Sisaikhara": cell("tʊʈo"),
        "RNS_Sisana": cell("tʊʈo"),
        "DkR": cell("tʊʈʌl"),
        "SkP": cell("tʊʈʌl"),
        "KkP": cell("tʊʈʌl"),
        "RKB": cell("dʰuʈo"),
        "DGC": cell("tuʈʌla"),
        "BNT": cell("pʰuʈa"),
        "DDK": cell("pʰuʈʌlw̃"),
        "DKS": cell("tutgaijja"),
        "RKM": cell("dukʌra", labels="2"),
        "TkN": cell("duʈi", labels="3"),
        "CCC": cell(None),
    }),
    179: ("few", {
        "HIN": cell("kʊtʃʰ / tʰoɾi / t̪ʰoɾa / kʌm", labels="1 / 2 / 2 / 4"),
        "BNT": cell("kʊtʃʰ"),
        "BNM": cell("kʊtʃʰ / tʰoɾi", labels="1 / 2"),
        "RNK": cell("tʰoɾi", labels="2"),
        "RKB": cell("tʰoɾi / nikʌna", labels="2 / 6"),
        "DGC": cell("tʰoɾiek / tʰɔɾewun", labels="2 / 7"),
        "KkP": cell("tʰoɾiek", labels="2"),
        "RKM": cell("tʰoɾi / dʒʌɾakenahæ̃", labels="2 / 8"),
        "RNS_Sisaikhara": cell("dʒʌɾase", labels="3"),
        "DkR": cell("kʌm", labels="4"),
        "TkN": cell("kʌm", labels="4"),
        "DDK": cell("kʌm", labels="4"),
        "DKS": cell("kʌm / tʰoɖtʃʊn", labels="4 / 7"),
        "SkP": cell("tʌndjaka", labels="5"),
        "RNS_Sisana": cell("dʒʌɾʌjegʰaj", labels="9"),
        "CCC": cell(None),
    }),
    180: ("many", {
        "HIN": cell("bʌhʊt / dʒada / adʰik", labels="1 / 3 / 5", column="right"),
        "BNM": cell("bʌhʊt", column="right"),
        "SkP": cell("bʌhʊt", column="right"),
        "BNT": cell("bʌhʊt", column="right"),
        "RNK": cell("bʰɔt", column="right"),
        "RKB": cell("bʰɔt", column="right"),
        "RNS_Sisaikhara": cell("bɔhʌt", column="right"),
        "TkN": cell("bɔt", column="right"),
        "CCC": cell("bʌhut", column="right"),
        "DKS": cell("bʌhɔt / dʰer", labels="1 / 2", column="right"),
        "DGC": cell("barider", labels="2", column="right"),
        "DkR": cell("dʰjar", labels="2", column="right"),
        "DDK": cell("bʌhuteder", labels="2", column="right"),
        "KkP": cell("bʌhuttʰer / dʒeʈa", labels="2 / 3", column="right"),
        "RKM": cell("bʌɖadʒoi", labels="4", column="right"),
        "RNS_Sisana": cell("bʌɾadʒo", labels="4", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(176, 181):
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
                    "duplicate source code RNS; within-item occurrence order assigned to "
                    "metadata row order"
                )
            if blank:
                uncertainty = "site code absent from the complete printed item block"
            rows.append({
                "Item": str(item), "Gloss": gloss, "Site_Key": site,
                "Source_Code": "DK" if item == 176 and site == "DKS" else SOURCE_CODES[site],
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
    assert sum(row["Review_Status"] == "attested" for row in rows) == 78
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 2
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 98
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 85
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
