#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 181-185."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_181_185_hand_keyed.tsv")
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


def cell(form, labels="1", page="64", printed="59", column="left", qualifier=""):
    return form, labels, page, printed, column, qualifier


# Independently keyed by eye from 900/1800-dpi rendered-page crops before any
# comparison with the legacy CSV.
ITEMS = {
    181: ("all", {
        "HIN": cell("sare / sʌb", labels="1 / 2", page="63", printed="58", column="right"),
        "BNT": cell("sare", page="63", printed="58", column="right"),
        "BNM": cell("sare / sʌb", labels="1 / 2", page="63", printed="58", column="right"),
        "DkR": cell("sʌɾʌdʒ", page="63", printed="58", column="right"),
        "SkP": cell("sara", page="63", printed="58", column="right"),
        "RNK": cell("sʌb", labels="2", page="63", printed="58", column="right"),
        "RNS_Sisaikhara": cell("sʌb", labels="2", page="63", printed="58", column="right"),
        "DGC": cell("sʌb / kul", labels="2 / 7", page="63", printed="58", column="right"),
        "KkP": cell("sʌb / dʒʌmːa", labels="2 / 5", page="63", printed="58", column="right"),
        "RKB": cell("sʌb", labels="2", page="63", printed="58", column="right", qualifier="(176)"),
        "TkN": cell("sʌb", labels="2", page="63", printed="58", column="right", qualifier="(176)"),
        "RKM": cell("sʌb", labels="2", page="63", printed="58", column="right", qualifier="(176)"),
        "DDK": cell("sʌkːu", labels="3", page="63", printed="58", column="right", qualifier="(176)"),
        "DKS": cell("sʌkːu", labels="3", page="63", printed="58", column="right", qualifier="(176)"),
        "RNS_Sisana": cell("sʌbbʰar", labels="4", page="63", printed="58", column="right"),
        "CCC": cell("dʒʌmai / bʰare", labels="5 / 6", page="63", printed="58", column="right"),
    }),
    182: ("eat!; he ate", {
        "HIN": cell("khao / kʰaja", page="63", printed="58", column="right"),
        "RKB": cell("kao / kajalɔ", page="63", printed="58", column="right"),
        "RNK": cell("kʰao / kʰale", page="63", printed="58", column="right"),
        "RNS_Sisaikhara": cell("kʰao / kʰale", page="63", printed="58", column="right"),
        "BNM": cell("kʰao / kʰalæ", page="63", printed="58", column="right"),
        "DGC": cell("kʰale", page="63", printed="58", column="right"),
        "SkP": cell(
            "kʰajʌlja / halʌe", page="63", printed="58", column="right",
            qualifier="second response prefixed by literal ellipsis (...)"
        ),
        "DkR": cell("kʰa / kʰʌinu", page="63", printed="58", column="right"),
        "TkN": cell("kʰajle / kʰanlʊ"),
        "DKS": cell("kʰao / kʰadʌnu"),
        "BNT": cell("kʰalo / kʰali"),
        "RKM": cell("kʰalere / kʰaʊrʌhʊ"),
        "RNS_Sisana": cell("kʰailo / kʰailʊ"),
        "CCC": cell("kʰʌi"),
        "DDK": cell(
            "kʰadʒʰʌtʌe / kʰa",
            qualifier="second response followed by literal ellipsis (...)"
        ),
        "KkP": cell("kʰaliʰʌl"),
    }),
    183: ("bite!; he bit", {
        "HIN": cell("kaʈo / kaʈa"),
        "RKB": cell("kaʈo / kaʈi"),
        "RNK": cell("kaʈo / kaʈle"),
        "RNS_Sisaikhara": cell("kaʈo / kaʈle"),
        "DkR": cell("kaʈ / kaʈʌnu"),
        "SkP": cell("kaʈdja / kaʈʌlhala"),
        "TkN": cell("kaʈde / kaʈdu"),
        "DKS": cell("kaʈ / kaʈʌlas"),
        "BNM": cell("kaʈkʰao"),
        "BNT": cell("kaʈʌlo / kaʈʌleta"),
        "RKM": cell("kaʈo / kaʈoɾʌhe"),
        "RNS_Sisana": cell("kaʈ / kaʈdʌʊ"),
        "CCC": cell("kʌtʌi / tokʌi", labels="1 / 3"),
        "DDK": cell("kʌʈ / kaʈlelu"),
        "DGC": cell("kaʈdæ̃ / kaʈduinu"),
        "KkP": cell("kaʈʌdʌhʌl"),
    }),
    184: ("he is/was hungry", {
        "HIN": cell("bʰukʰa / bʰukʰa"),
        "RNK": cell("bʰukʰo"),
        "RNS_Sisaikhara": cell("bʰukʰo"),
        "RKB": cell("bʰukʰo / bʰukʰa"),
        "BNM": cell("bʰukʰao / bʰukʰtʰa"),
        "DGC": cell("bʰukʰ"),
        "DkR": cell("bʰukʰʌlʌ̃ / bʰukʰʌl"),
        "SkP": cell(
            "bʰukʰero / rʌhe",
            qualifier="second response prefixed by literal ellipsis (...)"
        ),
        "TkN": cell("bʰukʰohae / bʰukʰo"),
        "DKS": cell(
            "bʰukʰ / bʰukʰajʌl",
            qualifier=(
                "first response followed by literal ellipsis (...); second response "
                "followed by two literal periods (..)"
            ),
        ),
        "BNT": cell("bʰukʰahae / bʰukʰatʰa"),
        "RKM": cell(
            "bʰukʰo / bʰukʰorʌ",
            qualifier="second response followed by literal ellipsis (...)"
        ),
        "RNS_Sisana": cell("bʰukʰo / bʰukʰorʌhʊ"),
        "DDK": cell(
            "bʰukʰajlʌwa / bʰukʰ",
            qualifier="second response followed by two literal periods (..)"
        ),
        "KkP": cell("bʰukʰajlʌhʌle"),
        "CCC": cell(None),
    }),
    185: ("drink!; he drank", {
        "HIN": cell("pijo / pija"),
        "BNM": cell("pijo / pija"),
        "RNK": cell("pile / pilʌi"),
        "RNS_Sisaikhara": cell("pile / pilʌi"),
        "DGC": cell("pile"),
        "RKB": cell("pil / pilija"),
        "DkR": cell("pi / pisekʌnu"),
        "SkP": cell("pilja / piljahʌl"),
        "TkN": cell("pile / pilʌ", column="right"),
        "DKS": cell("pijo / pikʰʌnu", column="right"),
        "BNT": cell("pilo / pilija", column="right"),
        "RKM": cell("pile / pilʊ", column="right"),
        "RNS_Sisana": cell("pile / pilʊhʊ̃", column="right"),
        "CCC": cell("piʌi", column="right"),
        "DDK": cell("pile / pilelu", column="right"),
        "KkP": cell("pilʌhʌl", column="right"),
    }),
}


def main() -> None:
    rows = []
    for item in range(181, 186):
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 135
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 125
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
