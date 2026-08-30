#!/usr/bin/env python3
"""Write the OCR/PDF-text/legacy-blind manual ledger for items 16-20."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_016_020_hand_keyed.tsv")
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


def cell(form, labels, page, printed, column, qualifier=""):
    return form, labels, str(page), str(printed), column, qualifier


# Independently keyed by eye from 1200/1600-dpi crops before legacy comparison.
# Repeated identical responses remain separate slash-delimited occurrences.
ITEMS = {
    16: ("finger", {
        "HIN": cell("ʌ̃ŋɡʌli", "1", 34, 29, "left"),
        "BNM": cell("ʌ̃ŋɡʌli", "1", 34, 29, "left"),
        "BNT": cell("ʌ̃ŋɡʌli", "1", 34, 29, "left"),
        "RNS_Sisaikhara": cell("ʌ̃ŋɡʌɾja", "1", 34, 29, "left"),
        "RNK": cell("ũŋɡʌɾɪja", "1", 34, 29, "left"),
        "RKM": cell("ũŋɡʌɾɪja", "1", 34, 29, "left"),
        "RNS_Sisana": cell("ũŋɡʌɾɪja", "1", 34, 29, "left"),
        "DGC": cell("ʌŋɡuɾi", "1", 34, 29, "left"),
        "DkR": cell("ʌ̃ŋɡʌɾi", "1", 34, 29, "left"),
        "DKS": cell("ʌ̃ŋɡʌɾi", "1", 34, 29, "left"),
        "DDK": cell("ʌ̃ŋɡʌɾi", "1", 34, 29, "left"),
        "SkP": cell("ũŋɡʌɾi", "1", 34, 29, "left"),
        "RKB": cell("ũŋɡʌɾi / ũŋɡʌli", "1 / 1", 34, 29, "left"),
        "TkN": cell("uŋɡʌɾɪja", "1", 34, 29, "left"),
        "CCC": cell("juŋɡuri", "1", 34, 29, "left"),
        "KkP": cell("ɔŋɡʌɾi", "1", 34, 29, "left"),
    }),
    17: ("nail", {
        "HIN": cell("nãkʰun", "1", 34, 29, "left"),
        "RNS_Sisaikhara": cell("nãha", "2", 34, 29, "left"),
        "RNK": cell("nãha", "2", 34, 29, "left"),
        "SkP": cell("nã", "2", 34, 29, "left"),
        "RKB": cell("nah", "2", 34, 29, "left"),
        "TkN": cell("nah", "2", 34, 29, "left"),
        "RKM": cell("nahã", "2", 34, 29, "left"),
        "RNS_Sisana": cell("nahã", "2", 34, 29, "left"),
        "CCC": cell("nahuː", "2", 34, 29, "left"),
        "DGC": cell("nɔh", "2", 34, 29, "left"),
        "DkR": cell("nʊ̃", "2", 34, 29, "left"),
        "DKS": cell("nu / nuhũ", "2 / 2", 34, 29, "left"),
        "DDK": cell("no", "2", 34, 29, "left"),
        "KkP": cell("nãhu", "2", 34, 29, "left"),
        "BNM": cell("nũɡ", "3", 34, 29, "left"),
        "BNT": cell("nũɡ", "3", 34, 29, "left"),
    }),
    18: ("leg", {
        "HIN": cell("pæɾ", "1", 34, 29, "right"),
        "RNK": cell("ʈãŋɡ", "2", 34, 29, "right"),
        "RNS_Sisaikhara": cell("ʈãŋɡ / pãv", "2 / 4", 34, 29, "right"),
        "SkP": cell("ʈãŋɡ", "2", 34, 29, "right"),
        "RKB": cell("ʈãŋɡ", "2", 34, 29, "right"),
        "RNS_Sisana": cell("ʈãŋɡ", "2", 34, 29, "right"),
        "BNM": cell("ʈaŋɡ", "2", 34, 29, "right"),
        "TkN": cell("ʈãɡ", "2", 34, 29, "right"),
        "RKM": cell("ʈãɡ", "2", 34, 29, "right"),
        "CCC": cell("ʈaŋ", "2", 34, 29, "right"),
        "DkR": cell("ɡoɾa", "3", 34, 29, "right"),
        "DKS": cell("ɡoɾa", "3", 34, 29, "right"),
        "DDK": cell("ɡoɾ", "3", 34, 29, "right"),
        "DGC": cell("ɡoɾ / lat", "3 / 5", 34, 29, "right"),
        "KkP": cell("ɡoɾ", "3", 34, 29, "right"),
        "BNT": cell("paj", "4", 34, 29, "right"),
    }),
    19: ("skin", {
        "HIN": cell("kʰal / tʃʌmɾi", "1 / 2", 34, 29, "right"),
        "RNK": cell("kʰal", "1", 34, 29, "right"),
        "RNS_Sisaikhara": cell("kʰal", "1", 34, 29, "right"),
        "BNM": cell("kʰal", "1", 34, 29, "right"),
        "RKB": cell("kʰal", "1", 34, 29, "right"),
        "TkN": cell("kʰal", "1", 34, 29, "right"),
        "BNT": cell("kʰal", "1", 34, 29, "right"),
        "RKM": cell("kʰal", "1", 34, 29, "right"),
        "RNS_Sisana": cell("kʰal", "1", 34, 29, "right"),
        "DkR": cell("tʃokʌʈa / tʃʰala", "3 / 5", 34, 29, "right"),
        "DDK": cell("tʃɔkʌʈa", "3", 34, 29, "right"),
        "KkP": cell("tʃʰutʌka", "3", 34, 29, "right"),
        "SkP": cell("kʌlʌɾi", "4", 34, 29, "right"),
        "DKS": cell("tʃʰala", "5", 34, 29, "right"),
        "CCC": cell("tʃʰala", "5", 34, 29, "right"),
        "DGC": cell("tʃʰala", "5", 34, 29, "right"),
    }),
    20: ("bone", {
        "HIN": cell("hʌɖːi", "1", 34, 29, "right"),
        "RNK": cell("hʌɖːi", "1", 34, 29, "right"),
        "RNS_Sisaikhara": cell("hʌɖːi", "1", 34, 29, "right"),
        "BNM": cell("hʌɖːi", "1", 34, 29, "right"),
        "DkR": cell("hʌɖːi", "1", 34, 29, "right"),
        "SkP": cell("hʌɖːi", "1", 34, 29, "right"),
        "TkN": cell("hʌɖːi", "1", 34, 29, "right"),
        "DKS": cell("hʌɖːi / ɖaŋɡʌɾ", "1 / 2", "34 / 35", "29 / 30", "right / left"),
        "RNS_Sisana": cell("hʌɖːi", "1", 34, 29, "right"),
        "DGC": cell("hʌɖːi", "1", 34, 29, "right"),
        "RKB": cell("hʌɖːa", "1", 34, 29, "right"),
        "BNT": cell("hʌɖːa", "1", 34, 29, "right"),
        "KkP": cell("hʌɖːa", "1", 34, 29, "right"),
        "RKM": cell("hʌɽːi", "1", 35, 30, "left"),
        "CCC": cell("hʌɖ", "1", 35, 30, "left"),
        "DDK": cell("ɖoŋʌɾ", "2", 35, 30, "left"),
    }),
}


def main() -> None:
    rows = []
    for item in range(16, 21):
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
                    "metadata row order; sole group occurrence assigned to metadata row 1 "
                    "(Sisaikhara)"
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
    assert sum(int(row["Manual_Form_Count"]) for row in rows) == 87
    assert sum(
        int(row["Manual_Form_Count"]) for row in rows if row["Scope"] == "target"
    ) == 81
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
