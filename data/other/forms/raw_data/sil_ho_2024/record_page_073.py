#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 73 (items 4-6)."""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "manual_review.tsv"
SITES = ["HO1","HTH","HKA","HKE","HCH","HCU","HSU","HSA","HJO","HDH","HBG","HO2","HRA","HO3","HOP","HBA","HNI","BBG","BMA","BOP","BRA","BGH","MU1","MU2","SA1","SBA","OCU"]

# Literal source punctuation and similarity-class numbers are retained. None
# denotes a visually confirmed printed dash.
PAGE = {
    4: ("face", [
        "1 med muwa, 2 med motʃa", "2 motʃa", "1 met? muva", "1 me?t mɨe",
        "1 met? mɨva", "1 med? muẽ", "1 met? muve, 2 a motʃa", "2 motʃa",
        "1 mot? muva", "1 me?d mɨe", "1 met? mɨa", "1 med mua", "3 me?na",
        None, "2 met? muta", "1 met? muve", "5 tʃehera", "1 met?ŋ mɨana",
        "2 met mute", "1 me?t moan", "2 me?d mote", "2 met muta", None,
        "1 med muanra", "3 mẽt? ãhã", "3 met tʰʌ", "6 mɖhe",
    ]),
    5: ("eye", [
        "1 med", "1 met?", None, "1 me?n", "1 met?", "1 met?", "1 met?",
        "1 met?", None, "1 me?d", "1 met?", "1 med", "1 me?n", "1 met",
        "1 me?n", "1 met?", "1 met?ŋ", "1 met?ŋ", "1 met?", "1 me?n",
        "1 met?", "1 met?", "1 me?d", "1 med", "1 mẽt", "1 met", "2 akhi",
    ]),
    6: ("ear", [
        "1 luṯur", "1 luṯur", "1 luṯur", "1 luṯur", "1 luṯur", "1 luṯur",
        "1 luṯur", "1 luṯur", "1 luṯur", "1 luṯur", "1 luṯur", "1 luṯur",
        "1 luṯur", "1 luṯur", "1 luṯur", "1 luṯur", "1 luṯur", "1 luṯur",
        "1 luṯur", "1 luṯur", "1 luṯur", "1 luṯur", "1 luṯur", "1 luṯur",
        "1 luṯur", "1 luṯur", "2 kaŋo",
    ]),
}

MEDIUM = {
    (4,"HKE"): "central close vowel read from the report alphabet as ɨ",
    (4,"HCH"): "central close vowel read from the report alphabet as ɨ",
    (4,"HDH"): "central close vowel read from the report alphabet as ɨ",
    (4,"HBG"): "central close vowel read from the report alphabet as ɨ",
    (4,"BBG"): "central close vowel and final velar nasal resolved from alphabet chart",
    (4,"SA1"): "nasalized vowels retained with combining tilde",
    (4,"OCU"): "initial cluster contains source retroflex voiced stop glyph",
}


def main() -> None:
    with LEDGER.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t"); fields=reader.fieldnames; rows=list(reader)
    assert fields
    expected={(item,site) for item in PAGE for site in SITES}; seen=set()
    for row in rows:
        key=(int(row["Item"]),row["Site_Code"])
        if int(row["PDF_Page"]) != 73 or key not in expected: continue
        item,site=key; gloss,forms=PAGE[item]; form=forms[SITES.index(site)]
        uncertainty=MEDIUM.get(key, "")
        values={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if uncertainty else "high","Uncertainty":uncertainty,"Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
        if row["Review_Status"] == "unreviewed": row.update(values)
        else:
            for field,value in values.items():
                if row[field] != value: raise AssertionError(f"ledger conflict {item}/{site}/{field}")
        seen.add(key)
    if seen != expected: raise AssertionError(f"page topology drift: {len(seen)}")
    with LEDGER.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter="\t"); writer.writeheader(); writer.writerows(rows)
    print("recorded 81 manually reviewed cells for PDF page 73 (items 4-6)")

if __name__ == "__main__": main()
