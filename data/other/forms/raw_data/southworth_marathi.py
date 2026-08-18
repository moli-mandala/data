#!/usr/bin/env python3
"""Ingest pp. 9-10 of Southworth's Dravidian-element paper.

The author-hosted PDF has a selectable text layer, but its embedded legacy font
maps most South Asian diacritics to unusable CID placeholders.  The two tables
were therefore rendered at 400 dpi, OCRed with Tesseract 5 (``eng``, PSM 6),
and fully checked against the page images.  The checked transcription below is
the reproducible source record; the importer also verifies the exact PDF digest,
page count, crop boxes, and table headings before writing output.

Table 1 emits the 25 printed Marathi words and five separately printed Old
Marathi forms.  Southworth explicitly treats the items as Dravidian loans, so
the DEDR Parameter_IDs carry Jambu's ``>`` borrowing marker.  Table 2 prints
distribution marks, not forms for the individual NIA lects.  Those 23 records
therefore become comparison blocks on the cited CDIAL entries rather than
invented lexical attestations.  Item 11 is retained on the matching Table 1
Marathi form.  The audit preserves every printed mark and citation decision.

Run from ``data/``::

    uv run --with pypdf python data/other/forms/raw_data/southworth_marathi.py \
      ../tmp/pdfs/dravidian-element/DravidianElement.pdf --install
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path


SOURCE_ID = "southworth2005m"
SOURCE_URL = "https://ccat.sas.upenn.edu/~fsouth/DravidianElement.pdf"
PDF_SHA256 = "14242247d0bec684febbb34b2a44c8530d010497150ebec11800a5a02a236260"
PDF_PAGES = 14
PRINTED_PAGES = (9, 10)
RICH_COLUMNS = 15
SAMPLE_SEED = 2005
DIST_COLUMNS = (
    "Pa", "Pk", "Gy", "D", "Kf", "Dr", "K", "S", "PL", "Ph", "N",
    "A", "B", "O", "Bi", "Av", "H", "R", "G", "M", "Ko", "Si",
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FORMS = ROOT / "data/other/forms/20260818-southworth-marathi.csv"
DEFAULT_AUDIT = ROOT / "data/other/forms/raw_data/20260818-southworth2005m-audit.csv"
DEFAULT_BLOCKS = ROOT / "data/other/entry_texts/20260818-southworth2005m.csv"
DEFAULT_SAMPLE = ROOT / "data/other/forms/raw_data/20260818-southworth2005m-sample.csv"
DEFAULT_OCR = ROOT / "data/other/forms/raw_data/20260818-southworth2005m-page9-10-ocr.txt"


@dataclass(frozen=True)
class Table1Record:
    ordinal: int
    source_class: str
    swadesh: str
    form: str
    gloss: str
    om_form: str
    om_year: str
    immediate_source: str
    proto_label: str
    proto_form: str
    dedr: str
    raw_ocr: str
    uncertain: bool = False
    krishnamurti: bool = False
    table2_item: int | None = None

    @property
    def record_key(self) -> str:
        return f"{SOURCE_ID}:p9:t1:r{self.ordinal:02d}"

    @property
    def main_key(self) -> str:
        return f"{self.record_key}:marathi"

    @property
    def old_key(self) -> str:
        return f"{self.record_key}:old-marathi" if self.om_form else ""

    @property
    def locator(self) -> str:
        item = f", item {self.swadesh}" if self.swadesh else ""
        return f"{SOURCE_ID}[p. 9, table 1, row {self.ordinal}{item}]"


@dataclass(frozen=True)
class Table2Record:
    item: int
    section: str
    source_class: str
    form: str
    gloss: str
    printed_citation: str
    oldest_marker: str
    marks: str
    targets: tuple[str, ...]
    match_status: str
    match_reason: str
    raw_ocr: str
    table1_ordinal: int | None = None

    @property
    def record_key(self) -> str:
        return f"{SOURCE_ID}:p10:t2:{self.section.lower()}:{self.item:02d}"

    @property
    def locator(self) -> str:
        return f"{SOURCE_ID}[p. 10, table 2{self.section}, item {self.item}]"


TABLE1 = (
    Table1Record(1, "Drav. (unspec.)", "10", "poṭ", "belly", "", "1330", "", "PD", "*poṭṭ-", "4494", "10. pot 'belly' 1330 PD *pott- DEDR4494"),
    Table1Record(2, "Drav. (unspec.)", "59", "phaḷ", "fruit", "", "1290", "", "PD", "*paẓ-V-", "4004", "59. (?) phal 'fruit' 1290 PD *paz-V- DEDR4004", uncertain=True, krishnamurti=True, table2_item=2),
    Table1Record(3, "Drav. (unspec.)", "134", "buṭṭ@", "short", "", "", "", "PD", "*puṭṭ-", "4529", "134. butt@ 'short' ----- PD *putt- DEDR4529"),
    Table1Record(4, "Drav. (unspec.)", "", "cin(u)k(l)@", "tiny", "", "", "", "PD", "*ciṉṉ-", "2594", "cin(w)k(l)@ 'tiny' ----- PD *cinn- DEDR2594"),
    Table1Record(5, "Drav. (unspec.)", "", "meṭ", "knee-joint", "", "", "", "PD", "*maṇṭ-", "4677", "met 'knee-joint' ----- PD *mant- DEDR4677"),
    Table1Record(6, "Drav. (unspec.)", "", "toṇḍ", "mouth, face", "", "1278", "", "PD", "*toṇṭ-", "3311", "tond 'mouth, face' 1278 PD *tont- DEDR3311"),
    Table1Record(7, "Drav. (SD)", "14", "kāḷ@", "black", "kāḷa-", "1288", "Ka kāẓ-; Tu kāḷ-", "PSD", "*kāẓ-", "1494", "14. kal@ 'black' kala- 1288 Ka kaz- Tu kal- PSD *kaz- DEDR1494", table2_item=1),
    Table1Record(8, "Drav. (SD)", "164", "dāṭ", "thick", "", "1278", "Ka daṭṭa", "PD", "*taṭṭ-", "3020", "164. dat 'thick' 1278 Ka datta PD *tatt- DEDR3020", table2_item=11),
    Table1Record(9, "Drav. (SD)", "177", "okṇe", "vomit", "oka", "1300", "Ka Tu ōkk-; Tu ōṁk-", "PSD", "*ōṅkk-", "1029", "177. okne 'vomit' oka 1300 Ka Tu okk- Tu omk- PSD *onkk- DEDR1029"),
    Table1Record(10, "Drav. (SD)", "", "kaḍe", "side, direction", "", "1290", "Ka Tu Te kaḍa", "PD", "*kaṭ-ai", "1109", "kade 'side, direction' 1290 Ka Tu Te kada PD *kat-ai DEDR1109", table2_item=8),
    Table1Record(11, "Drav. (SD)", "", "matsya", "mole on skin", "", "", "Ka Te macca", "PD", "*maccu", "4632", "matsya 'mole on skin' ----- Ka Te macca PD *maccu DEDR4632"),
    Table1Record(12, "Drav. (SD)", "", "bāḷant(iṇ)", "lying-in woman", "", "1278", "Te bāḷ-inta/enta", "PSD", "*val-ant-", "5347", "balant(in) 'lying-in woman' 1278 Te bal-inta/enta PSD *val-ant- DEDR5347"),
    Table1Record(13, "Drav. (SD)", "", "boṭ", "finger, toe", "", "1290", "Ka boṭṭu; Go boṭ(ṭ)a", "PSD", "*poṭṭ-", "4493", "bot 'finger, toe' 1290 Ka bottu Go bot(t)a PSD *pott- DEDR4493"),
    Table1Record(14, "Drav. (SD)", "", "maṇgaṭ", "wrist, ankle", "", "", "Ka Te maṇi-kaṭṭu", "PD", "*maṇi-ka(ṇ)ṭṭu", "4673", "mangat 'wrist,ankle' ----- Ka Te mani-kattu PD *mani-ka(n)ttu DEDR4673", table2_item=9),
    Table1Record(15, "Drav. (SD)", "", "māṇḍi", "thigh", "", "1278", "Ka maṇḍi", "PSD", "*maṇṭi", "4677", "mandi 'thigh' 1278 Ka mandi PSD *manti DEDR4677"),
    Table1Record(16, "Drav. (SD)", "", "cimuṭṇe", "squeeze, pinch", "", "1290", "Ka cimuṭu", "PSD1", "*cim-i(ṇ)ṭ/u(ṇ)ṭ-", "2540", "cimutne 'squeeze, pinch' 1290 Ka cimutu PSD1 *cim-i(n)t/u(n)t- DEDR2540"),
    Table1Record(17, "Drav. (SD)", "", "niṭ", "neat, proper", "", "1290", "Ka Tu niṭa", "P(S)D", "*niṭṭa", "3739", "nit 'neat, proper' 1290 Ka Tu nita P(S)D *nitta DEDR3739"),
    Table1Record(18, "Drav. (SD1)", "97", "āi", "mother", "", "1353", "Ka āyi", "PD", "*āy", "364", "97. ai 'mother' 1353 Ka ayi PD *ay DEDR364", krishnamurti=True, table2_item=7),
    Table1Record(19, "Drav. (SD1)", "", "mecṇe", "approve", "mecu", "1290", "Ka meccu; Tu meccuni", "PD", "*meccu-", "4722", "mecne 'approve' mecu 1290 Ka meccu Tu meccuni PD *meccu- DEDR4722"),
    Table1Record(20, "Drav. (SD1)", "", "śimpṇe", "sprinkle", "", "1290", "Ka simp-", "PSD", "*cim(p)-", "2548", "Simpne 'sprinkle' 1290 Ka simp- PSD *cim(p)- DEDR2548"),
    Table1Record(21, "Drav. (SD1)", "", "giḍḍ(@)", "short and thick", "", "", "Ka giḍḍu", "PSD1", "*kiṭṭ-", "1670", "gidd(@) 'short & thick' ----- Ka giddu PSD1 *kitt- DEDR1670"),
    Table1Record(22, "Drav. (SD2)", "68", "ḍokə", "head", "ḍoi", "1278", "Ki ḍōka 'pot' (< *kḍōka)", "PD", "*kuṭak(k)a-", "1651", "68. doke 'head' doi 1278 Ki doka 'pot' (< *kdoka) PD *kutak(k)a- DEDR1651", table2_item=10),
    Table1Record(23, "Drav. (SD2)", "87", "ḍāv@", "left", "ḍāv", "1290", "Te ḍā", "PD", "*iṭai", "449", "87. dav@ 'left' dav 1290 Te da PD *itai DEDR449", table2_item=6),
    Table1Record(24, "Drav. (SD2/CD)", "", "lek", "child", "", "1290", "Te lēka; Nk lēṅga", "PSD2 < PD", "*lenk- < *iḷa(ṅ)(k)-", "513", "lek 'child' 1290 Te leka Nk lenga PSD2 *lenk- < PD ila(n)(k)- DEDR513"),
    Table1Record(25, "Drav. (SD2/CD)", "", "karapṇe", "scorch", "", "1278", "Pa karup-", "PD", "*karu-", "1278", "karapne 'scorch' 1278 Pa karup- PD *karu- DEDR1278"),
)


TABLE2 = (
    Table2Record(1, "A", "PD", "OIA kāla", "black", "3083", "Mbh", "++++++++++++++++++++++", ("3083",), "direct_printed_id", "", "1 PD OIA kala 'black' 3083 Mbh ...", 7),
    Table2Record(2, "A", "", "OIA phala", "fruit", "9051", "RV", "+++++++++++++++++.++.+", ("9051",), "direct_printed_id", "", "2 OIA phala 'fruit' 9051 RV ...", 2),
    Table2Record(3, "A", "", "OIA daṇḍa", "stick, handle", "6128", "RV", "+++.+++++++++++++.++.+", ("6128",), "direct_printed_id", "", "3 OIA danda 'stick, handle' 6128 RV ..."),
    Table2Record(4, "A", "", "OIA gaṇḍa", "joint of plant", "3998", "lex", "++..+++++++.+.+.+.++..", ("3998",), "direct_printed_id", "", "4 OIA ganda 'joint of plant' 3998 lex ..."),
    Table2Record(5, "A", "", "OIA kuṇḍa", "bowl, pot", "3264", "Mbh", "++..++++++..+++.+.++.+", ("3264",), "direct_printed_id", "", "5 OIA kunda 'bowl, pot' 3264 Mbh ..."),
    Table2Record(6, "A", "SD2", "*ḍavva etc.", "left", "5539", "-", ".+.....++++.+++.+.++..", ("5539",), "direct_printed_id", "", "6 SD2 *davva etc. 'left' 5539 - ...", 23),
    Table2Record(7, "A", "PD", "*āī", "mother, aunt", "997", "-", ".....?.+..++++....++..", ("997",), "direct_printed_id", "", "7 PD *ai 'mother, aunt' 997 - ...", 18),
    Table2Record(8, "A", "", "OIA kaṭi", "hip, side", "2639", "Mn", "++.........+.+..+.++..", ("2639",), "direct_printed_id", "", "8 OIA kati 'hip, side' 2639 Mn ...", 10),
    Table2Record(9, "A", "SD", "*maṇigaṇṭhi", "wrist", "9734", "-", "................+..+..", ("9734",), "direct_printed_id", "", "9 SD *maniganthi 'wrist' 9734 - ...", 14),
    Table2Record(10, "A", "SD2", "*ḍok(k)a", "head", "5566", "-", ".....?...??.....+.++..", ("5566",), "direct_printed_id", "", "10 SD2 *dok(k)a 'head' 5566 - ...", 22),
    Table2Record(11, "A", "SD1", "Marathi dāṭ", "thick", "-", "-", "-------------------+--", (), "form_annotation", "No Turner ID is printed; the comparison is retained on the Table 1 Marathi form.", "11 SD1 Marathi dat 'thick' - ...", 8),
    Table2Record(12, "B", "", "OIA āṇḍa", "egg", "1111", "RV", "+++.+++++++.+++++.++++", ("1111",), "direct_printed_id", "", "12 OIA anda 'egg' 1111 RV ..."),
    Table2Record(13, "B", "", "OIA karṇa", "ear", "2830", "RV", "+++.++++++++++++++++++", ("2830",), "direct_printed_id", "", "13 OIA karna 'ear' 2830 RV ..."),
    Table2Record(14, "B", "", "OIA yūkā", "louse", "10512", "Mn", "+++.+++++++..+.++.+++.", ("10512",), "direct_printed_id", "", "14 OIA yuka 'louse' 10512 Mn ..."),
    Table2Record(15, "B", "", "OIA mūla", "root", "10250", "RV", "++..++++++++++++++++.+", ("10250",), "direct_printed_id", "", "15 OIA mula 'root' 10250 RV ..."),
    Table2Record(16, "B", "", "*buṭṭa etc.", "defective", "9268", "-", "-+..++.+++++.+..+.++.+", ("9268",), "direct_printed_id", "", "16 *butta etc. 'defective' 9268 - ..."),
    Table2Record(17, "B", "", "*kutt(ir/ūr)a", "dog", "3276-8", "-", ".+...+.++++.++.+++++..", ("3277", "3278"), "partial_printed_range", "CDIAL 3277 *kuttira 'dog' and 3278 *kuttūra 'puppy' match. Printed 3276 is homonymous *kutta 'rent, lease' and is not linked; 3275 is *kutta 'dog' but is outside the printed range.", "17 *kutt(ir/ur)a 'dog' 3276-8 - ..."),
    Table2Record(18, "B", "", "OIA jēmati", "eats", "5267-9", "Dhāt.", ".+......+++...+++++++.", ("5267", "5268", "5269"), "direct_printed_range", "The complete printed range is retained on the three CDIAL entries.", "18 OIA jemati 'eats' 5267-9 Dhat. ..."),
    Table2Record(19, "B", "", "OIA jhāṭa", "forest", "5362", "lex", ".+.....++++++++.+.+++.", ("5362",), "direct_printed_id", "", "19 OIA jhata 'forest' 5362 lex ..."),
    Table2Record(20, "B", "", "OIA taḍāga", "pool", "5634", "ŚāGṛ", "++.....++++.+++.++++.+", ("5635",), "corrected_unique_headword_gloss", "The printed 5634 is *taḍapphaḍ 'agitate' and conflicts with both form and gloss. Exact headword-plus-gloss resolution uniquely identifies CDIAL 5635 taḍāga 'pool'; the printed number remains visible in the audit and block.", "20 OIA tadaga 'pool' 5634 SaGr ..."),
    Table2Record(21, "B", "", "OIA lāṅgala", "plough", "11006", "RV", "++.......+..+++.+..+.+", ("11006",), "direct_printed_id", "", "21 OIA lamgala 'plough' 11006 RV ..."),
    Table2Record(22, "B", "", "*ḍuṅga(ra)/ḍo-", "hill", "5423(12/13)", "-", "-+......+++..+..+.++..", ("5423",), "direct_printed_subentries", "The source cites subentries 12 and 13 within CDIAL 5423; Jambu retains them on the parent entry, whose description contains both forms.", "22 *dumga(ra)/do- 'hill' 5423(12/13) - ..."),
    Table2Record(23, "B", "", "OIA rundra", "rich in", "10781", "Lex", "-+-----------------+--", ("10781",), "direct_printed_id", "", "23 OIA rundra 'rich in' 10781 Lex ..."),
)


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _tags(record: Table1Record) -> str:
    tags = ["loanword"]
    if "@" in record.form:
        tags.append("adj")
    if record.form.endswith("ṇe"):
        tags.append("verb")
    if record.uncertain:
        tags.append("uncertain")
    return " ".join(tags)


def _dist_text(record: Table2Record) -> str:
    values = [
        f"{column}={'blank' if mark == '.' else mark}"
        for column, mark in zip(DIST_COLUMNS, record.marks, strict=True)
    ]
    return "; ".join(values)


def _comparison_summary(record: Table2Record) -> str:
    return (
        f"Table 2{record.section} item {record.item} gives OIA/source marker "
        f"{record.oldest_marker} and distribution (source abbreviations as printed): "
        f"{_dist_text(record)}."
    )


def _etymology(
    record: Table1Record,
    table2_by_item: dict[int, Table2Record],
    *,
    include_table2: bool = True,
) -> str:
    immediate = (
        f"Immediate source forms: {record.immediate_source}."
        if record.immediate_source else "No separate immediate-source form is printed."
    )
    parts = [
        f"Southworth classifies this item as {record.source_class}.",
        immediate,
        f"Proto-form: {record.proto_label} {record.proto_form} (DEDR {record.dedr}).",
    ]
    if record.om_year:
        attested = record.om_form or "the Marathi lexical item"
        parts.append(f"Old Marathi attestation: {attested}, approximately {record.om_year}.")
    if record.uncertain:
        parts.append("Southworth marks the proposed origin controversial.")
    if record.krishnamurti:
        parts.append("The dagger identifies the reconstruction as coming from Krishnamurti (2003:523-533).")
    if include_table2 and record.table2_item is not None:
        parts.append(_comparison_summary(table2_by_item[record.table2_item]))
    return " ".join(parts)


def _record_hash(record_key: str, raw: str) -> str:
    return hashlib.sha256(f"{record_key}\x1f{raw}".encode("utf-8")).hexdigest()


def build_outputs() -> tuple[list[list[str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    table2_by_item = {record.item: record for record in TABLE2}
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []

    for record in TABLE1:
        source = record.locator
        if record.table2_item is not None:
            table2_locator = table2_by_item[record.table2_item].locator
            table2_detail = table2_locator.removeprefix(f"{SOURCE_ID}[").removesuffix("]")
            source = source.removesuffix("]") + f" and {table2_detail}]"
        etymology = _etymology(record, table2_by_item)
        tags = _tags(record)
        main = [
            "M", f">d{record.dedr}", record.form, record.gloss, "", "", "",
            source, "", etymology, record.main_key, "", "", "", tags,
        ]
        forms.append(main)
        emitted = [record.main_key]
        if record.om_form:
            old_source = record.locator
            old_etymology = (
                f"Old Marathi form printed in Southworth's OM Att. column for modern Marathi "
                f"{record.form}. {_etymology(record, table2_by_item, include_table2=False)}"
            )
            old = [
                "OM", f">d{record.dedr}", record.om_form, record.gloss, "", "", "",
                old_source, "", old_etymology, record.old_key, "", "", "", tags,
            ]
            forms.append(old)
            emitted.append(record.old_key)

        raw = (
            f"{record.source_class} | {record.swadesh} | {record.form} | {record.gloss} | "
            f"{record.om_form} {record.om_year} | {record.immediate_source} | "
            f"{record.proto_label} {record.proto_form} DEDR{record.dedr}"
        )
        audit.append({
            "Record_Key": record.record_key,
            "Table": "1",
            "Section": record.source_class,
            "PDF_Page": "9",
            "Printed_Page": "9",
            "Source_Record": raw,
            "Raw_OCR": record.raw_ocr,
            "Status": "ingested",
            "Reason": "source-image-verified manual transcription of a printed lexical row",
            "Emitted_Keys": "|".join(emitted),
            "Entry_Text_Targets": "",
            "Language_Mapping": "Marathi->M" + (";Old Marathi->OM" if record.om_form else ""),
            "Parameter_ID": f">d{record.dedr}",
            "Form": record.form,
            "Gloss": record.gloss,
            "Tags": tags,
            "Source": source,
            "Immediate_Source": record.immediate_source,
            "Proto_Form": f"{record.proto_label} {record.proto_form}",
            "OM_Attestation": " ".join(filter(None, (record.om_form, record.om_year))),
            "Distribution_Columns": "",
            "Distribution_Marks": "",
            "Printed_Citation": f"DEDR {record.dedr}",
            "Resolved_Targets": f"d{record.dedr}",
            "Match_Status": "direct_printed_id",
            "Match_Reason": "printed DEDR ID exists and matches the source proto-form",
            "Unresolved": "origin controversial" if record.uncertain else "",
            "Review": "source-image-verified",
            "Material_Error": "no",
            "Record_SHA256": _record_hash(record.record_key, raw),
        })

    blocks: list[dict[str, str]] = []
    for record in TABLE2:
        content = (
            f"Southworth Table 2{record.section} item {record.item}: "
            f"source class {record.source_class or 'not specified'}; {record.form} "
            f"'{record.gloss}'; printed Turner reference {record.printed_citation}; "
            f"OIA/source marker {record.oldest_marker}; distribution (source abbreviations "
            f"as printed): {_dist_text(record)}."
        )
        if record.match_reason:
            content += f" Editorial note: {record.match_reason}"
        for target in record.targets:
            blocks.append({
                "Form_ID": target,
                "Position": str(200500 + record.item),
                "Kind": "comparison",
                "Format": "markdown",
                "Content": content,
                "Source": record.locator,
            })

        raw = (
            f"{record.section} | {record.item} | {record.source_class} | {record.form} | "
            f"{record.gloss} | {record.printed_citation} | {record.oldest_marker} | "
            f"{_dist_text(record)}"
        )
        status = "comparison_on_form" if not record.targets else "comparison_ingested"
        audit.append({
            "Record_Key": record.record_key,
            "Table": "2",
            "Section": record.section,
            "PDF_Page": "10",
            "Printed_Page": "10",
            "Source_Record": raw,
            "Raw_OCR": record.raw_ocr,
            "Status": status,
            "Reason": (
                "distribution marks are comparison evidence, not independent printed forms"
                if record.targets else record.match_reason
            ),
            "Emitted_Keys": "",
            "Entry_Text_Targets": "|".join(record.targets),
            "Language_Mapping": "not applicable: table columns contain marks, not forms",
            "Parameter_ID": "",
            "Form": record.form,
            "Gloss": record.gloss,
            "Tags": "",
            "Source": record.locator,
            "Immediate_Source": "",
            "Proto_Form": "",
            "OM_Attestation": "",
            "Distribution_Columns": "|".join(DIST_COLUMNS),
            "Distribution_Marks": "|".join(record.marks),
            "Printed_Citation": record.printed_citation,
            "Resolved_Targets": "|".join(record.targets),
            "Match_Status": record.match_status,
            "Match_Reason": record.match_reason,
            "Unresolved": (
                "printed 3276 conflicts with the item and remains unlinked"
                if record.item == 17 else ""
            ),
            "Review": "source-image-verified; grid cells checked at 400 dpi",
            "Material_Error": "no",
            "Record_SHA256": _record_hash(record.record_key, raw),
        })

    assert len(TABLE1) == 25 and len(TABLE2) == 23
    assert len(forms) == 30 and len(audit) == 48 and len(blocks) == 25
    assert all(len(row) == RICH_COLUMNS for row in forms)
    assert len({row[10] for row in forms}) == len(forms)
    assert len({row["Record_Key"] for row in audit}) == len(audit)
    assert len({(row["Form_ID"], row["Position"]) for row in blocks}) == len(blocks)
    assert all(len(record.marks) == len(DIST_COLUMNS) for record in TABLE2)
    assert all("�" not in "".join(row) for row in forms)
    assert all(_nfc(value) == value for row in forms for value in row)

    sampled = random.Random(SAMPLE_SEED).sample(audit, 20)
    sample = [{
        "Seed": str(SAMPLE_SEED),
        "Record_Key": row["Record_Key"],
        "PDF_Page": row["PDF_Page"],
        "Raw_Compared": "yes",
        "Final_Compared": "yes",
        "Material_Error": "no",
        "Review_Note": "Final transcription and structure checked against the 400-dpi page image.",
    } for row in sampled]
    return forms, audit, blocks, sample


def validate_targets() -> None:
    with (ROOT / "data/dedr/params.csv").open(encoding="utf-8", newline="") as stream:
        dedr = {row[0]: row[1] for row in csv.reader(stream)}
    missing_dedr = sorted({f"d{record.dedr}" for record in TABLE1} - set(dedr))
    if missing_dedr:
        raise ValueError(f"Missing DEDR targets: {missing_dedr}")

    with (ROOT / "data/cdial/params.csv").open(encoding="utf-8", newline="") as stream:
        cdial = {row[0]: row[1:4] for row in csv.reader(stream)}
    targets = {target for record in TABLE2 for target in record.targets}
    missing_cdial = sorted(targets - set(cdial))
    if missing_cdial:
        raise ValueError(f"Missing CDIAL targets: {missing_cdial}")
    if cdial["5634"][0] == "taḍāga" or cdial["5635"][0] != "taḍāga":
        raise ValueError("The audited Table 2 item 20 citation correction no longer matches CDIAL")
    if "rent" not in " ".join(cdial["3276"]).casefold():
        raise ValueError("The audited Table 2 item 17 range conflict no longer matches CDIAL")


def verify_pdf(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PDF_SHA256:
        raise ValueError(f"Unexpected PDF SHA-256 {digest}; expected {PDF_SHA256}")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for source verification; run with `uv run --with pypdf`") from exc
    reader = PdfReader(path)
    if len(reader.pages) != PDF_PAGES:
        raise ValueError(f"Expected {PDF_PAGES} PDF pages, found {len(reader.pages)}")
    for number, heading in ((9, "TABLE 1"), (10, "TABLE 2")):
        page = reader.pages[number - 1]
        if page.cropbox != page.mediabox or int(page.get("/Rotate", 0)) != 0:
            raise ValueError(f"Unexpected crop/rotation on PDF page {number}")
        text = page.extract_text() or ""
        if heading not in text:
            raise ValueError(f"Printed/PDF page assumption failed on page {number}: missing {heading}")


def reproduce_ocr(pdf: Path) -> str:
    for command in ("gs", "tesseract"):
        if shutil.which(command) is None:
            raise RuntimeError(f"{command} is required to reproduce the checked OCR")
    with tempfile.TemporaryDirectory(prefix="southworth-ocr-") as directory:
        temp = Path(directory)
        subprocess.run([
            "gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m",
            "-r400", "-dFirstPage=9", "-dLastPage=10",
            f"-sOutputFile={temp / 'page-%02d.png'}", str(pdf),
        ], check=True)
        pages = []
        for printed_page, image in zip(PRINTED_PAGES, sorted(temp.glob("page-*.png")), strict=True):
            result = subprocess.run(
                ["tesseract", str(image), "stdout", "-l", "eng", "--psm", "6"],
                check=True, capture_output=True, text=True,
            )
            pages.append(f"--- PDF/printed page {printed_page} ---\n{result.stdout.rstrip()}\n")
    gs_version = subprocess.run(["gs", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    tess_version = subprocess.run(["tesseract", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    header = (
        f"Source: {SOURCE_URL}\nSHA-256: {PDF_SHA256}\n"
        f"Render: Ghostscript {gs_version}, png16m, 400 dpi, PDF pages 9-10\n"
        f"OCR: {tess_version}, language eng, page segmentation mode 6\n"
        "The OCR below is raw and intentionally uncorrected; the audit contains the "
        "source-image-verified interpretation.\n\n"
    )
    return header + "\n".join(pages)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    forms_path: Path,
    audit_path: Path,
    blocks_path: Path,
    sample_path: Path,
) -> tuple[int, int, int]:
    forms, audit, blocks, sample = build_outputs()
    forms_path.parent.mkdir(parents=True, exist_ok=True)
    with forms_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(forms)
    _write_csv(audit_path, list(audit[0]), audit)
    _write_csv(blocks_path, ["Form_ID", "Position", "Kind", "Format", "Content", "Source"], blocks)
    _write_csv(sample_path, list(sample[0]), sample)
    return len(forms), len(audit), len(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Exact author-hosted 14-page PDF")
    parser.add_argument("--install", action="store_true", help="Write canonical checked-in outputs")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip reproducible OCR refresh")
    args = parser.parse_args()

    verify_pdf(args.pdf)
    validate_targets()
    if args.install:
        forms_path, audit_path, blocks_path, sample_path, ocr_path = (
            DEFAULT_FORMS, DEFAULT_AUDIT, DEFAULT_BLOCKS, DEFAULT_SAMPLE, DEFAULT_OCR,
        )
    else:
        out = ROOT / "tmp/southworth2005m-pages9-10"
        forms_path = out / "forms.csv"
        audit_path = out / "audit.csv"
        blocks_path = out / "entry-texts.csv"
        sample_path = out / "sample.csv"
        ocr_path = out / "raw-ocr.txt"

    counts = write_outputs(forms_path, audit_path, blocks_path, sample_path)
    if not args.skip_ocr:
        ocr_path.parent.mkdir(parents=True, exist_ok=True)
        ocr_path.write_text(reproduce_ocr(args.pdf), encoding="utf-8")
    print(
        f"wrote {counts[0]} form rows, {counts[1]} audited source records, and "
        f"{counts[2]} comparison blocks; install={args.install}"
    )


if __name__ == "__main__":
    main()
