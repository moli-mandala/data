#!/usr/bin/env python3
"""Parse Rajapurohit's Drasi Shina vocabulary into Jambu form rows.

``drasi`` is a lightly cleaned text extraction of pp. 77--166 of the PDF.  Most
lines are CSV-like, but the PDF's Gandhari Unicode font has a broken ToUnicode
map: a few glyphs are private-use characters and some words disappeared from
the extraction altogether.  ``ITEM_OVERRIDES`` records the readings checked
against rendered pages of the source.  Keeping those repairs here, rather than
silently editing the extraction, makes the transcription audit reproducible.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "drasi"
OUTPUT = HERE.parent / "20260725-drasi.csv"

LANGUAGE = "dr"
SOURCE = "rajapurohit2012"

# The common PUA glyph is visibly /ʌ/ in the rendered PDF.  The two rare glyphs
# encode a base plus combining marks in the old Gandhari Unicode font.
FONT_REPAIRS = str.maketrans({"\ue89d": "ʌ", "\ue899": "ə̃ũ", "\ue8bb": "ʌ̃y"})

# Complete payloads for entries damaged by PDF extraction.  Values use the same
# bracketed syntax as the raw file and are parsed by the normal path below.
ITEM_OVERRIDES: dict[int, str] = {
    36: "[bír] f.sg. [bíre] f.pl.",
    101: "[billẽ́ĩ]",
    143: "[ʧúɳi a:ʒéu bʌréu]",
    253: "[mũẽ]",
    265: "[biyã́ri]",
    273: "[dĩ] sg. [dĩye] pl.",
    278: "[ʤũ] sg. [ʤũẽ] pl.",
    283: "[mũ] sg. [mũẽ] pl.",
    286: "[hʌri ʧẽ́ĩ]",
    308: "[yó:ʧẽ́ĩ]",
    327: "[krĩ́:] sg. [krĩ́yẽ] pl.",
    385: "[ʌŋṹ] sg. [ʌŋṹẽ] pl.",
    388: "[byáli ʌŋṹ]",
    389: "[ʧúɳi ʌŋṹ]",
    428: "[kí] sg. [kíye] pl.",
    479: "[ã́ʐló:] sg. [ã́ʐlé:] pl.",
    574: "[briũ sãu]",
    576: "[ãŋũ:] (~ [ãó])",
    577: "[ə̃ŋʧár]",
    583: "[brĩũ]",
    594: "[khʌpẽ́ĩ] sg. [khʌpẽ́ẽ] pl.",
    642: "[kʌɳ(ə)wá:ʤi] small [ʤuró:ŋe] big, hollow",
    675: "[nilím] sg. [niliméh] pl.",
    683: "[riʃím]",
    729: "[goyã́:l]",
    745: "[ʃṹĩ]",
    750: (
        "[ʧhʌrí:] general, sg. [ʧhʌryéh] general, pl. "
        "[ʧhʌstʌ́n] grass, sg. [ʧhʌstʌn zʌŋgóʂ] grass, pl. "
        "[zʌŋgóʂ] goat skin, sg. [zʌŋgoʂéh] goat skin, pl."
    ),
    770: "[ə̃ʃ pá:l] sg. [ə̃ʃ pʌléh] pl.",
    771: "[ʃṹĩ]",
    777: "[gut] (~ [gʌt])",
    785: "[kʌt]",
    795: "[gʌɽá] big [ʧhʌɽʌ́l] small [táʈʂi] carpenter's tool",
    796: "[bʌ:s]",
    867: "[kʌũʌ̃l]",
    880: "[sãt(ə)rá:] (~ [sʌntʌrá:])",
    888: "[ʧi:v byẽĩ]",
    891: (
        "[keló: byẽĩ] tree (~ [byẽĩ]) [keló: púʂo] flower "
        "[keló: me:vá:] fruit [keló: pə́:ʈe] leaf"
    ),
    896: "[bíri]",
    901: "[brĩũ]",
    908: "[bĩ:]",
    915: "[me:vó: bĩ:]",
    920: "[tʌbʌríɳɖyo byẽĩ]",
    921: "[bíri]",
    1033: "[ʈʂãũ] sg. [ʈʂʌ̃ũẽ] pl.",
    1104: "[miʃríhʌ́k]",
    1129: "[kirkíro]",
    1180: "[móto hʌ̃ũk] [da:ná:]",
    1224: "[ã:ʂʈ]",
    1230: "[ʧó:dãi]",
    1231: "[pʌ́zilʌ̃ĩ]",
    1232: "[ʂõ:ĩ]",
    1234: "[ʌ̃ʂʈẽĩ]",
    1253: "[ek ɣʌ mu:ki ʈʂe:]",
    1273: "[pʌʂ] [mo:s bʌɣái] half of the month",
    1321: "[bʌzó:no] after winter [uʦ] of water",
    1636: "[ra:s] (~ [ra:z]) [hi:vo:no]",
    1761: "[gʌ̃yã:l]",
    1764: "[siʈe:ʤ]",
    1866: "[zulúm thyó:no] [móʐ(i) dyó:no]",
    1921: "[ʂã: thyó:no]",
    1942: "[khʌ́rigiʌ] [do:kha dyó:no]",
    2017: "[ʃʌk thyó:no] [hi:v ne preiʒó:no]",
    2107: "[rʌʧhó:no] [hɛfázʌt thyó:no]",
    2125: "[ʂã: thyó:no]",
    2158: "[kuʈyó:no] [ɖil thyó:no]",
    2210: "[á:ʤʌ̃y]",
    2216: "[roŋ khaʂ thyó:no] [rã: dyó:no]",
    2217: "[ʂã: ó:no]",
    2473: "[krã: dyó:no] [hʌryó:no]",
    2505: "[bʌrá:v] Hindi",
    2532: "[lã:ʈi]",
    2540: "[muɣúr] cup of the lo:ʈa: type",
}

# One vocabulary item at every PDF page break was lost by the original text
# extraction.  These rows come from the rendered page footers/headers.  Items
# 741 and 1958 are also absent from the printed source itself, so are not
# invented here.
INJECTED_ITEMS = {
    116: ("Female", "[sõ:ʈʂi]"),
    165: ("Paternal aunt's son", "[phapyó bá:l]"),
    221: ("Cat", "[píʃu] male, sg. [píʃe] male, pl."),
    288: ("Owl", "[ɦú:] sg. [ɦúe] pl."),
    353: ("Buttock", "[phoŋs] [sʌŋáy]"),
    423: ("Kidney", "[ʐúk] sg. [ʐúki] pl."),
    494: ("Tuberculosis", "[ʃuʃuró:k]"),
    555: ("Kitchen", "[bái thenek gó:ʂ]"),
    619: ("Banian (under wear of shirt)", "[bʌniyán]"),
    677: ("Scarf", "[gulbʌ́n] sg. [gulbʌnéh] pl."),
    739: ("Hearth (fire place)", "[ʈʂʌŋú:l]"),
    862: ("Land", "[kúi]"),
    926: ("Tree", "[byẽĩ]"),
    992: ("Postman", "[ɖakpá:]"),
    1052: ("Curl", "[khiŋíro] [ʤʌkuí]"),
    1109: ("Neatness", "[sʌfái]"),
    1168: ("Various", "[buʧhé] (~ [buʧʧhé]) many, much"),
    1218: ("Two", "[du:]"),
    1278: ("February", "[phʌrvari]"),
    1334: ("Sunset", "[bé:ʈi] [byóno]"),
    1387: ("Fury", "[ro:ʂ]"),
    1443: ("Ink", "[mil] sg. [míle] pl."),
    1506: ("President", "[preziɖeɳʈ]"),
    1564: ("Prisoner", "[qɛdí:] sg. [qaidyé:] pl."),
    1621: ("Loss", "[nuksá:n]"),
    1677: ("Festival", "[syó: de:s] (~ [syódde:s]) note juncture"),
    1732: ("Kabaddi (a sport)", "[kʌpʌɽí:]"),
    1782: ("Lead", "[səŋgá:]"),
    1837: ("Near", "[ʔéili] [nʌlá:]"),
    1884: ("Arrange", "[sí:te ʧoryó:no] (~ [sí:the...])"),
    1984: ("Cultivate", "[kúi vó:no] [bá:n thyó:no]"),
    2036: ("Encourage", "[hi:v bʌ:ɽo théiryó:no]"),
    2085: ("Get", "[ʌryó:no] [giɳyó:no]"),
    2136: ("Increase", "[bʌɽo bó:no] intr."),
    2190: ("Mend", "[prʌyó:no] (~ [prʌ:no])"),
    2246: ("Predict", "[dʌʂʈyó:no]"),
    2301: ("Run", "[dʌrbʌk thyó:no] intr. [bolyó:no] tr."),
    2347: ("Slide", "[hiná:l vʌʒó:no]"),
    2397: ("Study", "[ʧó:ko bó:no] [tʌya:r bó:no]"),
    2448: ("Turn", "[phiryó:no] around [phʌryó:no] aside"),
    2496: ("Wrap up", "[ʈópul thyó:no]"),
}


@dataclass(frozen=True)
class Form:
    item: int
    gloss: str
    form: str
    notes: str = ""


def _clean_note(note: str) -> str:
    note = note.translate(FONT_REPAIRS)
    note = note.replace("`", "").replace("’", "'")
    # Unbalanced brackets are remnants of a damaged alternate, never prose.
    note = re.sub(r"\[.*$", "", note).replace("]", "")
    note = re.sub(
        r"\b(?:Names of the months|Week days|Trees|Birds|Wild animals|"
        r"Tamed animals|House-hold things|Agricultural equipments):?\s*$",
        "",
        note,
    )
    note = re.sub(r"\(\s*~\s*$", "", note)
    note = re.sub(r"^[\s;,.~()]+|[\s;,.~()]+$", "", note)
    return re.sub(r"\s+", " ", note)


def parse_payload(item: int, gloss: str, payload: str) -> list[Form]:
    payload = payload.translate(FONT_REPAIRS)
    matches = list(re.finditer(r"\[([^\[\]]+)\]", payload))
    forms: list[Form] = []
    for index, match in enumerate(matches):
        form = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;")
        if not form:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(payload)
        notes = _clean_note(payload[match.end() : end])
        forms.append(Form(item, gloss.strip(), form, notes))
    return forms


def parse_lines(lines: list[str]) -> list[Form]:
    logical: list[list[str]] = []
    current: list[str] | None = None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(("\\d+", "(^", "(\\[")):
            continue
        if re.match(r"^9\.\d+\s*,?", line) or line == "9.26 Verbs:":
            continue
        continuation = re.match(r"^,,(.*)$", line)
        if continuation:
            if current is None:
                raise ValueError(f"continuation before an item: {line}")
            current[2] += " " + continuation.group(1)
            continue

        match = re.match(r"^(\d+)(?:[.,]\s*|\s+)(.*)$", line)
        if not match:
            raise ValueError(f"unrecognised Drasi line: {line}")
        item = int(match.group(1))
        remainder = match.group(2)
        if item == 25 and remainder.strip() == "Functional words:":
            current = None
            continue
        if "," in remainder:
            gloss, payload = remainder.split(",", 1)
        else:
            # Source item 2464 is printed without punctuation between gloss/form.
            bracket = remainder.find("[")
            if bracket < 0:
                gloss, payload = remainder, ""
            else:
                gloss, payload = remainder[:bracket], remainder[bracket:]
        # The extraction misread source number 2540 as 2240.
        if item == 2240 and gloss.strip().lower().startswith("cup"):
            item = 2540
        if item == 272 and gloss.strip() == "Lion’s cub":
            item = 275
        current = [str(item), gloss.strip(), payload.strip()]
        logical.append(current)

    for item, (gloss, payload) in INJECTED_ITEMS.items():
        logical.append([str(item), gloss, payload])

    result: list[Form] = []
    seen_items: set[int] = set()
    for item_text, gloss, payload in logical:
        item = int(item_text)
        if item in ITEM_OVERRIDES:
            payload = ITEM_OVERRIDES[item]
        forms = parse_payload(item, gloss, payload)
        if not forms:
            raise ValueError(f"item {item} ({gloss}) has no recoverable form")
        result.extend(forms)
        seen_items.add(item)

    missing_overrides = ITEM_OVERRIDES.keys() - seen_items
    if missing_overrides:
        raise ValueError(f"overrides do not correspond to source items: {sorted(missing_overrides)}")
    result.sort(key=lambda form: form.item)
    return result


def write_csv(forms: list[Form], destination: Path = OUTPUT) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for entry in forms:
            writer.writerow(
                [
                    LANGUAGE,
                    "",  # unetymologised manual rows become lone nodes
                    entry.form,
                    entry.gloss,
                    "",
                    entry.form,
                    entry.notes,
                    SOURCE,
                ]
            )


def main() -> None:
    forms = parse_lines(INPUT.read_text(encoding="utf-8").splitlines())
    write_csv(forms)
    print(f"Wrote {len(forms)} Drasi forms from {len({f.item for f in forms})} items to {OUTPUT}")


if __name__ == "__main__":
    main()
