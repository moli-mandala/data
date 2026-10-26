"""ASCII digraph aspirates (kh, bh, bɦ …) reach Form as the house aspirate Cʰ, source by source.

Each mapping was checked against the source's own transcription: Tulpule against its Devanagari
column, the SIL survey IPA (plain h after voiceless stops, ɦ after voiced ones), the Dadra Varli
ASCII key, Weinreich's Domaaki, and the Bengali rows of the SIL Bangladesh lists. Genuine
sequences and boundaries (Varli phul|har, Bareli bɔʈh hɔju, IPA j+h = yh) are kept.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pytest
from segments.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
FORMS = ROOT / "data/other/forms"
STOP_H = re.compile(r"[kgcjṭḍtdpbʈɖ][̀-ͯ]*h")


@lru_cache(maxsize=None)
def tokenizer_for(profile: str) -> Tokenizer:
    return Tokenizer(str(ROOT / f"conversion/{profile}.txt"))


def convert(profile: str, form: str) -> str:
    tokenizer = tokenizer_for(profile)
    out = tokenizer(form.strip("1234⁴5⁵67⁷,;.-"), column="IPA")
    return unicodedata.normalize("NFC", out.replace(" ", "").replace("#", " "))


def routed_rows(stem: str, key: str):
    import source_meta

    meta = source_meta.load()
    path = f"data/other/forms/{stem}.csv"
    with (FORMS / f"{stem}.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if row[7].split("[", 1)[0].split(";", 1)[0] != key:
                continue
            profile, on = meta.transcription(key, path, row[0])
            yield row, (profile if on else None)


@pytest.mark.parametrize("profile, source, expected", [
    # Tulpule: IAST aspirates, Marathi aspirated sonorants (म्ह न्ह ल्ह व्ह), vocalic ṛ
    ("tulpule-old-marathi", "akhara", "akʰara"),
    ("tulpule-old-marathi", "mhaṇaṇē", "mʰaṇaṇē"),
    ("tulpule-old-marathi", "avalhāvō", "avalʰāvō"),
    ("tulpule-old-marathi", "gavhara", "gavʰara"),
    ("tulpule-old-marathi", "cinha", "cinʰa"),
    ("tulpule-old-marathi", "dṛṣṭa", "dr̩ṣṭa"),
    # ण्ह, ऱ्ह and anusvāra + h stay sequences, as CDIAL writes Marathi kaṇhe, kurhāḍ
    ("tulpule-old-marathi", "kaṇhāṇē", "kaṇhāṇē"),
    ("tulpule-old-marathi", "kuṟhāḍī", "kuṟhāḍī"),
    ("tulpule-old-marathi", "siṁhāḍā", "siṁhāḍā"),
    # SIL Bareli: h after voiceless stops, ɦ after voiced ones; the dental bridge may sit on h/ɦ
    ("sil-bareli-pauri", "dʒibɦ", "jībʰ"),
    ("sil-bareli-pauri", "mɑth̪ɑ", "mātʰā"),
    ("sil-bareli-pauri", "mɑtɦ̪e", "mātʰe"),
    ("sil-bareli-pauri", "dz̪ɦ̪ʌʈa", "ʣʰaṭa"),
    ("sil-bareli-pauri", "mats̪h̪ʌ", "maʦʰa"),
    ("sil-bareli-pauri", "mɦʌis", "mʰaīs"),
    ("sil-bareli-pauri", "vʌɾhɑt̪", "varhāt"),          # r + h: CDIAL keeps rh
    ("sil-bareli-pauri", "bɔʈhhɔju", "bɔṭʰhɔyū"),       # bɔʈh + hɔju: the second h is a word
    ("sil-jaunsari", "d̪ɦəgɐ", "dʰaga"),
    ("sil-jaunsari", "matʃɦi", "macʰi"),
    ("sil-jaunsari", "tshɐppɐɾ", "ʦʰappar"),
    ("sil-jaunsari", "ɠhal", "ɠhal"),                   # unexplained font ɠ: left for review
    ("sil-noira", "khum", "kʰūm"),
    ("sil-noira", "dɦukla", "dʰūklā"),
    ("sil-noira", "ijha", "iyhā"),                     # IPA j + h is y + h, not jʰ
    ("sil-noira", "jʌʔhi", "yaʔhi"),
    ("sil-northern-dhule-bhils", "at̪h", "ātʰ"),
    ("sil-northern-dhule-bhils", "ɖuɖɦ", "ḍuḍʰ"),
    ("sil-northern-dhule-bhils", "mat̪s̪ho", "māʦʰo"),
    ("sil-northern-dhule-bhils", "atʱi", "ātʰi"),
    ("sil-northern-dhule-bhils", "jɦaki", "yhāki"),
    ("kannauji", "kʊlhaɖi", "kulʰāḍī"),
    # Dadra Varli ASCII: capital R is ṛ, and phul-har 'flower garland' keeps its boundary
    ("dadra-varli", "daRhi", "daṛʰi"),
    ("dadra-varli", "phulhar", "pʰulhar"),
    ("dadra-varli", "girha:n", "girhān"),
    # Weinreich's Domaaki; parentheses mark optional aspiration
    ("weinreich-domaaki", "kháša", "kʰáśa"),
    ("weinreich-domaaki", "ċhúṭo", "ʦʰúṭo"),
    ("weinreich-domaaki", "p(h)ačoó", "p(ʰ)acoó"),
    # Bengali rows of the SIL Bangladesh lists (Garo and Tripura reports)
    ("sil-bangladesh-bengali", "mɛɡh", "mɛgʰ"),
    ("sil-bangladesh-bengali", "ʃondha", "śondʰa"),
    ("sil-bangladesh-bengali", "maʈʃh", "macʰ"),
    ("sil-bangladesh-bengali", "bhuʈːa", "bʰuṭṭa"),
    ("zoller-2023", "buṛhī", "buṛʰī"),
    ("zoller-2023", "gɦɒrɔsiya", "gʰɒrɔsiya"),
])
def test_difficult_mappings(profile, source, expected):
    assert convert(profile, source) == expected


def test_tulpule_every_digraph_is_a_devanagari_aspirate_or_murmured_sonorant():
    asp = {"kh": "ख", "gh": "घ", "ch": "छ", "jh": "झ", "ṭh": "ठ", "ḍh": "ढ", "th": "थ",
           "dh": "ध", "ph": "फ", "bh": "भ"}
    son = {"mh": "म्ह", "nh": "न्ह", "lh": "ल्ह", "vh": "व्ह", "ḷh": "ळ्ह"}
    rows = list(routed_rows("20260810-tulpule-old-marathi", "tulpule1999"))
    assert {profile for _, profile in rows} == {"tulpule-old-marathi"}
    for row, _ in rows:
        form, native = unicodedata.normalize("NFC", row[2]), row[4]
        for digraph, letter in {**asp, **son}.items():
            assert form.count(digraph) == native.count(letter), (form, native, digraph)
        assert not STOP_H.search(convert("tulpule-old-marathi", row[2]))


@pytest.mark.parametrize("stem, key", [
    ("20260810-tulpule-old-marathi", "tulpule1999"),
    ("20260828-sil-bareli-pauri", "varkey-vunnamatla2018bareli"),
    ("20230526-kannauji", "kannauji"),
    ("20260828-sil-jaunsari", "john2008jaunsari"),
    ("20260829-sil-noira", "varghesekumar2015noira"),
    ("20260828-sil-eastern-gujari", "hugoniot-polster-ahmad-rajan2023easterngujari"),
    ("20260829-sil-northern-dhule-bhils", "watters2013northerndhule"),
    ("20260826-sil-garobd", "kim-kim-sangma2012garo"),
    ("20260826-sil-tripura", "kim-kim-sangma-ahmad2011tripura"),
    ("20260813-weinreich-domaaki", "weinreich2008"),
    ("20260911-dadra-varli", "pattanaik-koul2003varli"),
])
def test_no_ascii_stop_digraph_survives_conversion(stem, key):
    checked = 0
    for row, profile in routed_rows(stem, key):
        if key.startswith("kim-kim") and row[0] != "B":
            # Tibeto-Burman rows: C+h is not yet reviewed (Kuki-Chin əthɔu̯ is a sequence)
            assert profile != "sil-bangladesh-bengali"
            continue
        out = convert(profile, unicodedata.normalize("NFC", row[2]))
        assert "�" not in out, (row[2], out)
        assert not STOP_H.search(out), (row[2], out)
        checked += 1
    assert checked


def test_cdial_placeholder_vowel_and_markup():
    # DSAL prints "n/a" for a glyph it could not encode; Turner's own notes show its vowel is ε
    # (kṭg. nhn/aṇõ: "ε fr. intr. nhn/aṇõ"; bn/a, obl. bέa 'wedding')
    assert convert("cdial", "gn/arɔ") == "gɛrɔ"
    assert convert("cdial", "bn/aṇ") == "bɛṇ"
    assert convert("cdial", "tī́ thn/ar") == "tī́ tʰɛr"
    assert convert("cdial", "am<superscript>ŕ</superscript>n/aŕə") == "amʳ̆ɛr̆ə"
    assert convert("cdial", "z<smallcaps>ã</smallcaps>r") == "zãr"
    # superscripts are transitional glides, kept as modifier letters
    assert convert("cdial", "kagŕä<superscript>ŕ</superscript>") == "kagr̆æʳ̆"
    assert convert("cdial", "khuy<superscript>ĕ</superscript>hōm") == "kʰuyᵋhōm"
    with (ROOT / "data/cdial/cdial.csv").open(encoding="utf-8", newline="") as stream:
        leaked = [
            (row[0], piece) for row in csv.reader(stream) for piece in row[2].split(",")
            if ("<" in piece or "n/a" in piece)
            and any(mark in convert("cdial", piece) for mark in ("<", "n/", "�"))
        ]
    assert leaked == []
