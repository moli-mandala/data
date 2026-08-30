import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKSPACE = ROOT.parent
PACKAGE = ROOT / "data/other/forms/raw_data/sil_rabha_2013"
CHUNKS = [
    PACKAGE / "manual_chunks/p022-p023-items-S001-S010.tsv",
    PACKAGE / "manual_chunks/p023-items-S011-S020.tsv",
    PACKAGE / "manual_chunks/p023-p024-items-S021-S030.tsv",
    PACKAGE / "manual_chunks/p024-p025-items-S031-S040.tsv",
    PACKAGE / "manual_chunks/p025-items-S041-S050.tsv",
    PACKAGE / "manual_chunks/p025-p026-items-S051-S060.tsv",
    PACKAGE / "manual_chunks/p026-items-S061-S070.tsv",
    PACKAGE / "manual_chunks/p027-items-S071-S080.tsv",
    PACKAGE / "manual_chunks/p027-p028-items-S081-S090.tsv",
    PACKAGE / "manual_chunks/p028-items-S091-S100.tsv",
    PACKAGE / "manual_chunks/p028-p029-items-S101-S110.tsv",
    PACKAGE / "manual_chunks/p029-p030-items-S111-S120.tsv",
    PACKAGE / "manual_chunks/p030-items-S121-S130.tsv",
    PACKAGE / "manual_chunks/p030-p031-items-S131-S140.tsv",
    PACKAGE / "manual_chunks/p031-p032-items-S141-S150.tsv",
    PACKAGE / "manual_chunks/p032-items-S151-S160.tsv",
    PACKAGE / "manual_chunks/p032-p033-items-S161-S170.tsv",
    PACKAGE / "manual_chunks/p033-items-S171-S180.tsv",
    PACKAGE / "manual_chunks/p033-p034-items-S181-S190.tsv",
    PACKAGE / "manual_chunks/p034-items-S191-S194.tsv",
]
PDF = WORKSPACE / "tmp/pdfs/sil_rabha_2013/silesr2013_016.pdf"
INSTALLED = ROOT / "data/other/forms/20260813-rabha.csv"
CENSUS = ROOT / "data/other/forms/raw_data/sil_survey_sources.md"


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def manual_rows():
    return [row for chunk in CHUNKS for row in read_tsv(chunk)]


def test_source_pin_and_appendix_topology():
    manifest = json.loads((PACKAGE / "source_manifest.json").read_text(encoding="utf-8"))
    assert PDF.stat().st_size == manifest["bytes"] == 1_118_896
    assert hashlib.sha256(PDF.read_bytes()).hexdigest() == manifest["sha256"]
    assert manifest["physical_pages"] == 37

    appendix = manifest["lexical_appendix"]
    assert appendix["elicited_standard_items_reported"] == 210
    assert appendix["published_prompt_rows"] == 194
    assert appendix["omitted_problematic_items"] == 16
    assert appendix["lists"] == 2
    assert appendix["conceptual_cells"] == 388

    pages = read_tsv(PACKAGE / "page_review.tsv")
    assert [int(row["PDF_Page"]) for row in pages] == list(range(22, 35))
    assert [int(row["Printed_Page"]) for row in pages] == list(range(18, 31))
    assert sum(int(row["Prompt_Rows"]) for row in pages) == 194
    assert sum(int(row["Conceptual_Cells"]) for row in pages) == 388
    assert sum(int(row["Reviewed_Cells"]) for row in pages) == 388
    assert sum(int(row["Pending_Cells"]) for row in pages) == 0
    assert next(row for row in pages if row["PDF_Page"] == "23")["Status"] == "complete"
    assert next(row for row in pages if row["PDF_Page"] == "24")["Status"] == "complete"
    assert next(row for row in pages if row["PDF_Page"] == "25")["Status"] == "complete"
    assert next(row for row in pages if row["PDF_Page"] == "26")["Status"] == "complete"
    assert next(row for row in pages if row["PDF_Page"] == "27")["Status"] == "complete"
    assert next(row for row in pages if row["PDF_Page"] == "28")["Status"] == "complete"
    assert next(row for row in pages if row["PDF_Page"] == "29")["Status"] == "complete"
    assert next(row for row in pages if row["PDF_Page"] == "30")["Status"] == "complete"
    assert next(row for row in pages if row["PDF_Page"] == "31")["Status"] == "complete"
    assert next(row for row in pages if row["PDF_Page"] == "32")["Status"] == "complete"
    assert next(row for row in pages if row["PDF_Page"] == "33")["Status"] == "complete"
    assert next(row for row in pages if row["PDF_Page"] == "34")["Status"] == "complete"
    assert [(row["PDF_Page"], row["Prompt_Rows"]) for row in pages if row["Prompt_Rows"] != "16"] == [
        ("22", "6"), ("28", "15"), ("34", "13")
    ]


def test_covered_chunks_are_exhaustive_manual_nfc_and_coordinate_cited():
    rows = manual_rows()
    assert [chunk.name for chunk in CHUNKS] == [
        "p022-p023-items-S001-S010.tsv",
        "p023-items-S011-S020.tsv",
        "p023-p024-items-S021-S030.tsv",
        "p024-p025-items-S031-S040.tsv",
        "p025-items-S041-S050.tsv",
        "p025-p026-items-S051-S060.tsv",
        "p026-items-S061-S070.tsv",
        "p027-items-S071-S080.tsv",
        "p027-p028-items-S081-S090.tsv",
        "p028-items-S091-S100.tsv",
        "p028-p029-items-S101-S110.tsv",
        "p029-p030-items-S111-S120.tsv",
        "p030-items-S121-S130.tsv",
        "p030-p031-items-S131-S140.tsv",
        "p031-p032-items-S141-S150.tsv",
        "p032-items-S151-S160.tsv",
        "p032-p033-items-S161-S170.tsv",
        "p033-items-S171-S180.tsv",
        "p033-p034-items-S181-S190.tsv",
        "p034-items-S191-S194.tsv",
    ]
    assert len(rows) == 388
    assert len({row["Cell_Key"] for row in rows}) == 388
    assert Counter(row["Cell_Status"] for row in rows) == {
        "attested": 387,
        "source_blank": 1,
    }
    assert {row["Confidence"] for row in rows} == {"high"}
    assert {row["Reviewer"] for row in rows} == {"OpenAI Codex"}
    assert all("manual visual transcription" in row["Review_Method"] for row in rows)
    assert all("never to supply or verify" in row["Review_Method"] for row in rows)
    assert all("OCR" not in key.upper() for key in rows[0])
    assert all("physical PDF p." in row["Notes"] and "printed p." in row["Notes"] for row in rows)
    assert all("source-order item S" in row["Notes"] and "list " in row["Notes"] for row in rows)
    assert all(
        unicodedata.is_normalized("NFC", value)
        for row in rows
        for value in row.values()
    )

    expected_keys = {
        f"S{item:03d}_{lect}"
        for item in range(1, 195)
        for lect in ("RDG", "MTR")
    }
    assert {row["Cell_Key"] for row in rows} == expected_keys
    assert sum(
        len(row["Manual_Form"].split(", "))
        for row in rows
        if row["Cell_Status"] == "attested"
    ) == 399
    forms = {row["Cell_Key"]: row["Manual_Form"] for row in rows}
    assert {key: forms[key] for key in (
        "S021_MTR", "S025_MTR", "S028_MTR", "S030_RDG", "S030_MTR",
        "S032_RDG", "S034_RDG", "S038_RDG", "S040_RDG",
        "S044_RDG", "S045_MTR", "S048_RDG", "S048_MTR", "S050_RDG",
        "S052_RDG", "S055_RDG", "S055_MTR", "S057_MTR", "S060_RDG",
        "S061_RDG", "S061_MTR", "S067_RDG", "S068_MTR", "S070_MTR",
        "S072_RDG", "S072_MTR", "S075_RDG", "S078_MTR", "S079_RDG",
        "S081_RDG", "S082_RDG", "S085_RDG", "S085_MTR", "S090_RDG",
        "S091_RDG", "S092_MTR", "S096_MTR", "S099_MTR", "S100_RDG",
        "S101_RDG", "S102_RDG", "S105_RDG", "S105_MTR", "S109_RDG",
        "S111_RDG", "S114_RDG", "S118_RDG", "S120_MTR",
        "S121_RDG", "S122_RDG", "S125_RDG", "S126_RDG", "S130_RDG",
        "S132_RDG", "S134_RDG", "S137_MTR", "S138_MTR", "S140_RDG",
        "S142_RDG", "S145_RDG", "S148_RDG", "S148_MTR", "S149_MTR",
        "S153_RDG", "S154_RDG", "S155_RDG", "S156_MTR", "S157_RDG",
        "S158_RDG", "S159_RDG", "S160_MTR",
        "S161_MTR", "S163_RDG", "S163_MTR", "S164_RDG", "S165_RDG",
        "S166_RDG", "S168_MTR", "S169_RDG", "S170_MTR",
        "S171_RDG", "S172_MTR", "S173_MTR", "S174_RDG", "S176_MTR",
        "S178_RDG", "S179_RDG", "S180_RDG",
        "S181_RDG", "S182_MTR", "S183_RDG", "S183_MTR", "S184_MTR",
        "S185_RDG", "S187_RDG", "S189_RDG", "S190_MTR",
        "S191_RDG", "S191_MTR", "S192_RDG", "S192_MTR", "S193_RDG",
        "S194_MTR"
    )} == {
        "S021_MTR": "pemuŋni kʰɑpɑk̚",
        "S025_MTR": "nok̚tʃ̑ɑl",
        "S028_MTR": "noʋek̚",
        "S030_RDG": "gɑjnɑŋ",
        "S030_MTR": "ɡɑʲnɨŋ",
        "S032_RDG": "kɑŋkə, tʃ̑ɑku",
        "S034_RDG": "pɑgɑ",
        "S038_RDG": "tʃ̑ətʃ̑əkɑm",
        "S040_RDG": "lɑŋgɾe",
        "S044_RDG": "tʃ̑ɪ̃ kɑ",
        "S045_MTR": "tʃ̑kɑ d͡ʒoɾɑ",
        "S048_RDG": "bɑjsənkʰi, ɾɑmdʰonuʃ",
        "S048_MTR": "bɑʲtʃ̑oɾoŋ",
        "S050_RDG": "ɾóŋkɑ",
        "S052_RDG": "hɑ́ŋtʃ̑eŋ",
        "S055_RDG": "pʰúŋd͡ʒi",
        "S055_MTR": "pʰuŋd͡ʒi",
        "S057_MTR": "hɑdɑbuɾ",
        "S060_RDG": "tʃ̑ɑk̚",
        "S061_RDG": "tʃ̑itɾɑŋ",
        "S061_MTR": "tʃ̑eɾtɑŋ",
        "S067_RDG": "gom",
        "S068_MTR": "mɑʲɾuŋ",
        "S070_MTR": "bɑʲgon",
        "S072_RDG": "d͡ʒəluk",
        "S072_MTR": "d͡ʒɑluk̚",
        "S075_RDG": "ɾɑjsun sɑqːɑj",
        "S078_MTR": "teŋɑ bɑjgun",
        "S079_RDG": "tʰutʃ̑i",
        "S081_RDG": "qɑqɑ, mɑqɑn",
        "S082_RDG": "pɨtɑm",
        "S085_RDG": "pitʃ̑í",
        "S085_MTR": "toʔ pɪtʃ̑i",
        "S090_RDG": "d͡ʒímiŋ",
        "S091_RDG": "pɾɨn",
        "S092_MTR": "kɪʔ",
        "S096_MTR": "tʃ̑oŋ sɑmɑɾɑ",
        "S099_MTR": "kɑʲ",
        "S100_RDG": "métʃ̑ɑkɑj, métʃ̑ɑtɑŋ",
        "S101_RDG": "qɑj sɑ́bɾɑ",
        "S102_RDG": "bɑbɑ, bɑbɾɑ",
        "S105_RDG": "pʰod͡ʒoŋ brɑ",
        "S105_MTR": "pʰod͡ʒoŋ bɾɑ",
        "S109_RDG": "métʃ̑ɑ sɑbɾɑ",
        "S111_RDG": "mítʃ̑ik bəɾɑ",
        "S114_RDG": "sʰɑn, din",
        "S118_RDG": "ɾɑŋsəɾi",
        "S120_MTR": "teʔ",
        "S121_RDG": "gɑpʰuŋ",
        "S122_RDG": "hɑt̚pəksɑ",
        "S125_RDG": "mɑʲtʃ̑ɑm",
        "S126_RDG": "pidam",
        "S130_RDG": "ɾɑ́nkɑj",
        "S132_RDG": "soŋqɑj",
        "S134_RDG": "tʃ̑ikɑj",
        "S137_MTR": "mukuŋgi, mukʰɑm",
        "S138_MTR": "d͡ʒɑnːɑ",
        "S140_RDG": "mɨlːɑ",
        "S142_RDG": "tʃ̑éŋɑ",
        "S145_RDG": "boqːɑ",
        "S148_RDG": "gosɑ",
        "S148_MTR": "",
        "S149_MTR": "tʃ̑ɑŋ",
        "S153_RDG": "bentʃ̑ek̚",
        "S154_RDG": "bekʰɾe",
        "S155_RDG": "ikɑj",
        "S156_MTR": "howgo",
        "S157_RDG": "ibid͡ʒəm",
        "S158_RDG": "ubid͡ʒəm",
        "S159_RDG": "gosɑn",
        "S160_MTR": "beɾgɑ",
        "S161_MTR": "dibinɑ d͡ʒɑŋtʃ̑ɑ",
        "S163_RDG": "tʃ̑ɨpən",
        "S163_MTR": "tʃ̑ipɑŋ",
        "S164_RDG": "pə́ŋqɑj",
        "S165_RDG": "dɨmdɑk̚",
        "S166_RDG": "sɑ́ʔɑ",
        "S168_MTR": "bukʰɨd͡ʒo",
        "S169_RDG": "ɾɨŋkɑj",
        "S170_MTR": "ɾiŋɑ nuŋd͡ʒoʔ",
        "S171_RDG": "guɾetɑ",
        "S172_MTR": "guɾɨd͡ʒo",
        "S173_MTR": "nuʔ",
        "S174_RDG": "ɾɑ́kʰu",
        "S176_MTR": "sɨd͡ʒo",
        "S178_RDG": "puɾitɑ",
        "S179_RDG": "ɾéŋɑmkɑj",
        "S180_RDG": "dɑɾtʰúŋkɑj",
        "S181_RDG": "ɾéŋ",
        "S182_MTR": "ɾɨʋɑ",
        "S183_RDG": "qɑni, bobɑj",
        "S183_MTR": "tep̚",
        "S184_MTR": "nɑd͡ʒoʔ",
        "S185_RDG": "nuk̚d͡ʒo",
        "S187_RDG": "nɑ́ŋ",
        "S189_RDG": "ɨ, u",
        "S190_MTR": "i",
        "S191_RDG": "t͡ʃiŋ",
        "S191_MTR": "t͡ʃɨŋ",
        "S192_RDG": "t͡ʃiŋ qɑmiŋ",
        "S192_MTR": "t͡ʃɨŋ kɑmkɑʲ",
        "S193_RDG": "nɑɾoŋ",
        "S194_MTR": "oɾoŋ",
    }

    prompts = read_tsv(PACKAGE / "prompt_review.tsv")
    assert [row["Source_Order_Item"] for row in prompts] == [
        f"S{item:03d}" for item in range(1, 195)
    ]
    assert {(row["PDF_Page"], row["Printed_Page"]) for row in prompts} == {
        ("22", "18"), ("23", "19"), ("24", "20"), ("25", "21"),
        ("26", "22"), ("27", "23"), ("28", "24"), ("29", "25"),
        ("30", "26"), ("31", "27"), ("32", "28"), ("33", "29"),
        ("34", "30")
    }
    assert {row["Status"] for row in prompts} == {"complete"}


def test_checkpoint_accounting_and_unresolved_ledger():
    manifest = json.loads((PACKAGE / "source_manifest.json").read_text(encoding="utf-8"))
    review = manifest["manual_review"]
    assert review["complete"] is True
    assert review["reviewed_cells"] == 388
    assert review["attested_cells"] == 387
    assert review["source_blank_cells"] == 1
    assert review["ambiguous_cells"] == 0
    assert review["illegible_cells"] == 0
    assert review["pending_cells"] == 0
    assert review["response_occurrences"] == 399
    assert review["installed_forms"] == 0
    assert review["reviewed_cells"] + review["pending_cells"] == 388
    assert review["chunk_sha256"] == {
        chunk.name: hashlib.sha256(chunk.read_bytes()).hexdigest()
        for chunk in CHUNKS
    }
    assert "100% cell-by-cell visual review" in review["declaration"]
    assert "never to supply, normalize, or verify" in review["declaration"]

    unresolved = read_tsv(PACKAGE / "unresolved_readings.tsv")
    assert unresolved == []


def test_post_entry_reconciliation_with_existing_installation():
    manual = manual_rows()
    manual_occurrences = {}
    for row in manual:
        if row["Cell_Status"] != "attested":
            continue
        item = int(row["Cell_Key"][1:4])
        lect = row["Cell_Key"][-3:]
        manual_occurrences[(item, lect)] = row["Manual_Form"].split(", ")

    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        installed = list(csv.reader(stream))
    assert len(installed) == 400
    assert Counter(row[0] for row in installed) == {
        "rabha_rongdani": 205,
        "rabha_maituri": 195,
    }

    installed_occurrences = {}
    for row in installed:
        _, item, lect, _ = row[10].split(":")
        if int(item) > 194 or row[2] == "no data":
            continue
        lect_id = "RDG" if lect == "rongdani" else "MTR"
        installed_occurrences.setdefault((int(item), lect_id), []).append(row[2])

    # This is a post-entry duplicate audit only. The source image, not this file, is
    # authoritative for the manual chunks; RECONCILIATION.md records that ordering.
    assert manual_occurrences.keys() == installed_occurrences.keys()
    differences = {
        key: (manual_occurrences[key], installed_occurrences[key])
        for key in manual_occurrences
        if manual_occurrences[key] != installed_occurrences[key]
    }
    assert differences == {
        (16, "RDG"): (["tʃ̑ɑskɑm", "tɑ́sikʰu"], ["tʃɑ̑skɑm", "tɑ́sikʰu"]),
        (17, "RDG"): (["tʃ̑ɑskoɾ"], ["tʃɑ̑skoɾ"]),
        (32, "RDG"): (["kɑŋkə", "tʃ̑ɑku"], ["kɑŋkə", "tʃɑ̑ku"]),
        (34, "RDG"): (["pɑgɑ"], ["pɑɡɑ"]),
        (38, "RDG"): (["tʃ̑ətʃ̑əkɑm"], ["tʃə̑tʃ̑əkɑm"]),
        (44, "RDG"): (["tʃ̑ɪ̃ kɑ"], ["tʃɪ̑ ̆kɑ"]),
        (44, "MTR"): (["tʃ̑kɑ"], ["tʃk̑ɑ"]),
        (45, "MTR"): (["tʃ̑kɑ d͡ʒoɾɑ"], ["tʃk̑ɑ d͡ʒoɾɑ"]),
        (52, "RDG"): (["hɑ́ŋtʃ̑eŋ"], ["hɑ́ŋtʃȇŋ"]),
        (60, "RDG"): (["tʃ̑ɑk̚"], ["tʃɑ̑k̚"]),
        (60, "MTR"): (["tʃ̑ɑk̚"], ["tʃɑ̑k̚"]),
        (61, "RDG"): (["tʃ̑itɾɑŋ"], ["tʃit̑ɾɑŋ"]),
        (61, "MTR"): (["tʃ̑eɾtɑŋ"], ["tʃȇɪtɑŋ"]),
        (67, "RDG"): (["gom"], ["ɡom"]),
        (67, "MTR"): (["gom"], ["ɡom"]),
        (68, "MTR"): (["mɑʲɾuŋ"], ["mɑʲɾun"]),
        (70, "MTR"): (["bɑʲgon"], ["bɑʲɡon"]),
        (76, "RDG"): (["pʰul gobʰi"], ["pʰul ɡobʰi"]),
        (76, "MTR"): (["pʰul gobi"], ["pʰul ɡobi"]),
        (77, "RDG"): (["bɑnd gobʰi"], ["bɑnd ɡobʰi"]),
        (77, "MTR"): (["bənd gobi"], ["bənd ɡobi"]),
        (78, "MTR"): (["teŋɑ bɑjgun"], ["teŋɑ bɑjɡun"]),
        (96, "MTR"): (["tʃ̑oŋ sɑmɑɾɑ"], ["tʃȏŋ sɑmɑɾɑ"]),
        (121, "RDG"): (["gɑpʰuŋ"], ["ɡɑpʰuŋ"]),
        (125, "RDG"): (["mɑʲtʃ̑ɑm"], ["mɑjtʃ̑ɑm"]),
        (134, "RDG"): (["tʃ̑ikɑj"], ["tʃik̑ɑj"]),
        (136, "MTR"): (["tʃ̑ɑbɾɑ bɑtɑm"], ["tʃɑ̑bɾɑ bɑtɑm"]),
        (137, "MTR"): (["mukuŋgi", "mukʰɑm"], ["mukuŋɡi", "mukʰɑm"]),
        (139, "RDG"): (["tʃ̑uŋɑ"], ["tʃȗŋɑ"]),
        (139, "MTR"): (["tʃ̑euŋɑ"], ["tʃeȗŋɑ"]),
        (142, "RDG"): (["tʃ̑éŋɑ"], ["tʃéŋ̑ɑ"]),
        (142, "MTR"): (["tʃ̑eŋɑ"], ["tʃȇŋɑ"]),
        (148, "RDG"): (["gosɑ"], ["ɡosɑ"]),
        (149, "RDG"): (["tʃ̑ɑŋ"], ["tʃɑ̑ŋ"]),
        (149, "MTR"): (["tʃ̑ɑŋ"], ["tʃɑ̑ŋ"]),
        (156, "MTR"): (["howgo"], ["howɡo"]),
        (158, "MTR"): (["howgo"], ["howɡo"]),
        (159, "RDG"): (["gosɑn"], ["ɡosɑn"]),
        (160, "RDG"): (["beɾgɑ"], ["beɾɡɑ"]),
        (160, "MTR"): (["beɾgɑ"], ["beɾɡɑ"]),
        (163, "RDG"): (["tʃ̑ɨpən"], ["tʃɨ̑pən"]),
        (163, "MTR"): (["tʃ̑ipɑŋ"], ["tʃip̑ɑŋ"]),
        (170, "RDG"): (["tʃ̑ikɑ́ŋqɑj"], ["tʃik̑ɑ́ŋqɑj"]),
        (171, "RDG"): (["guɾetɑ"], ["ɡuɾetɑ"]),
        (171, "MTR"): (["guɾitɑ"], ["ɡuɾitɑ"]),
        (172, "RDG"): (["guɾetojtɑ"], ["ɡuɾetojtɑ"]),
        (172, "MTR"): (["guɾɨd͡ʒo"], ["ɡuɾɨd͡ʒo"]),
    }
    exact_occurrences = sum(
        manual_form == installed_form
        for key in manual_occurrences
        for manual_form, installed_form in zip(
            manual_occurrences[key], installed_occurrences[key], strict=True
        )
    )
    assert exact_occurrences == 352


def test_census_records_sil_scope_topology_and_manual_checkpoint():
    census = CENSUS.read_text(encoding="utf-8")
    assert "ESR 2013-016 Rabha dialects of Meghalaya and Assam" in census
    assert "194 prompt rows" in census
    assert "388 conceptual cells" in census
    assert "S001-S194 / all 388 cells" in census
    assert "manual re-audit complete" in census
    assert "f690f404b793c601882b06940557a748e7932a9ca4afa28358fd75ca4396d02b" in census
    assert "Ahirwal" in census and "no SIL author, commission" in census
