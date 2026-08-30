"""Focused guards for the manual-only Kurux recovery."""

import csv
import hashlib
import json
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from segments.tokenizer import Tokenizer


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_kurux_2011_manual"
LINES_001_010 = PACKAGE / "manual_chunks/p039-items001-010-lines.tsv"
CELLS_001_010 = PACKAGE / "manual_chunks/p039-items001-010-cells.tsv"
LINES_011_020 = PACKAGE / "manual_chunks/p039-040-items011-020-lines.tsv"
CELLS_011_020 = PACKAGE / "manual_chunks/p039-040-items011-020-cells.tsv"
LINES_021_030 = PACKAGE / "manual_chunks/p040-items021-030-lines.tsv"
CELLS_021_030 = PACKAGE / "manual_chunks/p040-items021-030-cells.tsv"
LINES_031_040 = PACKAGE / "manual_chunks/p040-041-items031-040-lines.tsv"
CELLS_031_040 = PACKAGE / "manual_chunks/p040-041-items031-040-cells.tsv"
LINES_041_050 = PACKAGE / "manual_chunks/p041-042-items041-050-lines.tsv"
CELLS_041_050 = PACKAGE / "manual_chunks/p041-042-items041-050-cells.tsv"
LINES_051_060 = PACKAGE / "manual_chunks/p042-items051-060-lines.tsv"
CELLS_051_060 = PACKAGE / "manual_chunks/p042-items051-060-cells.tsv"
LINES_061_070 = PACKAGE / "manual_chunks/p042-043-items061-070-lines.tsv"
CELLS_061_070 = PACKAGE / "manual_chunks/p042-043-items061-070-cells.tsv"
LINES_071_080 = PACKAGE / "manual_chunks/p043-items071-080-lines.tsv"
CELLS_071_080 = PACKAGE / "manual_chunks/p043-items071-080-cells.tsv"
LINES_081_090 = PACKAGE / "manual_chunks/p043-044-items081-090-lines.tsv"
CELLS_081_090 = PACKAGE / "manual_chunks/p043-044-items081-090-cells.tsv"
LINES_091_100 = PACKAGE / "manual_chunks/p044-045-items091-100-lines.tsv"
CELLS_091_100 = PACKAGE / "manual_chunks/p044-045-items091-100-cells.tsv"
LINES_101_110 = PACKAGE / "manual_chunks/p045-items101-110-lines.tsv"
CELLS_101_110 = PACKAGE / "manual_chunks/p045-items101-110-cells.tsv"
LINES_111_120 = PACKAGE / "manual_chunks/p045-046-items111-120-lines.tsv"
CELLS_111_120 = PACKAGE / "manual_chunks/p045-046-items111-120-cells.tsv"
LINES_121_130 = PACKAGE / "manual_chunks/p046-items121-130-lines.tsv"
CELLS_121_130 = PACKAGE / "manual_chunks/p046-items121-130-cells.tsv"
LINES_131_140 = PACKAGE / "manual_chunks/p047-items131-140-lines.tsv"
CELLS_131_140 = PACKAGE / "manual_chunks/p047-items131-140-cells.tsv"
LINES_141_150 = PACKAGE / "manual_chunks/p047-048-items141-150-lines.tsv"
CELLS_141_150 = PACKAGE / "manual_chunks/p047-048-items141-150-cells.tsv"
LINES_151_160 = PACKAGE / "manual_chunks/p048-items151-160-lines.tsv"
CELLS_151_160 = PACKAGE / "manual_chunks/p048-items151-160-cells.tsv"
LINES_161_170 = PACKAGE / "manual_chunks/p048-049-items161-170-lines.tsv"
CELLS_161_170 = PACKAGE / "manual_chunks/p048-049-items161-170-cells.tsv"
LINES_171_180 = PACKAGE / "manual_chunks/p049-items171-180-lines.tsv"
CELLS_171_180 = PACKAGE / "manual_chunks/p049-items171-180-cells.tsv"
LINES_181_190 = PACKAGE / "manual_chunks/p049-050-items181-190-lines.tsv"
CELLS_181_190 = PACKAGE / "manual_chunks/p049-050-items181-190-cells.tsv"
LINES_191_200 = PACKAGE / "manual_chunks/p050-items191-200-lines.tsv"
CELLS_191_200 = PACKAGE / "manual_chunks/p050-items191-200-cells.tsv"
LINES_201_210 = PACKAGE / "manual_chunks/p051-items201-210-lines.tsv"
CELLS_201_210 = PACKAGE / "manual_chunks/p051-items201-210-cells.tsv"
LINES_211_220 = PACKAGE / "manual_chunks/p051-052-items211-220-lines.tsv"
CELLS_211_220 = PACKAGE / "manual_chunks/p051-052-items211-220-cells.tsv"
LINES_221_230 = PACKAGE / "manual_chunks/p052-items221-230-lines.tsv"
CELLS_221_230 = PACKAGE / "manual_chunks/p052-items221-230-cells.tsv"
LINES_231_240 = PACKAGE / "manual_chunks/p052-053-items231-240-lines.tsv"
CELLS_231_240 = PACKAGE / "manual_chunks/p052-053-items231-240-cells.tsv"
LINES_241_250 = PACKAGE / "manual_chunks/p053-items241-250-lines.tsv"
CELLS_241_250 = PACKAGE / "manual_chunks/p053-items241-250-cells.tsv"
LINES_251_260 = PACKAGE / "manual_chunks/p053-054-items251-260-lines.tsv"
CELLS_251_260 = PACKAGE / "manual_chunks/p053-054-items251-260-cells.tsv"
LINES_261_270 = PACKAGE / "manual_chunks/p054-055-items261-270-lines.tsv"
CELLS_261_270 = PACKAGE / "manual_chunks/p054-055-items261-270-cells.tsv"
LINES_271_280 = PACKAGE / "manual_chunks/p055-items271-280-lines.tsv"
CELLS_271_280 = PACKAGE / "manual_chunks/p055-items271-280-cells.tsv"
LINES_281_290 = PACKAGE / "manual_chunks/p055-056-items281-290-lines.tsv"
CELLS_281_290 = PACKAGE / "manual_chunks/p055-056-items281-290-cells.tsv"
LINES_291_300 = PACKAGE / "manual_chunks/p056-items291-300-lines.tsv"
CELLS_291_300 = PACKAGE / "manual_chunks/p056-items291-300-cells.tsv"
LINES_301_307 = PACKAGE / "manual_chunks/p056-057-items301-307-lines.tsv"
CELLS_301_307 = PACKAGE / "manual_chunks/p056-057-items301-307-cells.tsv"
MANIFEST = PACKAGE / "source_manifest.json"
POST_FREEZE_SCRIPT = PACKAGE / "build_post_freeze_package.py"
POST_FREEZE_MANIFEST = PACKAGE / "post_freeze_manifest.json"
RECONCILIATION = PACKAGE / "reconciliation.tsv"
STAGING_AUDIT = PACKAGE / "staging_audit.tsv"
STAGED_FORMS = PACKAGE / "staged_forms.csv"
SITE_METADATA = PACKAGE / "site_metadata.tsv"
REFERENCE_METADATA = PACKAGE / "reference_metadata.json"
EXCLUSION_POLICY = PACKAGE / "exclusion_policy.json"
SOUND_INVENTORY = PACKAGE / "sound_inventory.tsv"
SOUND_PROFILE = PACKAGE / "sound_profile.txt"
SOUND_DECISIONS = PACKAGE / "sound_profile_decisions.json"
INSTALLED_FORMS = ROOT / "data/other/forms/20260826-sil-kurux.csv"
SHARED_PROFILE = ROOT / "conversion/sil-kurux.txt"
DIALECT_REGISTRY = ROOT / "cldf/dialects.csv"
SOURCES_BIB = ROOT / "cldf/sources.bib"
BUILD_SCRIPT = ROOT / "make_cldf.py"
SHARED_INTEGRATION_MANIFEST = PACKAGE / "shared_integration_manifest.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_generator_is_exact_and_reproducible():
    subprocess.run([sys.executable, str(PACKAGE / "build_manual_chunks.py")], check=True)
    lines_1 = read_tsv(LINES_001_010)
    cells_1 = read_tsv(CELLS_001_010)
    lines_2 = read_tsv(LINES_011_020)
    cells_2 = read_tsv(CELLS_011_020)
    lines_3 = read_tsv(LINES_021_030)
    cells_3 = read_tsv(CELLS_021_030)
    lines_4 = read_tsv(LINES_031_040)
    cells_4 = read_tsv(CELLS_031_040)
    lines_5 = read_tsv(LINES_041_050)
    cells_5 = read_tsv(CELLS_041_050)
    lines_6 = read_tsv(LINES_051_060)
    cells_6 = read_tsv(CELLS_051_060)
    lines_7 = read_tsv(LINES_061_070)
    cells_7 = read_tsv(CELLS_061_070)
    lines_8 = read_tsv(LINES_071_080)
    cells_8 = read_tsv(CELLS_071_080)
    lines_9 = read_tsv(LINES_081_090)
    cells_9 = read_tsv(CELLS_081_090)
    lines_10 = read_tsv(LINES_091_100)
    cells_10 = read_tsv(CELLS_091_100)
    lines_11 = read_tsv(LINES_101_110)
    cells_11 = read_tsv(CELLS_101_110)
    lines_12 = read_tsv(LINES_111_120)
    cells_12 = read_tsv(CELLS_111_120)
    lines_13 = read_tsv(LINES_121_130)
    cells_13 = read_tsv(CELLS_121_130)
    lines_14 = read_tsv(LINES_131_140)
    cells_14 = read_tsv(CELLS_131_140)
    lines_15 = read_tsv(LINES_141_150)
    cells_15 = read_tsv(CELLS_141_150)
    lines_16 = read_tsv(LINES_151_160)
    cells_16 = read_tsv(CELLS_151_160)
    lines_17 = read_tsv(LINES_161_170)
    cells_17 = read_tsv(CELLS_161_170)
    lines_18 = read_tsv(LINES_171_180)
    cells_18 = read_tsv(CELLS_171_180)
    lines_19 = read_tsv(LINES_181_190)
    cells_19 = read_tsv(CELLS_181_190)
    lines_20 = read_tsv(LINES_191_200)
    cells_20 = read_tsv(CELLS_191_200)
    lines_21 = read_tsv(LINES_201_210)
    cells_21 = read_tsv(CELLS_201_210)
    lines_22 = read_tsv(LINES_211_220)
    cells_22 = read_tsv(CELLS_211_220)
    lines_23 = read_tsv(LINES_221_230)
    cells_23 = read_tsv(CELLS_221_230)
    lines_24 = read_tsv(LINES_231_240)
    cells_24 = read_tsv(CELLS_231_240)
    lines_25 = read_tsv(LINES_241_250)
    cells_25 = read_tsv(CELLS_241_250)
    lines_26 = read_tsv(LINES_251_260)
    cells_26 = read_tsv(CELLS_251_260)
    lines_27 = read_tsv(LINES_261_270)
    cells_27 = read_tsv(CELLS_261_270)
    lines_28 = read_tsv(LINES_271_280)
    cells_28 = read_tsv(CELLS_271_280)
    lines_29 = read_tsv(LINES_281_290)
    cells_29 = read_tsv(CELLS_281_290)
    lines_30 = read_tsv(LINES_291_300)
    cells_30 = read_tsv(CELLS_291_300)
    lines_31 = read_tsv(LINES_301_307)
    cells_31 = read_tsv(CELLS_301_307)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(lines_1) == 38
    assert len(cells_1) == 61
    assert len(lines_2) == 34
    assert len(cells_2) == 60
    assert len(lines_3) == 33
    assert len(cells_3) == 63
    assert len(lines_4) == 31
    assert len(cells_4) == 60
    assert len(lines_5) == 37
    assert len(cells_5) == 63
    assert len(lines_6) == 33
    assert len(cells_6) == 60
    assert len(lines_7) == 35
    assert len(cells_7) == 61
    assert len(lines_8) == 32
    assert len(cells_8) == 62
    assert len(lines_9) == 29
    assert len(cells_9) == 60
    assert len(lines_10) == 34
    assert len(cells_10) == 61
    assert len(lines_11) == 30
    assert len(cells_11) == 60
    assert len(lines_12) == 42
    assert len(cells_12) == 64
    assert len(lines_13) == 33
    assert len(cells_13) == 60
    assert len(lines_14) == 36
    assert len(cells_14) == 61
    assert len(lines_15) == 33
    assert len(cells_15) == 63
    assert len(lines_16) == 25
    assert len(cells_16) == 60
    assert len(lines_17) == 33
    assert len(cells_17) == 61
    assert len(lines_18) == 28
    assert len(cells_18) == 60
    assert len(lines_19) == 28
    assert len(cells_19) == 60
    assert len(lines_20) == 30
    assert len(cells_20) == 60
    assert len(lines_21) == 31
    assert len(cells_21) == 62
    assert len(lines_22) == 33
    assert len(cells_22) == 60
    assert len(lines_23) == 26
    assert len(cells_23) == 60
    assert len(lines_24) == 19
    assert len(cells_24) == 60
    assert len(lines_25) == 32
    assert len(cells_25) == 62
    assert len(lines_26) == 40
    assert len(cells_26) == 60
    assert len(lines_27) == 40
    assert len(cells_27) == 60
    assert len(lines_28) == 36
    assert len(cells_28) == 61
    assert len(lines_29) == 30
    assert len(cells_29) == 62
    assert len(lines_30) == 30
    assert len(cells_30) == 60
    assert len(lines_31) == 17
    assert len(cells_31) == 42
    assert manifest["response_lines"] == 988
    assert manifest["expanded_cell_rows"] == 1869
    assert manifest["conceptual_cells"] == 1842
    chunks = {chunk["id"]: chunk for chunk in manifest["chunks"]}
    assert sha256(LINES_001_010) == chunks["items001-010"]["manual_lines_sha256"]
    assert sha256(CELLS_001_010) == chunks["items001-010"]["expanded_cells_sha256"]
    assert sha256(LINES_011_020) == chunks["items011-020"]["manual_lines_sha256"]
    assert sha256(CELLS_011_020) == chunks["items011-020"]["expanded_cells_sha256"]
    assert sha256(LINES_021_030) == chunks["items021-030"]["manual_lines_sha256"]
    assert sha256(CELLS_021_030) == chunks["items021-030"]["expanded_cells_sha256"]
    assert sha256(LINES_031_040) == chunks["items031-040"]["manual_lines_sha256"]
    assert sha256(CELLS_031_040) == chunks["items031-040"]["expanded_cells_sha256"]
    assert sha256(LINES_041_050) == chunks["items041-050"]["manual_lines_sha256"]
    assert sha256(CELLS_041_050) == chunks["items041-050"]["expanded_cells_sha256"]
    assert chunks["items041-050"]["confidence_counts"] == {"high": 37}
    assert sha256(LINES_051_060) == chunks["items051-060"]["manual_lines_sha256"]
    assert sha256(CELLS_051_060) == chunks["items051-060"]["expanded_cells_sha256"]
    assert chunks["items051-060"]["confidence_counts"] == {"high": 33}
    assert sha256(LINES_061_070) == chunks["items061-070"]["manual_lines_sha256"]
    assert sha256(CELLS_061_070) == chunks["items061-070"]["expanded_cells_sha256"]
    assert chunks["items061-070"]["confidence_counts"] == {"high": 35}
    assert sha256(LINES_071_080) == chunks["items071-080"]["manual_lines_sha256"]
    assert sha256(CELLS_071_080) == chunks["items071-080"]["expanded_cells_sha256"]
    assert chunks["items071-080"]["confidence_counts"] == {"high": 32}
    assert sha256(LINES_081_090) == chunks["items081-090"]["manual_lines_sha256"]
    assert sha256(CELLS_081_090) == chunks["items081-090"]["expanded_cells_sha256"]
    assert chunks["items081-090"]["confidence_counts"] == {"high": 29}
    assert sha256(LINES_091_100) == chunks["items091-100"]["manual_lines_sha256"]
    assert sha256(CELLS_091_100) == chunks["items091-100"]["expanded_cells_sha256"]
    assert chunks["items091-100"]["confidence_counts"] == {"high": 34}
    assert sha256(LINES_101_110) == chunks["items101-110"]["manual_lines_sha256"]
    assert sha256(CELLS_101_110) == chunks["items101-110"]["expanded_cells_sha256"]
    assert chunks["items101-110"]["confidence_counts"] == {"high": 30}
    assert sha256(LINES_111_120) == chunks["items111-120"]["manual_lines_sha256"]
    assert sha256(CELLS_111_120) == chunks["items111-120"]["expanded_cells_sha256"]
    assert chunks["items111-120"]["confidence_counts"] == {"high": 42}
    assert sha256(LINES_121_130) == chunks["items121-130"]["manual_lines_sha256"]
    assert sha256(CELLS_121_130) == chunks["items121-130"]["expanded_cells_sha256"]
    assert chunks["items121-130"]["confidence_counts"] == {"high": 33}
    assert sha256(LINES_131_140) == chunks["items131-140"]["manual_lines_sha256"]
    assert sha256(CELLS_131_140) == chunks["items131-140"]["expanded_cells_sha256"]
    assert chunks["items131-140"]["confidence_counts"] == {"high": 36}
    assert sha256(LINES_141_150) == chunks["items141-150"]["manual_lines_sha256"]
    assert sha256(CELLS_141_150) == chunks["items141-150"]["expanded_cells_sha256"]
    assert chunks["items141-150"]["confidence_counts"] == {"high": 33}
    assert sha256(LINES_151_160) == chunks["items151-160"]["manual_lines_sha256"]
    assert sha256(CELLS_151_160) == chunks["items151-160"]["expanded_cells_sha256"]
    assert chunks["items151-160"]["confidence_counts"] == {"high": 25}
    assert sha256(LINES_161_170) == chunks["items161-170"]["manual_lines_sha256"]
    assert sha256(CELLS_161_170) == chunks["items161-170"]["expanded_cells_sha256"]
    assert chunks["items161-170"]["confidence_counts"] == {"high": 33}
    assert sha256(LINES_171_180) == chunks["items171-180"]["manual_lines_sha256"]
    assert sha256(CELLS_171_180) == chunks["items171-180"]["expanded_cells_sha256"]
    assert chunks["items171-180"]["confidence_counts"] == {"high": 28}
    assert sha256(LINES_181_190) == chunks["items181-190"]["manual_lines_sha256"]
    assert sha256(CELLS_181_190) == chunks["items181-190"]["expanded_cells_sha256"]
    assert chunks["items181-190"]["confidence_counts"] == {"high": 28}
    assert sha256(LINES_191_200) == chunks["items191-200"]["manual_lines_sha256"]
    assert sha256(CELLS_191_200) == chunks["items191-200"]["expanded_cells_sha256"]
    assert chunks["items191-200"]["confidence_counts"] == {"high": 30}
    assert sha256(LINES_201_210) == chunks["items201-210"]["manual_lines_sha256"]
    assert sha256(CELLS_201_210) == chunks["items201-210"]["expanded_cells_sha256"]
    assert chunks["items201-210"]["confidence_counts"] == {"high": 31}
    assert sha256(LINES_211_220) == chunks["items211-220"]["manual_lines_sha256"]
    assert sha256(CELLS_211_220) == chunks["items211-220"]["expanded_cells_sha256"]
    assert chunks["items211-220"]["confidence_counts"] == {"high": 33}
    assert sha256(LINES_221_230) == chunks["items221-230"]["manual_lines_sha256"]
    assert sha256(CELLS_221_230) == chunks["items221-230"]["expanded_cells_sha256"]
    assert chunks["items221-230"]["confidence_counts"] == {"high": 26}
    assert sha256(LINES_231_240) == chunks["items231-240"]["manual_lines_sha256"]
    assert sha256(CELLS_231_240) == chunks["items231-240"]["expanded_cells_sha256"]
    assert chunks["items231-240"]["confidence_counts"] == {"high": 19}
    assert sha256(LINES_241_250) == chunks["items241-250"]["manual_lines_sha256"]
    assert sha256(CELLS_241_250) == chunks["items241-250"]["expanded_cells_sha256"]
    assert chunks["items241-250"]["confidence_counts"] == {"high": 32}
    assert sha256(LINES_251_260) == chunks["items251-260"]["manual_lines_sha256"]
    assert sha256(CELLS_251_260) == chunks["items251-260"]["expanded_cells_sha256"]
    assert chunks["items251-260"]["confidence_counts"] == {"high": 40}
    assert sha256(LINES_261_270) == chunks["items261-270"]["manual_lines_sha256"]
    assert sha256(CELLS_261_270) == chunks["items261-270"]["expanded_cells_sha256"]
    assert chunks["items261-270"]["confidence_counts"] == {"high": 40}
    assert sha256(LINES_271_280) == chunks["items271-280"]["manual_lines_sha256"]
    assert sha256(CELLS_271_280) == chunks["items271-280"]["expanded_cells_sha256"]
    assert chunks["items271-280"]["confidence_counts"] == {"high": 36}
    assert sha256(LINES_281_290) == chunks["items281-290"]["manual_lines_sha256"]
    assert sha256(CELLS_281_290) == chunks["items281-290"]["expanded_cells_sha256"]
    assert chunks["items281-290"]["confidence_counts"] == {"high": 30}
    assert sha256(LINES_291_300) == chunks["items291-300"]["manual_lines_sha256"]
    assert sha256(CELLS_291_300) == chunks["items291-300"]["expanded_cells_sha256"]
    assert chunks["items291-300"]["confidence_counts"] == {"high": 30}
    assert sha256(LINES_301_307) == chunks["items301-307"]["manual_lines_sha256"]
    assert sha256(CELLS_301_307) == chunks["items301-307"]["expanded_cells_sha256"]
    assert chunks["items301-307"]["confidence_counts"] == {"high": 17}


def test_every_item_has_all_six_sites_and_only_one_duplicate_coordinate():
    cells = (read_tsv(CELLS_001_010) + read_tsv(CELLS_011_020) +
             read_tsv(CELLS_021_030) + read_tsv(CELLS_031_040) +
             read_tsv(CELLS_041_050) + read_tsv(CELLS_051_060) +
             read_tsv(CELLS_061_070) + read_tsv(CELLS_071_080))
    cells += read_tsv(CELLS_081_090)
    cells += read_tsv(CELLS_091_100)
    cells += read_tsv(CELLS_101_110)
    cells += read_tsv(CELLS_111_120)
    cells += read_tsv(CELLS_121_130)
    cells += read_tsv(CELLS_131_140)
    cells += read_tsv(CELLS_141_150)
    cells += read_tsv(CELLS_151_160)
    cells += read_tsv(CELLS_161_170)
    cells += read_tsv(CELLS_171_180)
    cells += read_tsv(CELLS_181_190)
    cells += read_tsv(CELLS_191_200)
    cells += read_tsv(CELLS_201_210)
    cells += read_tsv(CELLS_211_220)
    cells += read_tsv(CELLS_221_230)
    cells += read_tsv(CELLS_231_240)
    cells += read_tsv(CELLS_241_250)
    cells += read_tsv(CELLS_251_260)
    cells += read_tsv(CELLS_261_270)
    cells += read_tsv(CELLS_271_280)
    cells += read_tsv(CELLS_281_290)
    cells += read_tsv(CELLS_291_300)
    cells += read_tsv(CELLS_301_307)
    by_item = defaultdict(set)
    counts = Counter()
    for row in cells:
        item = int(row["item"])
        by_item[item].add(row["site_code"])
        counts[(item, row["site_code"])] += 1
    assert set(by_item) == set(range(1, 308))
    assert all(codes == set("ABCDE0") for codes in by_item.values())
    assert len(counts) == 1842
    assert {coordinate: count for coordinate, count in counts.items() if count > 1} == {
        (3, "A"): 2,
        (30, "B"): 2,
        (30, "D"): 2,
        (30, "E"): 2,
        (50, "B"): 2,
        (50, "C"): 2,
        (50, "D"): 2,
        (66, "B"): 2,
        (71, "A"): 2,
        (76, "D"): 2,
        (96, "A"): 2,
        (114, "A"): 2,
        (118, "D"): 2,
        (119, "B"): 2,
        (120, "0"): 2,
        (131, "A"): 2,
        (147, "E"): 2,
        (149, "C"): 2,
        (150, "E"): 2,
        (165, "C"): 2,
        (202, "D"): 2,
        (202, "E"): 2,
        (245, "A"): 2,
        (245, "D"): 2,
        (274, "D"): 2,
        (283, "D"): 2,
        (284, "D"): 2,
    }


def test_exact_dispositions_roles_and_variants():
    cells = (read_tsv(CELLS_001_010) + read_tsv(CELLS_011_020) +
             read_tsv(CELLS_021_030) + read_tsv(CELLS_031_040) +
             read_tsv(CELLS_041_050) + read_tsv(CELLS_051_060) +
             read_tsv(CELLS_061_070) + read_tsv(CELLS_071_080))
    cells += read_tsv(CELLS_081_090)
    cells += read_tsv(CELLS_091_100)
    cells += read_tsv(CELLS_101_110)
    cells += read_tsv(CELLS_111_120)
    cells += read_tsv(CELLS_121_130)
    cells += read_tsv(CELLS_131_140)
    cells += read_tsv(CELLS_141_150)
    cells += read_tsv(CELLS_151_160)
    cells += read_tsv(CELLS_161_170)
    cells += read_tsv(CELLS_171_180)
    cells += read_tsv(CELLS_181_190)
    cells += read_tsv(CELLS_191_200)
    cells += read_tsv(CELLS_201_210)
    cells += read_tsv(CELLS_211_220)
    cells += read_tsv(CELLS_221_230)
    cells += read_tsv(CELLS_231_240)
    cells += read_tsv(CELLS_241_250)
    cells += read_tsv(CELLS_251_260)
    cells += read_tsv(CELLS_261_270)
    cells += read_tsv(CELLS_271_280)
    cells += read_tsv(CELLS_281_290)
    cells += read_tsv(CELLS_291_300)
    cells += read_tsv(CELLS_301_307)
    assert Counter(row["status"] for row in cells) == {
        "attested": 1661, "blank": 136, "not_used": 72
    }
    assert Counter((row["role"], row["status"]) for row in cells) == {
        ("target", "attested"): 1365,
        ("target", "blank"): 136,
        ("target", "not_used"): 60,
        ("control", "attested"): 296,
        ("control", "not_used"): 12,
    }
    blanks = {(int(row["item"]), row["site_code"]) for row in cells
              if row["status"] == "blank"}
    assert blanks == {
        (1, "B"), (9, "A"), (10, "A"), (11, "A"), (11, "D"), (11, "E"),
        (12, "A"), (15, "A"), (21, "A"), (23, "A"), (28, "A"), (30, "A"),
        (33, "A"), (34, "A"), (37, "A"),
        (42, "A"), (44, "A"), (48, "A"), (50, "A"),
        (51, "A"), (55, "A"), (56, "A"), (56, "E"), (57, "A"), (58, "A"),
        (59, "A"), (59, "B"),
        (67, "A"), (68, "A"), (69, "A"), (70, "A"),
        (72, "A"), (75, "A"), (76, "A"), (77, "A"), (79, "A"),
        (85, "A"), (86, "A"), (90, "A"),
        (91, "A"), (92, "A"), (93, "A"), (94, "A"), (95, "A"),
        (101, "A"), (106, "A"),
        (112, "A"), (116, "A"), (117, "A"),
        (122, "A"), (122, "B"), (125, "A"), (127, "A"),
        (136, "A"), (137, "A"), (138, "A"), (139, "A"), (140, "A"),
        (145, "A"), (147, "A"), (148, "A"), (149, "A"),
        (153, "A"), (154, "A"), (158, "A"),
        (162, "A"), (164, "A"), (168, "A"), (169, "A"),
        (174, "A"), (179, "A"), (180, "A"),
        (181, "A"), (184, "A"), (185, "A"), (186, "A"),
        (187, "A"), (187, "B"),
        (192, "A"), (195, "A"), (196, "A"), (198, "A"),
        (199, "A"), (200, "A"),
        (201, "A"), (202, "A"), (203, "A"), (203, "B"),
        (204, "A"), (204, "B"), (205, "A"), (206, "A"),
        (207, "A"), (208, "A"), (209, "A"), (209, "E"),
        (210, "A"),
        (211, "A"), (212, "A"), (213, "A"), (215, "A"),
        (216, "A"), (217, "A"), (218, "A"), (218, "B"),
        (219, "A"),
        (221, "A"), (223, "A"), (223, "E"),
        (224, "A"), (224, "B"), (224, "E"),
        (239, "A"),
        (241, "A"), (250, "A"),
        (251, "A"), (252, "A"), (253, "A"), (254, "A"),
        (255, "A"), (255, "C"), (256, "A"), (257, "A"),
        (260, "A"),
        (261, "A"), (262, "A"), (263, "A"), (264, "A"),
        (265, "A"), (266, "A"), (267, "A"),
        (275, "A"), (275, "D"),
        (288, "A"), (289, "A"),
        (294, "A"),
    }
    not_used = {(int(row["item"]), row["site_code"]) for row in cells
                if row["status"] == "not_used"}
    assert not_used == {
        (item, code) for item in (31, 74, 107, 124, 152, 163, 171, 194, 240, 247, 301, 306)
        for code in "ABCDE0"
    }
    variants = [(int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
                if row["item"] == "3" and row["site_code"] == "A"]
    assert variants == [(1, "t͜ʃʰɛnd", 1), (2, "t͜ʃʰɛnd", 2)]
    for site_code in "BDE":
        noon_variants = [
            (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
            if row["item"] == "30" and row["site_code"] == site_code
        ]
        assert noon_variants == [(1, "dupar", 1), (2, "dupar", 2)]
    for site_code in "BCD":
        jackfruit_variants = [
            (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
            if row["item"] == "50" and row["site_code"] == site_code
        ]
        assert jackfruit_variants == [(1, "gatʰaɽa", 1), (2, "gatʰaɽa", 2)]
    pepper_variants = [
        (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
        if row["item"] == "66" and row["site_code"] == "B"
    ]
    assert pepper_variants == [(1, "marit͜ʃa", 1), (2, "marit͜ʃa", 2)]
    monkey_variants = [
        (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
        if row["item"] == "71" and row["site_code"] == "A"
    ]
    assert monkey_variants == [(1, "bəndər", 1), (2, "bəndər", 2)]
    turtle_variants = [
        (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
        if row["item"] == "76" and row["site_code"] == "D"
    ]
    assert turtle_variants == [(1, "katʃʰu̯a", 1), (2, "ɛkːa", 2)]
    spider_variants = [
        (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
        if row["item"] == "96" and row["site_code"] == "A"
    ]
    assert spider_variants == [(1, "məkərɛ", 1), (2, "məkərɛ", 2)]
    expected_variants = {
        (114, "A"): [(1, "ɛŋgul", 1), (2, "ɛŋgul", 2)],
        (118, "D"): [(1, "togra", 1), (2, "xotʃol", 2)],
        (119, "B"): [(1, "tʃarbi", 1), (2, "nɛta", 2)],
        (120, "0"): [(1, "tʃamra", 1), (2, "tʃamra", 2)],
    }
    for (item, site_code), expected in expected_variants.items():
        actual = [
            (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
            if row["item"] == str(item) and row["site_code"] == site_code
        ]
        assert actual == expected
    mother_variants = [
        (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
        if row["item"] == "131" and row["site_code"] == "A"
    ]
    assert mother_variants == [(1, "jo", 1), (2, "ijo", 2)]
    later_variants = {
        (147, "E"): [(1, "pɛtʃi", 1), (2, "dɛal", 2)],
        (149, "C"): [(1, "duʃa", 1), (3, "kɔmbol", 2)],
        (150, "E"): [(1, "mutdi", 1), (2, "aŋti", 2)],
    }
    for (item, site_code), expected in later_variants.items():
        actual = [
            (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
            if row["item"] == str(item) and row["site_code"] == site_code
        ]
        assert actual == expected
    fire_variants = [
        (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
        if row["item"] == "165" and row["site_code"] == "C"
    ]
    assert fire_variants == [(1, "tʃitʃʰi", 1), (2, "tʃitʃʰi", 2)]
    for site_code in "DE":
        dance_variants = [
            (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
            if row["item"] == "202" and row["site_code"] == site_code
        ]
        assert dance_variants == [(1, "bɛtʃ", 1), (2, "nal", 2)]
    small_a_variants = [
        (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
        if row["item"] == "245" and row["site_code"] == "A"
    ]
    assert small_a_variants == [(1, "tʃukːɛ", 1), (2, "tʃukːɛ", 2)]
    small_d_variants = [
        (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
        if row["item"] == "245" and row["site_code"] == "D"
    ]
    assert small_d_variants == [(1, "ʃanːi", 1), (2, "tʃʰotɛ", 2)]
    good_d_variants = [
        (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
        if row["item"] == "274" and row["site_code"] == "D"
    ]
    assert good_d_variants == [(1, "korɛ", 1), (2, "bʰalo", 2)]
    right_d_variants = [
        (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
        if row["item"] == "283" and row["site_code"] == "D"
    ]
    assert right_d_variants == [(1, "mari", 1), (2, "tina", 2)]
    left_d_variants = [
        (int(row["group"]), row["form"], int(row["site_variant"])) for row in cells
        if row["item"] == "284" and row["site_code"] == "D"
    ]
    assert left_d_variants == [(1, "lɛŋa", 1), (2, "dɛbːa", 2)]


def test_forms_are_nfc_and_preserve_difficult_visual_readings():
    cells = (read_tsv(CELLS_001_010) + read_tsv(CELLS_011_020) +
             read_tsv(CELLS_021_030) + read_tsv(CELLS_031_040) +
             read_tsv(CELLS_041_050) + read_tsv(CELLS_051_060) +
             read_tsv(CELLS_061_070) + read_tsv(CELLS_071_080))
    cells += read_tsv(CELLS_081_090)
    cells += read_tsv(CELLS_091_100)
    cells += read_tsv(CELLS_101_110)
    cells += read_tsv(CELLS_111_120)
    cells += read_tsv(CELLS_121_130)
    cells += read_tsv(CELLS_131_140)
    cells += read_tsv(CELLS_141_150)
    cells += read_tsv(CELLS_151_160)
    cells += read_tsv(CELLS_161_170)
    cells += read_tsv(CELLS_171_180)
    cells += read_tsv(CELLS_181_190)
    cells += read_tsv(CELLS_191_200)
    cells += read_tsv(CELLS_201_210)
    cells += read_tsv(CELLS_211_220)
    cells += read_tsv(CELLS_221_230)
    cells += read_tsv(CELLS_231_240)
    cells += read_tsv(CELLS_241_250)
    cells += read_tsv(CELLS_251_260)
    cells += read_tsv(CELLS_261_270)
    cells += read_tsv(CELLS_271_280)
    cells += read_tsv(CELLS_281_290)
    cells += read_tsv(CELLS_291_300)
    cells += read_tsv(CELLS_301_307)
    forms = {row["form"] for row in cells}
    for row in cells:
        assert row["form"] == unicodedata.normalize("NFC", row["form"])
        assert "�" not in row["form"]
        assert not any(0xE000 <= ord(char) <= 0xF8FF for char in row["form"])
        if row["status"] == "attested":
            assert row["form"]
        else:
            assert not row["form"]
    assert {"məⁱjə", "biɽi", "ʃuɹd͜ʒo", "t͜ʃɛ̃p", "ɾamdʰanu",
            "birt͜ʃʰi", "mɛgʰ gɔɽd͜ʒon", "pahaɽ", "kʰɛr", "xatd͜ʒɛ",
            "maʈi", "kʰəd", "pakʰɛna", "t͜ʃʰɛlkul", "t͜ʃʰutɛ", "ɪnːa",
            "ɪnːe", "ɪnːɛ", "gɔtokal / kalkɛ", "mɛⁱnɛ", "bat͜ʃʰar",
            "bɔt͜ʃʰor", "ʊlːa", "ʊlːɛ", "pʰɛⁱri"} <= forms
    assert {"mɛxɛb", "xɛʃɛ", "t͜ʃal", "tɛkʰil", "d͜ʒinhor", "d͜ʒinxor",
            "bʰuʈʈa", "aluə̯", "pʼulkopi", "badʰakopi", "bɛndɛkobi"} <= forms
    assert {"bai̯gun", "manːɛ", "gat͜ʃʰ", "daɽa", "ɖal", "ɛtkʰɛ / ɛtxɛ",
            "att͜ʃʰɛ", "kʰɛnd͜ʒə", "xand͜ʒpa", "gatʰaɽa", "gatʰɽa"} <= forms
    assert {"narikɛl", "kɔla", "ʈatxa", "tɛtkɛ / tɛtxɛ", "püp", "bid͜ʒ",
            "bit͜ʃi", "kuʃari", "t͜ʃunːa", "t͜ʃunːɛ", "borɔɛ", "dudʰi"} <= forms
    assert {"bɛⁱk", "lɔbon / nun", "pĩjɛd͜ʒ", "mɛrt͜ʃɛⁱ", "marit͜ʃa",
            "bagʰ", "bʰaluk", "hɛrin"} <= forms
    assert {"bəndər", "kʰɛrha", "kʰɔrgoʃ", "tɛtiŋga", "katʃʰu̯a",
            "katʃʰu̯ɛ", "ɛkːa", "kɔttʃʰop", "bæŋ", "ɛlːɛ", "aɖːo"} <= forms
    assert {"məŋkʰɛ", "mɛrɛk", "maraŋ", "ʃiŋ", "lɛdʒ", "tʃʰagol",
            "oʃga", "osga", "kʰɛr", "ɖim", "indʒo", "matʃʰ"} <= forms
    assert {"pakʰi", "potʃgo", "iʃuŋ potʃgo", "bʰau̯ro", "mou̯matʃʰi",
            "məkərɛ", "bʰuʃɛri", "kukːu", "matʰa", "galːɛ", "tʃɛrɛ"} <= forms
    assert {"xɛʃɛr", "tʃutːi", "xanːɛ", "mũĩ", "kʰɛbdɛ", "galːɛ",
            "tɛrxɛ", "dʒib", "palːɛ", "dãt"} <= forms
    assert {"xɛtkʰɛ", "xɛtkʰatala", "hatɛr tola", "ɛŋgul", "orokʰ",
            "xɛtdɛ", "kotʃʰol", "haɽ", "tʃɛrbi", "nɛta", "tʃʰɛmri"} <= forms
    assert {"xɛ̃ʃo", "rɔkto", "ɛrtʃɛrna", "garmaɽa", "gʰam", "pitʰ",
            "ga / dɛho", "mɛdʒ", "kurukʰ", "mukːɛ", "ɛlːi"} <= forms
    assert {"metɛs", "xatdɛʃ", "tʃʰɛlɛ", "ukʰos", "kukoi̯", "mɛjɛ",
            "bɔɽo bʰai̯ / dada", "dai̯", "bɔɽo bon / didi", "iŋgris",
            "tʃʰoʈo bʰai̯", "ʃaŋxi", "bondʰu"} <= forms
    assert {"lɛⁱn", "padːa", "baɽi / gʰor", "dɔrdʒa", "tʃʰɛⁱn",
            "pɛtʃi", "dɛal", "baliʃ", "kamɽa", "kɔmbol", "əŋti"} <= forms
    assert {"kitʃəri", "kitʃʰri", "kagodʒ", "ʃutʃ / ʃui", "sutɛ",
            "tʃʰɛlki", "dʒʰaɽu", "tʃamotʃ", "qanto", "mugɽa", "metʰul"} <= forms
    assert {"toŋːɛ", "tʃiɛri", "dʰɔnuk", "tʃʰitʃʰ", "tʃitʃʰi",
            "dʰõa", "dʰuŋgijɛ", "nau̯", "lau̯kɛ", "nou̯ka", "dɛhɛrɛ"} <= forms
    assert {"gəl", "dʒawa", "bɛdɽo", "hãʈa", "orakɛrɛ", "dʰoka",
            "latʰ", "lɔtʰ", "lɔtʰi", "latʰi mara"} <= forms
    assert {"ʃatar kaʈa", "dɛkʰa", "ɔpɛkkʰa kɔra", "kãda", "ʃiddʰo kɔra",
            "kʰawa", "pani kʰawa", "pɛd"} <= forms
    assert {"kamɽano", "ɛlːɛ purmija", "alkʰo", "tʰeⁱŋgɛ", "modɽo",
            "bʰulɛ dʒawa", "ʃopno dɛkʰa", "kadʒ kɔra"} <= forms
    assert {"bɛtʃ", "kʰɛla", "tʃʰora", "dʰakka dɛwa", "ʈan", "ʈana",
            "bãdʰa", "motʃʰa", "tatɛ bona", "otdʒo", "ʃɛlai kɔra"} <= forms
    assert {"dʰoa", "gosol kɔra", "kʰand", "kaʈa", "poɽano",
            "qɛŋkʰolɛli", "xɛnd", "bikri kɔra", "tʃuri kɔra",
            "dʒut katʰa bo", "mitʰa bo", "mittʰa bɔla", "nɛa", "dɛa"} <= forms
    assert {"mara fɛla", "kitʃɛskirɛs", "korɛlag", "bʰalobaʃa",
            "gʰrina kɔra", "ɔntɛ", "otʰan", "dui̯", "tʃɛr", "pɛntʃ",
            "tʃʰoi̯", "tʃʰou̯"} <= forms
    assert {"ʃat", "aʈ", "ɛtʰ", "nɛᵘ", "nɔi̯", "dɔʃ", "ægaro",
            "bɛrɛ", "biʃ", "ʃɔ", "sɛᵘ", "hadʒar"} <= forms
    assert {"tʰorɛ", "kitʃʰu", "bɔɽo", "gɛⁱlɛ", "ʃanːi", "sanːi",
            "tʃukːɛ", "tʃʰotɛ", "digha", "otʰːa", "odʒɛ̃", "nɛbːa",
            "odʒɛnməlɛ", "moʈa"} <= forms
    assert {"ʃarua", "ʃaruɛ", "tʃɛpta", "gaɖːi", "dipːa", "dipːɛ",
            "ɔgobʰir", "xai̯ka", "ɛndraha malla", "kʰirɛ", "kirɛ",
            "pipaʃa pawa", "ɛmba", "miʃti"} <= forms
    assert {"ʈɔk", "titɛm", "dʒʰal", "pandʒka", "kitːi̯a", "kitːiɛ̯",
            "dʰirɛ dʰirɛ", "sɛmɛn", "mamisri", "gitʃə", "xaɛ̯ka",
            "xai̯kɛ", "sukʰɛl"} <= forms
    assert {"tʃaɛ̯ka", "bʰɪɳɖʒal", "ʈʰanda", "pʰɛtʃʰɛ", "hotɛrka",
            "mɛ̃i̯nja", "mɛ̃njɛ", "kije", "kijɛ"} <= forms
    assert {"gɛttʃʰa", "hɛdːɛ", "kaʈʃʰɛ", "ɖan", "bɛⁱjɛ", "mokʰɛro",
            "ʃobudʒ", "ɛkabaki", "kɔkʰon"} <= forms
    assert {"kotʰai̯", "nekɛⁱ", "kai̯tari", "kɔi̯ta", "oʈa", "ibrɛ",
            "ɛgulo", "udɛndrɛ"} <= forms
    assert {"ɛs", "ʃɛ", "em / nam", "amra", "nɪn", "nim", "tomra",
            "ar", "tara"} <= forms


def test_each_cell_points_to_the_correct_rendered_page():
    cells_2 = (read_tsv(CELLS_011_020) + read_tsv(CELLS_021_030) +
               read_tsv(CELLS_031_040) + read_tsv(CELLS_041_050))
    cells_2 += read_tsv(CELLS_051_060)
    cells_2 += read_tsv(CELLS_061_070)
    cells_2 += read_tsv(CELLS_071_080)
    cells_2 += read_tsv(CELLS_081_090)
    cells_2 += read_tsv(CELLS_091_100)
    cells_2 += read_tsv(CELLS_101_110)
    cells_2 += read_tsv(CELLS_111_120)
    cells_2 += read_tsv(CELLS_121_130)
    cells_2 += read_tsv(CELLS_131_140)
    cells_2 += read_tsv(CELLS_141_150)
    cells_2 += read_tsv(CELLS_151_160)
    cells_2 += read_tsv(CELLS_161_170)
    cells_2 += read_tsv(CELLS_171_180)
    cells_2 += read_tsv(CELLS_181_190)
    cells_2 += read_tsv(CELLS_191_200)
    cells_2 += read_tsv(CELLS_201_210)
    cells_2 += read_tsv(CELLS_211_220)
    cells_2 += read_tsv(CELLS_221_230)
    cells_2 += read_tsv(CELLS_231_240)
    cells_2 += read_tsv(CELLS_241_250)
    cells_2 += read_tsv(CELLS_251_260)
    cells_2 += read_tsv(CELLS_261_270)
    cells_2 += read_tsv(CELLS_271_280)
    cells_2 += read_tsv(CELLS_281_290)
    cells_2 += read_tsv(CELLS_291_300)
    cells_2 += read_tsv(CELLS_301_307)
    hashes = {
        "39": "946ff8df00c62586ba1daeab766503297deb13d890c0a257c2268dafd9641e35",
        "40": "a943fd4d96a00cc1c76aca4409550a5c06dba65ffedf9a23b0f054fffaf1851e",
        "41": "1ae2acbb1a628ea21e1ad44868f1511e503c75d776c9e8613dbead650eb86131",
        "42": "691e3a92fc69930111ee1a5f4b66cdbca12dc8b182bb59075dfd36840ce8564c",
        "43": "5e24b3f741d34b4cf197ca849cd221ca25f22bf1948f020cd630cc23a7a684df",
        "44": "bdd0f705bf6e00ec853e3b910fc4b8f5f8c469afa8940604d42b4714e6afceaa",
        "45": "e7fd9d9b2c86eb59c5c9e48c0901cd81c06c87e0127eec280b83e1b0dcb3538d",
        "46": "d09898f5a931b2897783a6b517c18dca6dc8df464faa53ec717ea91700b6dbd4",
        "47": "826017bd8f270c01f07130290e70b400796d90d7c4b4a857935eda230de6238a",
        "48": "9b085fe9413daa4926c5004f910e9cf0f551d301d3cacf0c361cbc9b78a4582b",
        "49": "3961f6abf6887b0b57f57651e7c9f795f057cc2140fd6a1fcad0ba556096698c",
        "50": "27e740e67e861b0f7a1fd1c98e255b9d88ea4c910e577b677d5cf73718dce601",
        "51": "f523942c92303f3df63204c0d74ae05f6d0213fb674c8006c0d98b2c5f32024e",
        "52": "6ed56b95d7351fc9207a7e2688a5665ca9bd508efdcbf780480d903878031171",
        "53": "63b558e55499b53ca1c6c110af674e555eb4522c86f98d65c36664cc5fbbf381",
        "54": "883c0edbe3cc51069187ab56854259e757a38801444369680a3d35f7f479717a",
        "55": "a21143378b48042d8fce359523ca632e3f096ee914d0f0d5dc4e0408a5f0c2d7",
        "56": "c6088d6a0caa5d26d56f8d6df2b4a8eef1875e2c7f98fd408c98fe1473a71f38",
        "57": "84c0739aeaef327886753b5c2dd4f95c48f48e14b885cbe39756f76e84badd1d",
    }
    printed = {
        "39": "38", "40": "39", "41": "40", "42": "41", "43": "42", "44": "43",
        "45": "44",
        "46": "45",
        "47": "46",
        "48": "47",
        "49": "48",
        "50": "49",
        "51": "50",
        "52": "51",
        "53": "52",
        "54": "53",
        "55": "54",
        "56": "55",
        "57": "56",
    }
    for row in cells_2:
        assert row["evidence_sha256"] == hashes[row["physical_page"]]
        assert row["printed_page"] == printed[row["physical_page"]]


def test_manifest_keeps_controls_and_remainder_pending():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["state"] == "manual_review_complete"
    assert manifest["items_reviewed"] == [1, 307]
    assert manifest["pending_items"] == []
    assert manifest["not_used_coordinates"] == [
        "item-031/site-A", "item-031/site-B", "item-031/site-C",
        "item-031/site-D", "item-031/site-E", "item-031/site-0",
        "item-074/site-A", "item-074/site-B", "item-074/site-C",
        "item-074/site-D", "item-074/site-E", "item-074/site-0",
        "item-107/site-A", "item-107/site-B", "item-107/site-C",
        "item-107/site-D", "item-107/site-E", "item-107/site-0",
        "item-124/site-A", "item-124/site-B", "item-124/site-C",
        "item-124/site-D", "item-124/site-E", "item-124/site-0",
        "item-152/site-A", "item-152/site-B", "item-152/site-C",
        "item-152/site-D", "item-152/site-E", "item-152/site-0",
        "item-163/site-A", "item-163/site-B", "item-163/site-C",
        "item-163/site-D", "item-163/site-E", "item-163/site-0",
        "item-171/site-A", "item-171/site-B", "item-171/site-C",
        "item-171/site-D", "item-171/site-E", "item-171/site-0",
        "item-194/site-A", "item-194/site-B", "item-194/site-C",
        "item-194/site-D", "item-194/site-E", "item-194/site-0",
        "item-240/site-A", "item-240/site-B", "item-240/site-C",
        "item-240/site-D", "item-240/site-E", "item-240/site-0",
        "item-247/site-A", "item-247/site-B", "item-247/site-C",
        "item-247/site-D", "item-247/site-E", "item-247/site-0",
        "item-301/site-A", "item-301/site-B", "item-301/site-C",
        "item-301/site-D", "item-301/site-E", "item-301/site-0",
        "item-306/site-A", "item-306/site-B", "item-306/site-C",
        "item-306/site-D", "item-306/site-E", "item-306/site-0",
    ]
    assert manifest["site_identity_state"] == "pending_identity_and_schema_review"
    assert "audit-only" in manifest["control_policy"]
    assert manifest["ambiguous_coordinates"] == []
    assert manifest["illegible_coordinates"] == []
    assert manifest["reconciliation_state"] == "not_started_after_manual_freeze"
    assert "earlier audits supplied or verified none" in manifest["policy"]


def test_post_freeze_generator_is_exact_and_reproducible():
    subprocess.run([sys.executable, str(PACKAGE / "build_manual_chunks.py")], check=True)
    subprocess.run([sys.executable, str(POST_FREEZE_SCRIPT)], check=True)
    first_manifest_bytes = POST_FREEZE_MANIFEST.read_bytes()
    first_output_hashes = {
        path.name: sha256(path)
        for path in (
            RECONCILIATION, STAGING_AUDIT, STAGED_FORMS, SITE_METADATA,
            REFERENCE_METADATA, EXCLUSION_POLICY, SOUND_INVENTORY,
            SOUND_PROFILE, SOUND_DECISIONS,
        )
    }
    subprocess.run([sys.executable, str(POST_FREEZE_SCRIPT)], check=True)
    assert POST_FREEZE_MANIFEST.read_bytes() == first_manifest_bytes

    manifest = json.loads(POST_FREEZE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["state"] == "source_local_post_freeze_complete"
    assert manifest["manual_manifest_sha256"] == sha256(MANIFEST)
    assert manifest["manual_conceptual_cells"] == 1842
    assert manifest["manual_expanded_rows"] == 1869
    assert manifest["legacy_installed_rows"] == 1422
    assert manifest["legacy_audit_rows"] == 1809
    assert manifest["reconciliation_counts"] == {
        "blank_match": 136,
        "form_difference": 700,
        "form_exact": 722,
        "manual_recovered_legacy_unresolved": 239,
        "not_used_match": 72,
    }
    assert manifest["staging_counts"] == {
        "excluded_blank": 136,
        "excluded_control": 296,
        "excluded_not_used": 72,
        "staged_target": 1365,
    }
    assert manifest["outputs"] == first_output_hashes
    assert manifest["unresolved_lexical_coordinates"] == []
    assert len(manifest["deferred_shared_actions"]) == 6


def test_reconciliation_is_exhaustive_and_manual_readings_remain_authoritative():
    rows = read_tsv(RECONCILIATION)
    assert len(rows) == 1869
    assert len({row["manual_entry_key"] for row in rows}) == 1869
    assert Counter(row["comparison"] for row in rows) == {
        "form_exact": 722,
        "form_difference": 700,
        "manual_recovered_legacy_unresolved": 239,
        "blank_match": 136,
        "not_used_match": 72,
    }
    assert all(row["gloss_match"] == "true" for row in rows)
    assert all(row["group_match"] == "true" for row in rows)
    recovered = [
        row for row in rows
        if row["comparison"] == "manual_recovered_legacy_unresolved"
    ]
    assert len(recovered) == 239
    assert all(row["manual_status"] == "attested" and row["manual_form"] for row in recovered)
    assert all(row["legacy_status"] == "unresolved" for row in recovered)
    assert all(row["legacy_reason"] == "contains a glyph with no verified reading" for row in recovered)
    differences = [row for row in rows if row["comparison"] == "form_difference"]
    assert all(row["manual_form"] != row["legacy_raw_form"] for row in differences)
    item_274_d = [
        row for row in rows if row["item"] == "274" and row["site_code"] == "D"
    ]
    assert [(row["site_variant"], row["manual_form"]) for row in item_274_d] == [
        ("1", "korɛ"), ("2", "bʰalo")
    ]


def test_source_local_staging_metadata_and_exclusions_are_complete():
    with STAGED_FORMS.open(encoding="utf-8", newline="") as stream:
        staged = list(csv.reader(stream))
    audit = read_tsv(STAGING_AUDIT)
    sites = read_tsv(SITE_METADATA)
    reference = json.loads(REFERENCE_METADATA.read_text(encoding="utf-8"))
    exclusions = json.loads(EXCLUSION_POLICY.read_text(encoding="utf-8"))

    assert len(staged) == 1365
    assert {len(row) for row in staged} == {15}
    assert all(row[0] == "Kurux" and row[2] and row[5] == row[2] for row in staged)
    assert len({row[10] for row in staged}) == 1365
    assert all(row[10].startswith("silkurux2011:i") for row in staged)
    assert all(":0:" not in row[10] for row in staged)
    expected_dialects = {
        "A": "kurux2011-A-dima",
        "B": "kurux2011-B-gabindanagar",
        "C": "kurux2011-C-boldipukur",
        "D": "kurux2011-D-lohanipara",
        "E": "kurux2011-E-dulhapur",
    }
    for row in staged:
        code = row[10].split(":")[2]
        assert expected_dialects[code] in row[14]
        assert "kim-ahmad-kim-sangma2011kurux[p. " in row[7]
        assert row[8] == row[9] == row[11] == row[12] == row[13] == ""

    assert len(audit) == 1869
    assert Counter(row["disposition"] for row in audit) == {
        "staged_target": 1365,
        "excluded_control": 296,
        "excluded_blank": 136,
        "excluded_not_used": 72,
    }
    assert len(sites) == 6
    assert {row["site_code"]: row["site_name"] for row in sites} == {
        "A": "Dima", "B": "Gabindanagar", "C": "Boldipukur",
        "D": "Lohanipara", "E": "Dulhapur", "0": "Bangla",
    }
    assert all(not row["latitude"] and not row["longitude"] for row in sites)
    assert all("no exact site coordinate" in row["coordinate_note"] for row in sites)
    assert reference["id"] == "kim-ahmad-kim-sangma2011kurux"
    assert reference["authors"] == ["Amy Kim", "Mridul Ahmad", "Seung Kim", "Palash Roy Sangma"]
    assert reference["number"] == "2011-040" and reference["year"] == 2011
    assert reference["source_pdf_sha256"] == (
        "f2f06c25ac55462d6a40843539d8417e24a647bd1eb0bbe3f24ea3e45f0b9e4b"
    )
    assert exclusions["source_rows"] == 1869
    assert exclusions["staged_rows"] == 1365
    assert "audit-only" in exclusions["control_policy"]
    assert "No cognate or borrowing edge" in exclusions["etymology_policy"]


def test_source_local_sound_profile_covers_every_staged_form():
    with STAGED_FORMS.open(encoding="utf-8", newline="") as stream:
        staged = list(csv.reader(stream))
    inventory = read_tsv(SOUND_INVENTORY)
    decisions = json.loads(SOUND_DECISIONS.read_text(encoding="utf-8"))
    tokenizer = Tokenizer(str(SOUND_PROFILE))
    for row in staged:
        converted = tokenizer(row[2], column="IPA", segment_separator="", separator="")
        assert "�" not in converted
    assert tokenizer("t͜ʃʰɛnd", column="IPA", segment_separator="", separator="") == "cʰend"
    assert tokenizer("ʃuɹd͜ʒo", column="IPA", segment_separator="", separator="") == "śurjo"
    assert tokenizer("məⁱjə", column="IPA", segment_separator="", separator="") == "məⁱyə"
    assert tokenizer("püp", column="IPA", segment_separator="", separator="") == "püp"
    assert len(inventory) == 52
    assert len({row["codepoint"] for row in inventory}) == 52
    assert decisions["inventory_scope"] == {
        "all_attested_rows": 1661,
        "staged_target_rows": 1365,
        "unique_codepoints": 52,
    }
    assert decisions["unresolved_mappings"] == []


def test_shared_install_is_the_frozen_target_stage_byte_for_byte():
    assert INSTALLED_FORMS.read_bytes() == STAGED_FORMS.read_bytes()
    assert sha256(INSTALLED_FORMS) == (
        "1b970fe2e4f873dc9cb4806c104b727b6099b9d57b7e813d46a9f89f1c4531cb"
    )
    with INSTALLED_FORMS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 1365
    assert len({row[10] for row in rows}) == 1365
    assert all(row[10].startswith("silkurux2011:i") for row in rows)
    assert all(row[0] == "Kurux" and ":0:" not in row[10] for row in rows)


def test_shared_profile_is_exact_and_explicitly_routed():
    assert SHARED_PROFILE.read_bytes() == SOUND_PROFILE.read_bytes()
    assert sha256(SHARED_PROFILE) == (
        "ac76ab83a6d435e384cf7287fb275d3574343c35141409f2298009f97ffeeb23"
    )
    build = BUILD_SCRIPT.read_text(encoding="utf-8")
    route = 'if source_key == "kim-ahmad-kim-sangma2011kurux":'
    assert route in build
    route_block = build[build.index(route):build.index(route) + 180]
    assert 'row_ipa = "sil-kurux"' in route_block
    assert "row_convert = True" in route_block


def test_shared_site_registry_matches_the_frozen_source_metadata():
    sites = {row["site_id"]: row for row in read_tsv(SITE_METADATA)}
    with DIALECT_REGISTRY.open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    for site in sites.values():
        registered = dialects[site["site_id"]]
        assert registered["Tag"] == site["dialect_tag"]
        assert registered["Language_ID"] == site["language_id"]
        assert registered["Source_Language_ID"] == site["site_id"]
        assert registered["Name"] == site["site_name"]
        assert registered["Glottocode"] == site["glottocode"]
        assert registered["Latitude"] == registered["Longitude"] == ""
        assert registered["Quality"] == ""
        assert "coordinate" in registered["Location"] or site["role"] == "control"


def test_shared_reference_records_manual_only_provenance_and_exact_scope():
    text = SOURCES_BIB.read_text(encoding="utf-8")
    start = text.index("@techreport{kim-ahmad-kim-sangma2011kurux,")
    end = text.index("\n}\n", start) + 3
    entry = text[start:end]
    assert "items 1--307 at the five Kurux target sites A--E" in entry
    assert "standard Bangla comparison list at site 0 is retained audit-only" in entry
    assert "Every retained lexical reading was transcribed manually" in entry
    assert "OCR, PDF text, the embedded legacy font and prior installed forms" in entry
    assert "Aryaman Arora and OpenAI Codex" in entry
    assert "forms containing a glyph whose reading is not established" not in entry


def test_shared_integration_manifest_preserves_exclusions_and_deferred_gates():
    manifest = json.loads(SHARED_INTEGRATION_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["state"] == "shared_source_specific_integration_complete"
    assert manifest["installed"]["rows"] == 1365
    assert manifest["installed"]["sha256"] == sha256(INSTALLED_FORMS)
    assert manifest["audit"]["rows"] == 1869
    assert manifest["audit"]["sha256"] == sha256(STAGING_AUDIT)
    assert manifest["audit"]["dispositions"] == {
        "staged_target": 1365,
        "excluded_control": 296,
        "excluded_blank": 136,
        "excluded_not_used": 72,
    }
    assert manifest["audit"]["unresolved_lexical_coordinates"] == []
    assert manifest["profile"]["sha256"] == sha256(SHARED_PROFILE)
    assert len(manifest["deferred_gates"]) == 5
