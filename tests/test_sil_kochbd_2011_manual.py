"""Guard the cumulative manual-only partial recovery of ESR 2011-023."""

import csv
import hashlib
import json
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/sil_kochbd_2011_manual"
LINES = PACKAGE / "manual_chunks/p043-items001-013-lines.tsv"
CELLS = PACKAGE / "manual_chunks/p043-items001-013-cells.tsv"
LINES_44 = PACKAGE / "manual_chunks/p044-items014-018-lines.tsv"
CELLS_44 = PACKAGE / "manual_chunks/p044-items014-018-cells.tsv"
LINES_44B = PACKAGE / "manual_chunks/p044-items019-023-lines.tsv"
CELLS_44B = PACKAGE / "manual_chunks/p044-items019-023-cells.tsv"
LINES_44C = PACKAGE / "manual_chunks/p044-items024-028-lines.tsv"
CELLS_44C = PACKAGE / "manual_chunks/p044-items024-028-cells.tsv"
LINES_45 = PACKAGE / "manual_chunks/p045-items029-033-lines.tsv"
CELLS_45 = PACKAGE / "manual_chunks/p045-items029-033-cells.tsv"
LINES_45B = PACKAGE / "manual_chunks/p045-items034-037-lines.tsv"
CELLS_45B = PACKAGE / "manual_chunks/p045-items034-037-cells.tsv"
LINES_45C = PACKAGE / "manual_chunks/p045-items038-042-lines.tsv"
CELLS_45C = PACKAGE / "manual_chunks/p045-items038-042-cells.tsv"
LINES_45D = PACKAGE / "manual_chunks/p045-items043-046-lines.tsv"
CELLS_45D = PACKAGE / "manual_chunks/p045-items043-046-cells.tsv"
LINES_46 = PACKAGE / "manual_chunks/p046-items047-051-lines.tsv"
CELLS_46 = PACKAGE / "manual_chunks/p046-items047-051-cells.tsv"
LINES_46B = PACKAGE / "manual_chunks/p046-items052-056-lines.tsv"
CELLS_46B = PACKAGE / "manual_chunks/p046-items052-056-cells.tsv"
LINES_46C = PACKAGE / "manual_chunks/p046-items057-061-lines.tsv"
CELLS_46C = PACKAGE / "manual_chunks/p046-items057-061-cells.tsv"
LINES_47 = PACKAGE / "manual_chunks/p047-items062-066-lines.tsv"
CELLS_47 = PACKAGE / "manual_chunks/p047-items062-066-cells.tsv"
LINES_47B = PACKAGE / "manual_chunks/p047-items067-071-lines.tsv"
CELLS_47B = PACKAGE / "manual_chunks/p047-items067-071-cells.tsv"
LINES_47C = PACKAGE / "manual_chunks/p047-items072-076-lines.tsv"
CELLS_47C = PACKAGE / "manual_chunks/p047-items072-076-cells.tsv"
LINES_48 = PACKAGE / "manual_chunks/p048-items077-081-lines.tsv"
CELLS_48 = PACKAGE / "manual_chunks/p048-items077-081-cells.tsv"
LINES_48B = PACKAGE / "manual_chunks/p048-items082-085-lines.tsv"
CELLS_48B = PACKAGE / "manual_chunks/p048-items082-085-cells.tsv"
LINES_48C = PACKAGE / "manual_chunks/p048-items086-089-lines.tsv"
CELLS_48C = PACKAGE / "manual_chunks/p048-items086-089-cells.tsv"
LINES_48D = PACKAGE / "manual_chunks/p048-items090-093-lines.tsv"
CELLS_48D = PACKAGE / "manual_chunks/p048-items090-093-cells.tsv"
LINES_49 = PACKAGE / "manual_chunks/p049-items094-097-lines.tsv"
CELLS_49 = PACKAGE / "manual_chunks/p049-items094-097-cells.tsv"
LINES_49B = PACKAGE / "manual_chunks/p049-items098-100-lines.tsv"
CELLS_49B = PACKAGE / "manual_chunks/p049-items098-100-cells.tsv"
LINES_49C = PACKAGE / "manual_chunks/p049-items101-104-lines.tsv"
CELLS_49C = PACKAGE / "manual_chunks/p049-items101-104-cells.tsv"
LINES_49D = PACKAGE / "manual_chunks/p049-items105-107-lines.tsv"
CELLS_49D = PACKAGE / "manual_chunks/p049-items105-107-cells.tsv"
LINES_50 = PACKAGE / "manual_chunks/p050-items108-111-lines.tsv"
CELLS_50 = PACKAGE / "manual_chunks/p050-items108-111-cells.tsv"
LINES_50B = PACKAGE / "manual_chunks/p050-items112-114-lines.tsv"
CELLS_50B = PACKAGE / "manual_chunks/p050-items112-114-cells.tsv"
LINES_50C = PACKAGE / "manual_chunks/p050-items115-118-lines.tsv"
CELLS_50C = PACKAGE / "manual_chunks/p050-items115-118-cells.tsv"
LINES_50D = PACKAGE / "manual_chunks/p050-items119-121-lines.tsv"
CELLS_50D = PACKAGE / "manual_chunks/p050-items119-121-cells.tsv"
LINES_51 = PACKAGE / "manual_chunks/p051-items122-128-lines.tsv"
CELLS_51 = PACKAGE / "manual_chunks/p051-items122-128-cells.tsv"
LINES_51B = PACKAGE / "manual_chunks/p051-items129-135-lines.tsv"
CELLS_51B = PACKAGE / "manual_chunks/p051-items129-135-cells.tsv"
LINES_52 = PACKAGE / "manual_chunks/p052-items136-142-lines.tsv"
CELLS_52 = PACKAGE / "manual_chunks/p052-items136-142-cells.tsv"
LINES_52B = PACKAGE / "manual_chunks/p052-items143-149-lines.tsv"
CELLS_52B = PACKAGE / "manual_chunks/p052-items143-149-cells.tsv"
LINES_53 = PACKAGE / "manual_chunks/p053-items150-157-lines.tsv"
CELLS_53 = PACKAGE / "manual_chunks/p053-items150-157-cells.tsv"
LINES_53B = PACKAGE / "manual_chunks/p053-items158-167-lines.tsv"
CELLS_53B = PACKAGE / "manual_chunks/p053-items158-167-cells.tsv"
LINES_54 = PACKAGE / "manual_chunks/p054-items168-176-lines.tsv"
CELLS_54 = PACKAGE / "manual_chunks/p054-items168-176-cells.tsv"
LINES_54B = PACKAGE / "manual_chunks/p054-items177-183-lines.tsv"
CELLS_54B = PACKAGE / "manual_chunks/p054-items177-183-cells.tsv"
LINES_55 = PACKAGE / "manual_chunks/p055-items184-191-lines.tsv"
CELLS_55 = PACKAGE / "manual_chunks/p055-items184-191-cells.tsv"
LINES_55B = PACKAGE / "manual_chunks/p055-items192-199-lines.tsv"
CELLS_55B = PACKAGE / "manual_chunks/p055-items192-199-cells.tsv"
LINES_56 = PACKAGE / "manual_chunks/p056-items200-206-lines.tsv"
CELLS_56 = PACKAGE / "manual_chunks/p056-items200-206-cells.tsv"
LINES_56B = PACKAGE / "manual_chunks/p056-items207-213-lines.tsv"
CELLS_56B = PACKAGE / "manual_chunks/p056-items207-213-cells.tsv"
LINES_57 = PACKAGE / "manual_chunks/p057-items214-221-lines.tsv"
CELLS_57 = PACKAGE / "manual_chunks/p057-items214-221-cells.tsv"
LINES_57B = PACKAGE / "manual_chunks/p057-items222-228-lines.tsv"
CELLS_57B = PACKAGE / "manual_chunks/p057-items222-228-cells.tsv"
LINES_57C = PACKAGE / "manual_chunks/p057-item229-lines.tsv"
CELLS_57C = PACKAGE / "manual_chunks/p057-item229-cells.tsv"
LINES_58 = PACKAGE / "manual_chunks/p058-items230-237-lines.tsv"
CELLS_58 = PACKAGE / "manual_chunks/p058-items230-237-cells.tsv"
LINES_58B = PACKAGE / "manual_chunks/p058-items238-246-lines.tsv"
CELLS_58B = PACKAGE / "manual_chunks/p058-items238-246-cells.tsv"
LINES_59 = PACKAGE / "manual_chunks/p059-items247-253-lines.tsv"
CELLS_59 = PACKAGE / "manual_chunks/p059-items247-253-cells.tsv"
LINES_59B = PACKAGE / "manual_chunks/p059-item254-lines.tsv"
CELLS_59B = PACKAGE / "manual_chunks/p059-item254-cells.tsv"
LINES_59C = PACKAGE / "manual_chunks/p059-items255-261-lines.tsv"
CELLS_59C = PACKAGE / "manual_chunks/p059-items255-261-cells.tsv"
LINES_60 = PACKAGE / "manual_chunks/p060-items262-268-lines.tsv"
CELLS_60 = PACKAGE / "manual_chunks/p060-items262-268-cells.tsv"
LINES_60B = PACKAGE / "manual_chunks/p060-items269-275-lines.tsv"
CELLS_60B = PACKAGE / "manual_chunks/p060-items269-275-cells.tsv"
LINES_61 = PACKAGE / "manual_chunks/p061-items276-282-lines.tsv"
CELLS_61 = PACKAGE / "manual_chunks/p061-items276-282-cells.tsv"
LINES_61B = PACKAGE / "manual_chunks/p061-items283-291-lines.tsv"
CELLS_61B = PACKAGE / "manual_chunks/p061-items283-291-cells.tsv"
LINES_62 = PACKAGE / "manual_chunks/p062-items292-300-lines.tsv"
CELLS_62 = PACKAGE / "manual_chunks/p062-items292-300-cells.tsv"
LINES_62B = PACKAGE / "manual_chunks/p062-items301-307-lines.tsv"
CELLS_62B = PACKAGE / "manual_chunks/p062-items301-307-cells.tsv"
MANIFEST = PACKAGE / "source_manifest.json"

CELL_FILES = [CELLS, CELLS_44, CELLS_44B, CELLS_44C, CELLS_45, CELLS_45B,
              CELLS_45C, CELLS_45D, CELLS_46, CELLS_46B, CELLS_46C,
              CELLS_47, CELLS_47B, CELLS_47C, CELLS_48, CELLS_48B, CELLS_48C,
              CELLS_48D, CELLS_49, CELLS_49B, CELLS_49C, CELLS_49D, CELLS_50,
              CELLS_50B, CELLS_50C, CELLS_50D, CELLS_51, CELLS_51B, CELLS_52,
              CELLS_52B, CELLS_53, CELLS_53B, CELLS_54, CELLS_54B, CELLS_55,
              CELLS_55B, CELLS_56, CELLS_56B, CELLS_57, CELLS_57B,
              CELLS_57C, CELLS_58, CELLS_58B, CELLS_59, CELLS_59B,
              CELLS_59C, CELLS_60, CELLS_60B, CELLS_61, CELLS_61B,
              CELLS_62, CELLS_62B]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_cumulative_generator_is_exact_and_reproducible():
    subprocess.run([sys.executable, str(PACKAGE / "build_page43.py")], check=True)
    lines = read_tsv(LINES)
    cells = read_tsv(CELLS)
    lines_44 = read_tsv(LINES_44)
    cells_44 = read_tsv(CELLS_44)
    lines_44b = read_tsv(LINES_44B)
    cells_44b = read_tsv(CELLS_44B)
    lines_44c = read_tsv(LINES_44C)
    cells_44c = read_tsv(CELLS_44C)
    lines_45 = read_tsv(LINES_45)
    cells_45 = read_tsv(CELLS_45)
    lines_45b = read_tsv(LINES_45B)
    cells_45b = read_tsv(CELLS_45B)
    lines_45c = read_tsv(LINES_45C)
    cells_45c = read_tsv(CELLS_45C)
    lines_45d = read_tsv(LINES_45D)
    cells_45d = read_tsv(CELLS_45D)
    lines_46 = read_tsv(LINES_46)
    cells_46 = read_tsv(CELLS_46)
    lines_46b = read_tsv(LINES_46B)
    cells_46b = read_tsv(CELLS_46B)
    lines_46c = read_tsv(LINES_46C)
    cells_46c = read_tsv(CELLS_46C)
    lines_47 = read_tsv(LINES_47)
    cells_47 = read_tsv(CELLS_47)
    lines_47b = read_tsv(LINES_47B)
    cells_47b = read_tsv(CELLS_47B)
    lines_47c = read_tsv(LINES_47C)
    cells_47c = read_tsv(CELLS_47C)
    lines_48 = read_tsv(LINES_48)
    cells_48 = read_tsv(CELLS_48)
    lines_48b = read_tsv(LINES_48B)
    cells_48b = read_tsv(CELLS_48B)
    lines_48c = read_tsv(LINES_48C)
    cells_48c = read_tsv(CELLS_48C)
    lines_48d = read_tsv(LINES_48D)
    cells_48d = read_tsv(CELLS_48D)
    lines_49 = read_tsv(LINES_49)
    cells_49 = read_tsv(CELLS_49)
    lines_49b = read_tsv(LINES_49B)
    cells_49b = read_tsv(CELLS_49B)
    lines_49c = read_tsv(LINES_49C)
    cells_49c = read_tsv(CELLS_49C)
    lines_49d = read_tsv(LINES_49D)
    cells_49d = read_tsv(CELLS_49D)
    lines_50 = read_tsv(LINES_50)
    cells_50 = read_tsv(CELLS_50)
    lines_50b = read_tsv(LINES_50B)
    cells_50b = read_tsv(CELLS_50B)
    lines_50c = read_tsv(LINES_50C)
    cells_50c = read_tsv(CELLS_50C)
    lines_50d = read_tsv(LINES_50D)
    cells_50d = read_tsv(CELLS_50D)
    lines_51 = read_tsv(LINES_51)
    cells_51 = read_tsv(CELLS_51)
    lines_51b = read_tsv(LINES_51B)
    cells_51b = read_tsv(CELLS_51B)
    lines_52 = read_tsv(LINES_52)
    cells_52 = read_tsv(CELLS_52)
    lines_52b = read_tsv(LINES_52B)
    cells_52b = read_tsv(CELLS_52B)
    lines_53 = read_tsv(LINES_53)
    cells_53 = read_tsv(CELLS_53)
    lines_53b = read_tsv(LINES_53B)
    cells_53b = read_tsv(CELLS_53B)
    lines_54 = read_tsv(LINES_54)
    cells_54 = read_tsv(CELLS_54)
    lines_54b = read_tsv(LINES_54B)
    cells_54b = read_tsv(CELLS_54B)
    lines_55 = read_tsv(LINES_55)
    cells_55 = read_tsv(CELLS_55)
    lines_55b = read_tsv(LINES_55B)
    cells_55b = read_tsv(CELLS_55B)
    lines_56 = read_tsv(LINES_56)
    cells_56 = read_tsv(CELLS_56)
    lines_56b = read_tsv(LINES_56B)
    cells_56b = read_tsv(CELLS_56B)
    lines_57 = read_tsv(LINES_57)
    cells_57 = read_tsv(CELLS_57)
    lines_57b = read_tsv(LINES_57B)
    cells_57b = read_tsv(CELLS_57B)
    lines_57c = read_tsv(LINES_57C)
    cells_57c = read_tsv(CELLS_57C)
    lines_58 = read_tsv(LINES_58)
    cells_58 = read_tsv(CELLS_58)
    lines_58b = read_tsv(LINES_58B)
    cells_58b = read_tsv(CELLS_58B)
    lines_59 = read_tsv(LINES_59)
    cells_59 = read_tsv(CELLS_59)
    lines_59b = read_tsv(LINES_59B)
    cells_59b = read_tsv(CELLS_59B)
    lines_59c = read_tsv(LINES_59C)
    cells_59c = read_tsv(CELLS_59C)
    lines_60 = read_tsv(LINES_60)
    cells_60 = read_tsv(CELLS_60)
    lines_60b = read_tsv(LINES_60B)
    cells_60b = read_tsv(CELLS_60B)
    lines_61 = read_tsv(LINES_61)
    cells_61 = read_tsv(CELLS_61)
    lines_61b = read_tsv(LINES_61B)
    cells_61b = read_tsv(CELLS_61B)
    lines_62 = read_tsv(LINES_62)
    cells_62 = read_tsv(CELLS_62)
    lines_62b = read_tsv(LINES_62B)
    cells_62b = read_tsv(CELLS_62B)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    final_counts = (
        manifest["response_lines"], manifest["conceptual_cells"],
        manifest["expanded_rows"],
    )
    # Keep the long historical per-chunk tuple stable, then check this page's
    # two new chunks and the final cumulative totals explicitly below.
    manifest = {
        **manifest,
        "response_lines": 945,
        "conceptual_cells": 1827,
        "expanded_rows": 1837,
    }
    assert (len(lines), len(lines_44), len(lines_44b), len(lines_44c), len(lines_45), len(lines_45b), len(lines_45c), len(lines_45d), len(lines_46), len(lines_46b), len(lines_46c), len(lines_47), len(lines_47b), len(lines_47c), len(lines_48), len(lines_48b), len(lines_48c), len(lines_48d), len(lines_49), len(lines_49b), len(lines_49c), len(lines_49d), len(lines_50), len(lines_50b), len(lines_50c), len(lines_50d), len(lines_51), len(lines_51b), len(lines_52), len(lines_52b), len(lines_53), len(lines_53b), len(lines_54), len(lines_54b), len(lines_55), len(lines_55b), len(lines_56), len(lines_56b), len(lines_57), len(lines_57b), len(lines_57c), len(lines_58), len(lines_58b), len(lines_59), len(lines_59b), len(lines_59c), manifest["response_lines"]) == (55, 24, 18, 14, 18, 10, 10, 13, 20, 21, 13, 18, 18, 20, 17, 10, 14, 13, 19, 12, 16, 11, 18, 11, 18, 11, 30, 28, 30, 29, 28, 26, 22, 29, 28, 27, 28, 29, 28, 27, 3, 24, 29, 25, 6, 27, 945)
    assert (len(cells), len(cells_44), len(cells_44b), len(cells_44c), len(cells_45), len(cells_45b), len(cells_45c), len(cells_45d), len(cells_46), len(cells_46b), len(cells_46c), len(cells_47), len(cells_47b), len(cells_47c), len(cells_48), len(cells_48b), len(cells_48c), len(cells_48d), len(cells_49), len(cells_49b), len(cells_49c), len(cells_49d), len(cells_50), len(cells_50b), len(cells_50c), len(cells_50d), len(cells_51), len(cells_51b), len(cells_52), len(cells_52b), len(cells_53), len(cells_53b), len(cells_54), len(cells_54b), len(cells_55), len(cells_55b), len(cells_56), len(cells_56b), len(cells_57), len(cells_57b), len(cells_57c), len(cells_58), len(cells_58b), len(cells_59), len(cells_59b), len(cells_59c), manifest["conceptual_cells"], manifest["expanded_rows"]) == (91, 35, 35, 35, 35, 29, 36, 28, 35, 35, 35, 35, 35, 35, 35, 28, 28, 29, 29, 21, 28, 21, 28, 21, 28, 21, 50, 50, 51, 49, 56, 70, 63, 49, 56, 56, 49, 49, 56, 49, 7, 56, 65, 49, 7, 49, 1827, 1837)
    assert (len(lines_60), len(lines_60b), len(cells_60), len(cells_60b)) == (
        31, 31, 49, 49,
    )
    assert (len(lines_61), len(lines_61b), len(lines_62), len(lines_62b)) == (
        30, 28, 27, 21,
    )
    assert (len(cells_61), len(cells_61b), len(cells_62), len(cells_62b)) == (
        49, 63, 63, 49,
    )
    assert final_counts == (1113, 2149, 2159)
    chunks = {tuple(row["items"]): row for row in manifest["manual_chunks"]}
    assert sha256(LINES) == chunks[(1, 13)]["manual_lines_sha256"]
    assert sha256(CELLS) == chunks[(1, 13)]["expanded_cells_sha256"]
    assert sha256(LINES_44) == chunks[(14, 18)]["manual_lines_sha256"]
    assert sha256(CELLS_44) == chunks[(14, 18)]["expanded_cells_sha256"]
    assert sha256(LINES_44B) == chunks[(19, 23)]["manual_lines_sha256"]
    assert sha256(CELLS_44B) == chunks[(19, 23)]["expanded_cells_sha256"]
    assert sha256(LINES_44C) == chunks[(24, 28)]["manual_lines_sha256"]
    assert sha256(CELLS_44C) == chunks[(24, 28)]["expanded_cells_sha256"]
    assert sha256(LINES_45) == chunks[(29, 33)]["manual_lines_sha256"]
    assert sha256(CELLS_45) == chunks[(29, 33)]["expanded_cells_sha256"]
    assert sha256(LINES_45B) == chunks[(34, 37)]["manual_lines_sha256"]
    assert sha256(CELLS_45B) == chunks[(34, 37)]["expanded_cells_sha256"]
    assert chunks[(34, 37)]["conceptual_cells"] == 28
    assert chunks[(34, 37)]["expanded_rows"] == 29
    assert sha256(LINES_45C) == chunks[(38, 42)]["manual_lines_sha256"]
    assert sha256(CELLS_45C) == chunks[(38, 42)]["expanded_cells_sha256"]
    assert chunks[(38, 42)]["conceptual_cells"] == 35
    assert chunks[(38, 42)]["expanded_rows"] == 36
    assert sha256(LINES_45D) == chunks[(43, 46)]["manual_lines_sha256"]
    assert sha256(CELLS_45D) == chunks[(43, 46)]["expanded_cells_sha256"]
    assert sha256(LINES_46) == chunks[(47, 51)]["manual_lines_sha256"]
    assert sha256(CELLS_46) == chunks[(47, 51)]["expanded_cells_sha256"]
    assert sha256(LINES_46B) == chunks[(52, 56)]["manual_lines_sha256"]
    assert sha256(CELLS_46B) == chunks[(52, 56)]["expanded_cells_sha256"]
    assert sha256(LINES_46C) == chunks[(57, 61)]["manual_lines_sha256"]
    assert sha256(CELLS_46C) == chunks[(57, 61)]["expanded_cells_sha256"]
    assert sha256(LINES_47) == chunks[(62, 66)]["manual_lines_sha256"]
    assert sha256(CELLS_47) == chunks[(62, 66)]["expanded_cells_sha256"]
    assert sha256(LINES_47B) == chunks[(67, 71)]["manual_lines_sha256"]
    assert sha256(CELLS_47B) == chunks[(67, 71)]["expanded_cells_sha256"]
    assert sha256(LINES_47C) == chunks[(72, 76)]["manual_lines_sha256"]
    assert sha256(CELLS_47C) == chunks[(72, 76)]["expanded_cells_sha256"]
    assert sha256(LINES_48) == chunks[(77, 81)]["manual_lines_sha256"]
    assert sha256(CELLS_48) == chunks[(77, 81)]["expanded_cells_sha256"]
    assert sha256(LINES_48B) == chunks[(82, 85)]["manual_lines_sha256"]
    assert sha256(CELLS_48B) == chunks[(82, 85)]["expanded_cells_sha256"]
    assert sha256(LINES_48C) == chunks[(86, 89)]["manual_lines_sha256"]
    assert sha256(CELLS_48C) == chunks[(86, 89)]["expanded_cells_sha256"]
    assert sha256(LINES_48D) == chunks[(90, 93)]["manual_lines_sha256"]
    assert sha256(CELLS_48D) == chunks[(90, 93)]["expanded_cells_sha256"]
    assert chunks[(90, 93)]["expanded_rows"] == 29
    assert sha256(LINES_49) == chunks[(94, 97)]["manual_lines_sha256"]
    assert sha256(CELLS_49) == chunks[(94, 97)]["expanded_cells_sha256"]
    assert chunks[(94, 97)]["expanded_rows"] == 29
    assert sha256(LINES_49B) == chunks[(98, 100)]["manual_lines_sha256"]
    assert sha256(CELLS_49B) == chunks[(98, 100)]["expanded_cells_sha256"]
    assert sha256(LINES_49C) == chunks[(101, 104)]["manual_lines_sha256"]
    assert sha256(CELLS_49C) == chunks[(101, 104)]["expanded_cells_sha256"]
    assert sha256(LINES_49D) == chunks[(105, 107)]["manual_lines_sha256"]
    assert sha256(CELLS_49D) == chunks[(105, 107)]["expanded_cells_sha256"]
    assert sha256(LINES_50) == chunks[(108, 111)]["manual_lines_sha256"]
    assert sha256(CELLS_50) == chunks[(108, 111)]["expanded_cells_sha256"]
    assert sha256(LINES_50B) == chunks[(112, 114)]["manual_lines_sha256"]
    assert sha256(CELLS_50B) == chunks[(112, 114)]["expanded_cells_sha256"]
    assert sha256(LINES_50C) == chunks[(115, 118)]["manual_lines_sha256"]
    assert sha256(CELLS_50C) == chunks[(115, 118)]["expanded_cells_sha256"]
    assert sha256(LINES_50D) == chunks[(119, 121)]["manual_lines_sha256"]
    assert sha256(CELLS_50D) == chunks[(119, 121)]["expanded_cells_sha256"]
    assert sha256(LINES_51) == chunks[(122, 128)]["manual_lines_sha256"]
    assert sha256(CELLS_51) == chunks[(122, 128)]["expanded_cells_sha256"]
    assert sha256(LINES_51B) == chunks[(129, 135)]["manual_lines_sha256"]
    assert sha256(CELLS_51B) == chunks[(129, 135)]["expanded_cells_sha256"]
    assert sha256(LINES_52) == chunks[(136, 142)]["manual_lines_sha256"]
    assert sha256(CELLS_52) == chunks[(136, 142)]["expanded_cells_sha256"]
    assert sha256(LINES_52B) == chunks[(143, 149)]["manual_lines_sha256"]
    assert sha256(CELLS_52B) == chunks[(143, 149)]["expanded_cells_sha256"]
    assert sha256(LINES_53) == chunks[(150, 157)]["manual_lines_sha256"]
    assert sha256(CELLS_53) == chunks[(150, 157)]["expanded_cells_sha256"]
    assert sha256(LINES_53B) == chunks[(158, 167)]["manual_lines_sha256"]
    assert sha256(CELLS_53B) == chunks[(158, 167)]["expanded_cells_sha256"]
    assert sha256(LINES_54) == chunks[(168, 176)]["manual_lines_sha256"]
    assert sha256(CELLS_54) == chunks[(168, 176)]["expanded_cells_sha256"]
    assert sha256(LINES_54B) == chunks[(177, 183)]["manual_lines_sha256"]
    assert sha256(CELLS_54B) == chunks[(177, 183)]["expanded_cells_sha256"]
    assert sha256(LINES_55) == chunks[(184, 191)]["manual_lines_sha256"]
    assert sha256(CELLS_55) == chunks[(184, 191)]["expanded_cells_sha256"]
    assert sha256(LINES_55B) == chunks[(192, 199)]["manual_lines_sha256"]
    assert sha256(CELLS_55B) == chunks[(192, 199)]["expanded_cells_sha256"]
    assert sha256(LINES_56) == chunks[(200, 206)]["manual_lines_sha256"]
    assert sha256(CELLS_56) == chunks[(200, 206)]["expanded_cells_sha256"]
    assert sha256(LINES_56B) == chunks[(207, 213)]["manual_lines_sha256"]
    assert sha256(CELLS_56B) == chunks[(207, 213)]["expanded_cells_sha256"]
    assert sha256(LINES_57) == chunks[(214, 221)]["manual_lines_sha256"]
    assert sha256(CELLS_57) == chunks[(214, 221)]["expanded_cells_sha256"]
    assert sha256(LINES_57B) == chunks[(222, 228)]["manual_lines_sha256"]
    assert sha256(CELLS_57B) == chunks[(222, 228)]["expanded_cells_sha256"]
    assert sha256(LINES_57C) == chunks[(229, 229)]["manual_lines_sha256"]
    assert sha256(CELLS_57C) == chunks[(229, 229)]["expanded_cells_sha256"]
    assert sha256(LINES_58) == chunks[(230, 237)]["manual_lines_sha256"]
    assert sha256(CELLS_58) == chunks[(230, 237)]["expanded_cells_sha256"]
    assert sha256(LINES_58B) == chunks[(238, 246)]["manual_lines_sha256"]
    assert sha256(CELLS_58B) == chunks[(238, 246)]["expanded_cells_sha256"]
    assert sha256(LINES_59) == chunks[(247, 253)]["manual_lines_sha256"]
    assert sha256(CELLS_59) == chunks[(247, 253)]["expanded_cells_sha256"]
    assert sha256(LINES_59B) == chunks[(254, 254)]["manual_lines_sha256"]
    assert sha256(CELLS_59B) == chunks[(254, 254)]["expanded_cells_sha256"]
    assert sha256(LINES_59C) == chunks[(255, 261)]["manual_lines_sha256"]
    assert sha256(CELLS_59C) == chunks[(255, 261)]["expanded_cells_sha256"]
    assert sha256(LINES_60) == chunks[(262, 268)]["manual_lines_sha256"]
    assert sha256(CELLS_60) == chunks[(262, 268)]["expanded_cells_sha256"]
    assert sha256(LINES_60B) == chunks[(269, 275)]["manual_lines_sha256"]
    assert sha256(CELLS_60B) == chunks[(269, 275)]["expanded_cells_sha256"]
    assert sha256(LINES_61) == chunks[(276, 282)]["manual_lines_sha256"]
    assert sha256(CELLS_61) == chunks[(276, 282)]["expanded_cells_sha256"]
    assert sha256(LINES_61B) == chunks[(283, 291)]["manual_lines_sha256"]
    assert sha256(CELLS_61B) == chunks[(283, 291)]["expanded_cells_sha256"]
    assert sha256(LINES_62) == chunks[(292, 300)]["manual_lines_sha256"]
    assert sha256(CELLS_62) == chunks[(292, 300)]["expanded_cells_sha256"]
    assert sha256(LINES_62B) == chunks[(301, 307)]["manual_lines_sha256"]
    assert sha256(CELLS_62B) == chunks[(301, 307)]["expanded_cells_sha256"]


def test_every_item_has_all_seven_sites_and_exact_dispositions():
    cells = sum((read_tsv(path) for path in CELL_FILES), [])
    by_item = defaultdict(set)
    for row in cells:
        by_item[int(row["item"])].add(row["site_code"])
    assert set(by_item) == set(range(1, 308))
    assert all(codes == set("0bclmqr") for codes in by_item.values())
    priority = {"attested": 0, "ambiguous": 1, "blank": 2, "not_used": 3}
    conceptual = {}
    for row in cells:
        key = (row["item"], row["site_code"])
        if key not in conceptual or priority[row["status"]] < priority[conceptual[key]["status"]]:
            conceptual[key] = row
    assert Counter(row["status"] for row in conceptual.values()) == {
        "attested": 1780, "blank": 25, "ambiguous": 225, "not_used": 119,
    }
    assert Counter((row["role"], row["status"]) for row in conceptual.values()) == {
        ("target", "attested"): 1013,
        ("target", "blank"): 10,
        ("target", "ambiguous"): 137,
        ("target", "not_used"): 68,
        ("control", "attested"): 767,
        ("control", "blank"): 15,
        ("control", "ambiguous"): 88,
        ("control", "not_used"): 51,
    }


def test_forms_are_nfc_and_never_contain_legacy_or_ocr_placeholders():
    cells = sum((read_tsv(path) for path in CELL_FILES), [])
    for row in cells:
        assert row["form"] == unicodedata.normalize("NFC", row["form"])
        assert not any(0xE000 <= ord(char) <= 0xF8FF for char in row["form"])
        assert "�" not in row["form"]
        if row["status"] == "attested":
            assert row["form"]
        else:
            assert not row["form"]
    unresolved = {(int(row["item"]), row["site_code"], row["visible_base"])
                  for row in cells if row["status"] == "ambiguous"}
    assert unresolved == {
        (3, "c", "raŋgrɛk"),
        (7, "b", "ramdʰonuk"),
        (7, "c", "ramdʰonuk"),
        (17, "c", "haput"),
        (17, "r", "haput"),
        (29, "b", "manat"),
        (29, "c", "manat"),
        (32, "b", "pʰarok"),
        (32, "c", "pʰarok"),
        (32, "r", "pʰarok"),
        (52, "l", "rɛktʰai̯"),
        (52, "m", "rɛktʰai̯"),
        (56, "m", "golot"),
        (62, "b", "pɛkɛn"),
        (62, "c", "pɛkɛn"),
        (62, "r", "pɛkɛn"),
        (66, "b", "murtʃuk"),
        (66, "c", "murtʃuk"),
        (66, "r", "murtʃuk"),
        (68, "b", "matsa"),
        (68, "c", "matsa"),
        (68, "r", "matsa"),
        (68, "l", "mattʃʰa"),
        (68, "m", "mattʃʰa"),
        (69, "l", "makbul"),
        (69, "m", "makbul"),
        (70, "l", "mattʃok"),
        (70, "m", "mattʃok"),
        (71, "l", "hamak"),
        (71, "m", "hamak"),
        (73, "c", "dupu"),
        (73, "q", "dupu"),
        (77, "b", "luwak"),
        (77, "c", "luwak"),
        (77, "r", "luwak"),
        (85, "b", "wak"),
        (85, "c", "wak"),
        (85, "r", "wak"),
        (85, "q", "wak"),
        (85, "l", "wak"),
        (85, "m", "wak"),
        (86, "l", "midʒut"),
        (86, "m", "miʔtʃut"),
        (90, "l", "dao̯gɛt"),
        (90, "m", "dao̯gɛp"),
        (96, "b", "abrɛk"),
        (96, "c", "abrɛk"),
        (96, "r", "abrɛk"),
        (103, "b", "mokkon"),
        (103, "c", "mokkon"),
        (103, "r", "mokkon"),
        (104, "b", "nakkuŋ"),
        (104, "c", "nakkuŋ"),
        (104, "r", "nakkuŋ"),
        (106, "b", "pai̯tʰok"),
        (106, "c", "pai̯tʰok"),
        (106, "r", "pai̯tʰok"),
        (108, "l", "kʰutʃuk"),
        (108, "m", "kʰutʃuk"),
        (111, "b", "tʃak gilai̯"),
        (111, "r", "tʃak gilai̯"),
        (111, "l", "tʃaktʃuk"),
        (111, "m", "tʃakwɛŋ"),
        (112, "b", "tʃakaprak"),
        (112, "c", "tʃakaprak"),
        (112, "q", "tʃak"),
        (112, "r", "tʃak"),
        (112, "l", "tʃak"),
        (112, "m", "tʃak"),
        (113, "b", "tʃaktala"),
        (113, "c", "tʃaktala"),
        (113, "r", "tʃaktala"),
        (113, "l", "tʃakpʰa"),
        (113, "m", "tʃakpʰa"),
        (114, "l", "tʃakʃi"),
        (114, "m", "tʃakʃi"),
        (115, "b", "tʃakʃikor"),
        (115, "c", "tʃakʃikor"),
        (115, "r", "tʃakʃikor"),
            (115, "l", "tʃakʃikor"),
            (115, "m", "tʃakʃikʰor"),
            (117, "b", "tʃakrɛŋ aprak"),
            (119, "c", "hokpɛkɛn"),
            (120, "b", "kʰopak"),
            (120, "c", "kʰopak"),
            (120, "r", "kʰopak"),
            (122, "l", "tuŋok"),
            (123, "b", "hok"),
            (123, "c", "hok"),
            (123, "r", "hok"),
            (123, "m", "pipuk"),
            (124, "c", "pʰoktʃa"),
            (127, "b", "morot"),
            (127, "c", "morot"),
            (127, "q", "morot"),
            (127, "r", "morot"),
            (127, "l", "morot"),
            (132, "l", "dʒikbipʰa"),
            (132, "m", "dʒikbipʰa"),
            (133, "b", "mitʃik"),
            (133, "c", "mitʃik"),
            (133, "r", "mitʃik"),
            (133, "l", "dʒikgɨwui̯"),
            (133, "m", "dʒikgɨwui̯"),
            (143, "b", "nok"),
            (143, "c", "nok"),
            (143, "q", "nok"),
            (143, "r", "nok"),
            (143, "l", "nok"),
            (143, "m", "nok"),
            (145, "b", "kʰokai̯ dukat"),
            (153, "b", "pantʃak"),
            (153, "c", "pantʃak"),
            (153, "r", "pantʃak"),
            (154, "b", "lɛkkʰa"),
            (154, "c", "lɛkkʰa"),
            (154, "r", "lɛkkʰa"),
            (154, "l", "lɛkkʰa"),
            (154, "m", "lɛkkʰa"),
            (157, "b", "nohɛk"),
            (157, "c", "nohɛk"),
            (157, "r", "nohɛk"),
            (157, "l", "nogɛk"),
            (157, "m", "nogɛk"),
            (166, "b", "tʰappala"),
            (166, "c", "tʰappala"),
            (166, "q", "tʰappala"),
            (166, "r", "tʰappala"),
            (174, "b", "tʃap"),
            (174, "c", "tʃap"),
            (174, "q", "tʃap"),
            (174, "r", "tʃap"),
            (174, "l", "tʃap"),
            (174, "m", "tʃap"),
            (185, "b", "kʰɛt"),
            (185, "c", "kʰɛt"),
            (185, "r", "kʰɛt"),
            (185, "l", "kʰɛp"),
            (185, "m", "kʰɛp"),
            (187, "b", "rot"),
            (187, "c", "rot"),
            (187, "r", "rot"),
            (187, "l", "rot"),
            (187, "m", "rot"),
            (188, "m", "ʃat"),
            (191, "b", "kat"),
            (191, "c", "kat"),
            (191, "q", "kak"),
            (191, "r", "kak"),
            (191, "l", "kak"),
            (191, "m", "kak"),
            (196, "b", "wandatlai̯"),
            (196, "r", "wandatlai̯"),
            (198, "b", "dʒamuk nuk"),
            (198, "c", "dʒamuk nuk"),
            (198, "r", "dʒamuk nuk"),
            (198, "l", "dʒumaŋ nuk"),
            (198, "m", "dʒumaŋ nuk"),
            (203, "b", "tʰuk"),
            (203, "c", "tʰuk"),
            (203, "r", "tʰuk"),
            (203, "l", "tʰip"),
            (203, "m", "tʰip"),
            (204, "0", "tola"),
            (205, "l", "ʃipʰak"),
            (206, "b", "but"),
            (206, "c", "but"),
            (206, "r", "but"),
            (206, "l", "bɨt"),
            (206, "m", "bɨt"),
            (208, "b", "tʃʰit"),
            (208, "c", "tʃʰit"),
            (208, "r", "tʃʰit"),
            (208, "l", "ʃɛʃɛt"),
            (208, "m", "ʃɛʃɛt"),
            (209, "b", "tak"),
            (209, "c", "tak"),
            (209, "r", "tak"),
            (209, "m", "tok"),
            (210, "l", "ʃuk"),
            (210, "m", "ʃuk"),
            (211, "l", "ʃuʃuk"),
            (211, "m", "ʃuʃut"),
            (219, "b", "nak"),
            (219, "c", "nak"),
            (220, "m", "hut"),
            (223, "l", "namnuk"),
            (223, "m", "nɛmnuk"),
            (225, "b", "(got)ʃa"),
            (230, "l", "(miŋ)korok"),
            (230, "m", "(go)korok"),
            (232, "l", "(miŋ)tʃakik"),
            (234, "l", "(miŋ)tʃɛgik"),
            (234, "m", "tʃei̯gik"),
            (235, "m", "tʃei̯gik ʃa"),
            (236, "m", "tʃei̯gik nii̯"),
            (237, "l", "(miŋ)kʰolgik"),
            (241, "b", "tɛp"),
            (241, "c", "tɛp"),
            (241, "r", "tɛp"),
            (243, "c", "bɛbak"),
            (243, "q", "bɛbak"),
            (244, "c", "matta"),
            (244, "l", "tʃuŋ(bijok)"),
            (246, "l", "rao̯twa"),
            (246, "m", "rao̯twa"),
            (254, "l", "tʰotwa"),
            (265, "l", "sawok"),
            (267, "b", "tɛp"),
            (267, "c", "tɛp"),
            (269, "b", "bɛgat"),
            (269, "c", "bɛgat"),
            (269, "r", "bɛgat"),
            (270, "l", "raʔnok"),
            (271, "l", "pei̯ʃok"),
            (273, "l", "tʃikka"),
            (273, "m", "tʃikka"),
            (285, "b", "pinak"),
            (285, "c", "pinak"),
            (285, "r", "pinak"),
            (286, "l", "boka/gippok"),
            (286, "m", "boka/gippok"),
            (286, "q", "boka/gippok"),
            (286, "b", "pɛbok"),
            (286, "c", "pɛbok"),
            (286, "r", "pɛbok"),
        }


def test_items_14_18_preserve_direct_page_readings_and_repeat_metadata():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_44)}
    assert cells[(14, "q")]["form"] == "gaŋ"
    assert cells[(14, "l")]["form"] == "tei̯ kʰar"
    assert cells[(15, "0")]["form"] == "maʈi"
    assert cells[(16, "b")]["form"] == "haʔdilɛka / kadoŋ"
    assert cells[(16, "b")]["group"] == "2 | 3"
    assert "repeated source response" in cells[(16, "b")]["note"]
    assert cells[(17, "l")]["form"] == "habukʰu"
    assert cells[(18, "b")]["form"] == "loŋtʰai̯"
    assert cells[(18, "q")]["form"] == "loŋtʰɛŋ"
    assert cells[(18, "0")]["form"] == "patʰor"


def test_items_19_23_preserve_direct_page_readings_repeats_and_blank():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_44B)}
    assert cells[(19, "m")]["form"] == "hanʔtʃɛŋ"
    assert cells[(19, "c")]["form"] == "hatʃɛŋ"
    assert cells[(19, "c")]["group"] == "1 | 2"
    assert cells[(19, "b")]["form"] == "hasɛŋ"
    assert cells[(20, "q")]["form"] == "ʃona"
    assert cells[(21, "0")]["form"] == "rupa"
    assert cells[(22, "b")]["form"] == "tai̯ni"
    assert cells[(22, "0")]["form"] == "adʒ"
    assert cells[(23, "l")]["status"] == "blank"
    assert cells[(23, "m")]["form"] == "mɨja"
    assert cells[(23, "b")]["form"] == "mui̯ja"
    assert cells[(23, "0")]["form"] == "gɔtokal / kalke"
    assert cells[(23, "q")]["form"] == "ganɛkai̯"


def test_items_24_28_complete_physical_page_44():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_44C)}
    assert cells[(24, "l")]["status"] == "blank"
    assert cells[(24, "b")]["form"] == "gɛnɛ"
    assert cells[(24, "q")]["form"] == "oi̯dina"
    assert cells[(25, "b")]["form"] == "hatai̯"
    assert cells[(25, "0")]["form"] == "ʃɔpta"
    assert cells[(26, "m")]["form"] == "dʒa"
    assert cells[(26, "q")]["form"] == "maʃ"
    assert cells[(27, "l")]["form"] == "bɨlʃi"
    assert cells[(27, "b")]["form"] == "bɔtʃʰor"
    assert cells[(28, "m")]["form"] == "ʃan"
    assert cells[(28, "c")]["form"] == "sanok"
    assert cells[(28, "0")]["form"] == "din"


def test_items_29_33_preserve_page_45_readings_and_unresolved_modifiers():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_45)}
    assert cells[(29, "m")]["form"] == "walni"
    assert cells[(29, "b")]["status"] == "ambiguous"
    assert cells[(29, "b")]["visible_base"] == "manat"
    assert cells[(29, "r")]["form"] == "manap"
    assert cells[(29, "0")]["form"] == "ʃokal"
    assert cells[(30, "m")]["form"] == "ʃanmadʒi"
    assert cells[(30, "q")]["form"] == "dupur"
    assert cells[(31, "r")]["form"] == "dasum"
    assert cells[(31, "c")]["form"] == "gasum"
    assert cells[(31, "0")]["form"] == "ʃondʰa"
    assert cells[(32, "b")]["visible_base"] == "pʰarok"
    assert cells[(32, "q")]["form"] == "pʰar"
    assert cells[(33, "l")]["form"] == "mai̯"
    assert cells[(33, "0")]["form"] == "dʰan"


def test_items_34_37_preserve_page_readings_and_item_35_variant():
    rows = read_tsv(CELLS_45B)
    forms = defaultdict(list)
    for row in rows:
        forms[(int(row["item"]), row["site_code"])].append(row["form"])
    assert forms[(34, "b")] == ["mai̯ruŋ"]
    assert forms[(34, "m")] == ["mai̯roŋ"]
    assert forms[(34, "0")] == ["tʃal"]
    assert forms[(35, "m")] == ["mai̯", "mai̯min"]
    assert forms[(35, "0")] == ["bʰat"]
    assert forms[(36, "q")] == ["gɔm"]
    assert forms[(37, "b")] == ["makʰu"]
    assert forms[(37, "m")] == ["aboŋ"]
    assert forms[(37, "0")] == ["bʰuʈʈa"]


def test_items_38_42_preserve_page_readings_and_item_38_variant():
    rows = read_tsv(CELLS_45C)
    forms = defaultdict(list)
    for row in rows:
        forms[(int(row["item"]), row["site_code"])].append(row["form"])
    assert forms[(38, "m")] == ["kʰan", "alubʰuʈa"]
    assert forms[(38, "q")] == ["alu"]
    assert forms[(39, "0")] == ["pʰulkopi"]
    assert forms[(40, "r")] == ["badʰakopi"]
    assert forms[(41, "q")] == ["bai̯gon"]
    assert forms[(41, "0")] == ["bɛgun"]
    assert forms[(41, "b")] == ["bantao̯"]
    assert forms[(41, "m")] == ["mantao̯"]
    assert forms[(42, "l")] == ["badam"]


def test_items_43_46_complete_physical_page_45_and_preserve_repeat():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_45D)}
    assert cells[(43, "b")]["form"] == "pan"
    assert cells[(43, "m")]["form"] == "panpʰaŋ"
    assert cells[(43, "0")]["form"] == "gatʃʰ"
    assert cells[(44, "0")]["form"] == "dal"
    assert cells[(44, "q")]["form"] == "dala"
    assert cells[(45, "b")]["form"] == "lai̯tʃak"
    assert cells[(45, "b")]["group"] == "1 | 2"
    assert "Repeated source response" in cells[(45, "b")]["note"]
    assert cells[(45, "c")]["form"] == "lɛsak"
    assert cells[(45, "m")]["form"] == "pantʃak"
    assert cells[(45, "0")]["form"] == "pata"
    assert cells[(46, "b")]["form"] == "kanta"
    assert cells[(46, "0")]["form"] == "kaʈa"
    assert cells[(46, "l")]["form"] == "asu"


def test_items_47_61_complete_physical_page_46():
    rows = read_tsv(CELLS_46) + read_tsv(CELLS_46B) + read_tsv(CELLS_46C)
    forms = defaultdict(list)
    cells = {}
    for row in rows:
        forms[(int(row["item"]), row["site_code"])].append(row["form"])
        cells[(int(row["item"]), row["site_code"])] = row
    assert forms[(47, "l")] == ["tʃadi̯l"]
    assert cells[(47, "b")]["group"] == "1 | 4"
    assert forms[(48, "0")] == ["bãʃ"]
    assert forms[(49, "b")] == ["tʰai̯"]
    assert forms[(50, "q")] == ["pai̯tʃuŋ"]
    assert forms[(51, "0")] == ["narikɛl"]
    assert cells[(52, "l")]["status"] == "ambiguous"
    assert cells[(52, "l")]["visible_base"] == "rɛktʰai̯"
    assert forms[(53, "q")] == ["ambok"]
    assert cells[(54, "l")]["status"] == "blank"
    assert cells[(55, "b")]["status"] == "blank"
    assert forms[(55, "q")] == ["biʃun"]
    assert cells[(56, "m")]["visible_base"] == "golot"
    assert cells[(56, "l")]["status"] == "blank"
    assert cells[(57, "b")]["group"] == "1 | 2"
    assert forms[(58, "b")] == ["tʃunu"]
    assert forms[(59, "l")] == ["tʃi̯u"]
    assert forms[(59, "0")] == ["mɔd"]
    assert forms[(60, "q")] == ["dudʰ"]
    assert forms[(61, "0")] == ["tɛl"]


def test_items_62_76_complete_physical_page_47():
    rows = read_tsv(CELLS_47) + read_tsv(CELLS_47B) + read_tsv(CELLS_47C)
    cells = {(int(row["item"]), row["site_code"]): row for row in rows}
    assert cells[(62, "b")]["visible_base"] == "pɛkɛn"
    assert cells[(62, "m")]["form"] == "randai̯"
    assert cells[(64, "b")]["form"] == "pia̯o̯"
    assert cells[(64, "0")]["form"] == "pɛa̯dʒ"
    assert cells[(66, "r")]["visible_base"] == "murtʃuk"
    assert cells[(67, "l")]["form"] == "moŋma"
    assert cells[(68, "m")]["visible_base"] == "mattʃʰa"
    assert cells[(69, "q")]["form"] == "bʰaluk"
    assert cells[(70, "b")]["form"] == "matʃao̯"
    assert cells[(71, "b")]["form"] == "kao̯ i"
    assert cells[(72, "l")]["status"] == "blank"
    assert cells[(73, "l")]["form"] == "dipiu̯"
    assert cells[(73, "c")]["visible_base"] == "dupu"
    assert cells[(75, "l")]["status"] == "blank"
    assert cells[(76, "m")]["form"] == "katʰua̯"


def test_items_77_81_preserve_direct_page_48_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_48)}
    assert cells[(77, "l")]["form"] == "luklak"
    assert cells[(77, "b")]["status"] == "ambiguous"
    assert cells[(77, "b")]["visible_base"] == "luwak"
    assert cells[(77, "q")]["form"] == "bæŋ"
    assert cells[(78, "l")]["form"] == "kɨi"
    assert cells[(78, "b")]["form"] == "kui"
    assert cells[(79, "m")]["form"] == "bɨi̯ɾa"
    assert cells[(79, "b")]["form"] == "bilai̯"
    assert cells[(79, "q")]["form"] == "mɛa̯o̯"
    assert cells[(80, "q")]["form"] == "maʔʃu"
    assert cells[(80, "b")]["form"] == "maʔsu"
    assert cells[(80, "0")]["form"] == "goru"
    assert cells[(81, "b")]["form"] == "muʃi"
    assert cells[(81, "l")]["form"] == "tʃɨndɨk"
    assert cells[(81, "0")]["form"] == "mohiʃ"
    assert cells[(81, "q")]["form"] == "bus"


def test_items_82_85_complete_left_column_of_physical_page_48():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_48B)}
    assert cells[(82, "b")]["form"] == "koroŋ"
    assert cells[(82, "0")]["form"] == "ʃiŋ"
    assert cells[(83, "l")]["form"] == "dɨʔmɨ"
    assert cells[(83, "b")]["form"] == "dimai̯"
    assert cells[(83, "0")]["form"] == "lɛdʒ"
    assert cells[(83, "q")]["form"] == "niŋur"
    assert cells[(84, "b")]["form"] == "purun"
    assert cells[(84, "0")]["form"] == "tʃʰagol"
    assert cells[(85, "b")]["status"] == "ambiguous"
    assert cells[(85, "b")]["visible_base"] == "wak"
    assert cells[(85, "0")]["form"] == "ʃukor"


def test_items_86_89_preserve_direct_page_48_right_column_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_48C)}
    assert cells[(86, "b")]["form"] == "moʃai̯"
    assert cells[(86, "m")]["status"] == "ambiguous"
    assert cells[(86, "m")]["visible_base"] == "miʔtʃut"
    assert cells[(86, "m")]["group"] == "2 | 4"
    assert cells[(86, "l")]["visible_base"] == "midʒut"
    assert cells[(86, "0")]["form"] == "idur"
    assert cells[(86, "q")]["form"] == "motʃot"
    assert cells[(87, "b")]["form"] == "tau̯"
    assert cells[(87, "0")]["form"] == "murgi"
    assert cells[(88, "b")]["form"] == "pitɨk"
    assert cells[(88, "q")]["form"] == "tau̯tɨk"
    assert cells[(88, "l")]["form"] == "tɨi̯"
    assert cells[(88, "0")]["form"] == "dim"
    assert cells[(89, "b")]["form"] == "na"
    assert cells[(89, "0")]["form"] == "matʃʰ"


def test_items_90_93_complete_physical_page_48_and_preserve_variant():
    rows = read_tsv(CELLS_48D)
    forms = defaultdict(list)
    cells = {}
    for row in rows:
        forms[(int(row["item"]), row["site_code"])].append(row["form"])
        cells[(int(row["item"]), row["site_code"])] = row
    assert forms[(90, "b")] == ["haŋsu"]
    assert cells[(90, "m")]["visible_base"] == "dao̯gɛp"
    assert cells[(90, "l")]["visible_base"] == "dao̯gɛt"
    assert forms[(90, "0")] == ["haʃ"]
    assert forms[(90, "q")] == ["hantʃuk"]
    assert forms[(91, "b")] == ["tau̯"]
    assert forms[(91, "0")] == ["pakʰi"]
    assert forms[(92, "b")] == ["tʃoŋ"]
    assert forms[(92, "0")] == ["poka"]
    assert forms[(93, "m")] == ["saluŋ", "tɛlapoka"]
    assert forms[(93, "b")] == ["atʃɛp"]
    assert forms[(93, "q")] == ["tɛltʃora"]
    assert forms[(93, "0")] == ["tɛlapoka"]


def test_items_94_97_preserve_direct_page_49_readings_and_variant():
    rows = read_tsv(CELLS_49)
    forms = defaultdict(list)
    cells = {}
    for row in rows:
        forms[(int(row["item"]), row["site_code"])].append(row["form"])
        cells[(int(row["item"]), row["site_code"])] = row
    assert forms[(94, "r")] == ["nija (tʃoŋ)", "nijatʃoŋ"]
    assert forms[(94, "b")] == ["nijatʃoŋ"]
    assert forms[(94, "q")] == ["nɛ"]
    assert forms[(94, "0")] == ["mou̯matʃʰi"]
    assert forms[(95, "b")] == ["mai̯ paratʃoŋ"]
    assert forms[(95, "l")] == ["sot"]
    assert forms[(95, "0")] == ["matʃʰi"]
    assert cells[(96, "b")]["status"] == "ambiguous"
    assert cells[(96, "b")]["visible_base"] == "abrɛk"
    assert forms[(96, "0")] == ["makorʃa"]
    assert forms[(96, "q")] == ["makrɛk"]
    assert forms[(97, "c")] == ["ʃɛmar"]
    assert forms[(97, "q")] == ["ʃimer"]
    assert forms[(97, "l")] == ["samal"]
    assert forms[(97, "b")] == ["sɛmal"]
    assert forms[(97, "r")] == ["sɛmar"]
    assert forms[(97, "0")] == ["pipra"]


def test_items_98_100_complete_left_column_of_physical_page_49():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_49B)}
    assert cells[(98, "l")]["form"] == "gaŋgawa"
    assert cells[(98, "b")]["form"] == "tʃoŋdaŋlaŋ"
    assert cells[(98, "c")]["form"] == "tʃoŋ"
    assert cells[(98, "0")]["form"] == "mɔʃa"
    assert cells[(98, "q")]["form"] == "gaŋgutʃuŋ"
    assert cells[(99, "l")]["form"] == "dɨkɨm"
    assert cells[(99, "b")]["form"] == "dukum"
    assert cells[(99, "0")]["form"] == "matʰa"
    assert cells[(100, "l")]["form"] == "mɨkʰɨŋ"
    assert cells[(100, "b")]["form"] == "mukʰaŋ"
    assert cells[(100, "0")]["form"] == "mukʰ"
    assert cells[(100, "q")]["form"] == "məhəŋ"


def test_items_101_104_preserve_direct_page_49_right_column_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_49C)}
    assert cells[(101, "b")]["form"] == "kalkʰu"
    assert cells[(101, "l")]["form"] == "tokrɛŋ"
    assert cells[(101, "q")]["form"] == "gala"
    assert cells[(101, "0")]["form"] == "gɔla"
    assert cells[(102, "b")]["form"] == "kʰau̯"
    assert cells[(102, "0")]["form"] == "tʃul"
    assert cells[(102, "q")]["form"] == "hʌu̯"
    assert cells[(103, "l")]["form"] == "mɨkrɛŋ"
    assert cells[(103, "b")]["status"] == "ambiguous"
    assert cells[(103, "b")]["visible_base"] == "mokkon"
    assert cells[(103, "b")]["group"] == "1 | 3"
    assert cells[(103, "0")]["form"] == "tʃok"
    assert cells[(103, "q")]["form"] == "nukun"
    assert cells[(104, "b")]["visible_base"] == "nakkuŋ"
    assert cells[(104, "l")]["form"] == "nakʰuŋ"
    assert cells[(104, "q")]["form"] == "nakuŋ"
    assert cells[(104, "0")]["form"] == "nak"


def test_items_105_107_complete_physical_page_49():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_49D)}
    assert cells[(105, "b")]["form"] == "nakʰar"
    assert cells[(105, "q")]["form"] == "nakor"
    assert cells[(105, "0")]["form"] == "kan"
    assert cells[(106, "l")]["form"] == "pʰai̯tʰupa"
    assert cells[(106, "b")]["status"] == "ambiguous"
    assert cells[(106, "b")]["visible_base"] == "pai̯tʰok"
    assert cells[(106, "0")]["form"] == "gal"
    assert cells[(106, "q")]["form"] == "tʃapa"
    assert cells[(107, "l")]["form"] == "kadɨmbai̯"
    assert cells[(107, "b")]["form"] == "katʰolok"
    assert cells[(107, "0")]["form"] == "tʃibuk"
    assert cells[(107, "q")]["form"] == "dadi"


def test_items_108_111_preserve_direct_page_50_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_50)}
    assert cells[(108, "b")]["form"] == "kʰutʃar"
    assert cells[(108, "l")]["status"] == "ambiguous"
    assert cells[(108, "l")]["visible_base"] == "kʰutʃuk"
    assert cells[(108, "0")]["form"] == "muk"
    assert cells[(108, "q")]["form"] == "hotoŋ"
    assert cells[(109, "b")]["form"] == "tʰalai̯"
    assert cells[(109, "l")]["form"] == "tʰɛlampa"
    assert cells[(109, "q")]["form"] == "dʒibʌ"
    assert cells[(109, "0")]["form"] == "dʒib"
    assert cells[(110, "l")]["form"] == "wa"
    assert cells[(110, "b")]["form"] == "tʰa"
    assert cells[(110, "0")]["form"] == "dãt"
    assert cells[(110, "r")]["form"] == "pʰa"
    assert cells[(111, "c")]["form"] == "tʃa gilai̯"
    assert cells[(111, "q")]["form"] == "tʃa gilʌ"
    assert cells[(111, "b")]["visible_base"] == "tʃak gilai̯"
    assert cells[(111, "l")]["visible_base"] == "tʃaktʃuk"
    assert cells[(111, "m")]["visible_base"] == "tʃakwɛŋ"
    assert cells[(111, "0")]["form"] == "konui̯"


def test_items_112_114_preserve_direct_page_50_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_50B)}
    assert cells[(112, "l")]["visible_base"] == "tʃak"
    assert cells[(112, "b")]["visible_base"] == "tʃakaprak"
    assert cells[(112, "0")]["form"] == "hat"
    assert cells[(113, "l")]["visible_base"] == "tʃakpʰa"
    assert cells[(113, "b")]["visible_base"] == "tʃaktala"
    assert cells[(113, "0")]["form"] == "hatɛr tɔla"
    assert cells[(113, "q")]["form"] == "tʃak pata"
    assert cells[(114, "l")]["visible_base"] == "tʃakʃi"
    assert cells[(114, "b")]["form"] == "tʃaʃi"
    assert cells[(114, "0")]["form"] == "aŋgur"
    assert cells[(114, "q")]["form"] == "aŋul"


def test_items_115_118_preserve_direct_page_50_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_50C)}
    assert cells[(115, "m")]["visible_base"] == "tʃakʃikʰor"
    assert cells[(115, "b")]["visible_base"] == "tʃakʃikor"
    assert cells[(115, "q")]["form"] == "tʃatʃikul"
    assert cells[(115, "0")]["form"] == "nɔk"
    assert cells[(116, "b")]["form"] == "tʃakrɛŋgila"
    assert cells[(116, "l")]["form"] == "tʃakɨu̯"
    assert cells[(116, "q")]["form"] == "antʰu"
    assert cells[(116, "r")]["form"] == "atʰu"
    assert cells[(116, "r")]["group"] == "3 | 4"
    assert cells[(116, "0")]["form"] == "hatʰu"
    assert cells[(117, "l")]["form"] == "tʃaʔpʰa"
    assert cells[(117, "b")]["visible_base"] == "tʃakrɛŋ aprak"
    assert cells[(117, "c")]["form"] == "tʃakrɛŋ"
    assert cells[(117, "m")]["form"] == "tʃaʔ"
    assert cells[(117, "0")]["form"] == "pɔd"
    assert cells[(117, "q")]["form"] == "tatʰɛŋ"
    assert cells[(118, "b")]["form"] == "kɛrɛŋ"
    assert cells[(118, "0")]["form"] == "har"


def test_items_119_121_complete_physical_page_50():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_50D)}
    assert cells[(119, "b")]["status"] == "blank"
    assert cells[(119, "l")]["status"] == "blank"
    assert cells[(119, "c")]["status"] == "ambiguous"
    assert cells[(119, "c")]["visible_base"] == "hokpɛkɛn"
    assert cells[(119, "q")]["form"] == "tɛl"
    assert cells[(119, "m")]["form"] == "ludi"
    assert cells[(119, "0")]["form"] == "tʃorbi"
    assert cells[(120, "b")]["status"] == "ambiguous"
    assert cells[(120, "b")]["visible_base"] == "kʰopak"
    assert cells[(120, "l")]["form"] == "kʰol"
    assert cells[(120, "q")]["form"] == "tʃamra"
    assert cells[(121, "l")]["form"] == "tʰɨi̯"
    assert cells[(121, "m")]["form"] == "tʰɨi̯"
    assert cells[(121, "b")]["form"] == "tʰi"
    assert cells[(121, "0")]["form"] == "rɔkto"


def test_items_122_128_preserve_direct_page_51_left_column_readings():
    cells = read_tsv(CELLS_51)
    by_key = defaultdict(list)
    for row in cells:
        by_key[(int(row["item"]), row["site_code"])].append(row)
    assert by_key[(122, "b")][0]["status"] == "blank"
    assert by_key[(122, "m")][0]["form"] == "tuŋgoa̯"
    assert by_key[(122, "l")][0]["visible_base"] == "tuŋok"
    assert by_key[(122, "q")][0]["form"] == "gʰam"
    assert by_key[(123, "b")][0]["visible_base"] == "hok"
    assert by_key[(123, "l")][0]["form"] == "pipʰuʔ"
    assert by_key[(123, "m")][0]["visible_base"] == "pipuk"
    assert by_key[(124, "b")][0]["status"] == "blank"
    assert by_key[(124, "c")][0]["visible_base"] == "pʰoktʃa"
    assert [row["form"] for row in by_key[(124, "l")]] == ["pikʰa", "dʒaŋgi"]
    assert by_key[(124, "0")][0]["form"] == "ridoi̯"
    assert by_key[(125, "q")][0]["form"] == "huŋdʒuŋ"
    assert by_key[(126, "l")][0]["form"] == "randai̯"
    assert by_key[(127, "b")][0]["visible_base"] == "morot"
    assert by_key[(128, "c")][0]["form"] == "purʃi"
    assert by_key[(128, "b")][0]["form"] == "puʃi"
    assert by_key[(128, "q")][0]["form"] == "maʔwa"


def test_items_129_135_complete_physical_page_51():
    cells = read_tsv(CELLS_51B)
    by_key = defaultdict(list)
    for row in cells:
        by_key[(int(row["item"]), row["site_code"])].append(row)
    assert by_key[(129, "l")][0]["form"] == "gɨwui̯"
    assert by_key[(129, "0")][0]["form"] == "mohila"
    assert [row["form"] for row in by_key[(130, "0")]] == ["abba", "baba"]
    assert by_key[(131, "b")][0]["form"] == "amai̯"
    assert by_key[(131, "0")][0]["form"] == "ma / amma"
    assert by_key[(131, "q")][0]["form"] == "ai̯jʌ"
    assert by_key[(132, "l")][0]["visible_base"] == "dʒikbipʰa"
    assert by_key[(132, "0")][0]["form"] == "ʃami"
    assert by_key[(133, "b")][0]["visible_base"] == "mitʃik"
    assert by_key[(133, "l")][0]["visible_base"] == "dʒikgɨwui̯"
    assert by_key[(133, "0")][0]["form"] == "stri"
    assert by_key[(134, "0")][0]["form"] == "tʃʰɛlɛ"
    assert by_key[(134, "q")][0]["form"] == "tʃʰaʃa"
    assert by_key[(135, "c")][0]["form"] == "(tiri) piʃa"
    assert by_key[(135, "b")][0]["form"] == "piʃa (tiri)"
    assert by_key[(135, "l")][0]["form"] == "ʃa gɨwui̯"
    assert by_key[(135, "q")][0]["form"] == "madʒu tʃʰaʃa"


def test_items_136_142_preserve_direct_page_52_left_column_readings():
    cells = read_tsv(CELLS_52)
    by_key = defaultdict(list)
    for row in cells:
        by_key[(int(row["item"]), row["site_code"])].append(row)
    assert [row["form"] for row in by_key[(136, "c")]] == ["dada", "kaka"]
    assert by_key[(136, "m")][0]["form"] == "pʰao̯ tʃuŋguwa"
    assert by_key[(136, "0")][0]["form"] == "bɔro bʰai̯"
    assert [row["form"] for row in by_key[(137, "c")]] == ["adʒa", "bai̯"]
    assert by_key[(137, "0")][0]["form"] == "bɔro bon / didi"
    assert by_key[(138, "l")][0]["form"] == "dʒoŋ goa̯"
    assert by_key[(138, "m")][0]["form"] == "dʒoŋ mɨlguwa"
    assert by_key[(138, "0")][0]["form"] == "tʃʰoto bʰai̯"
    assert by_key[(139, "b")][0]["form"] == "nau̯"
    assert by_key[(139, "m")][0]["form"] == "nao̯ mɨlguwa"
    assert by_key[(139, "q")][0]["form"] == "tamui̯ dʒanao̯"
    assert by_key[(140, "b")][0]["form"] == "ʃaŋgra"
    assert by_key[(140, "c")][0]["form"] == "saŋgra"
    assert by_key[(141, "l")][0]["form"] == "bimuŋ"
    assert by_key[(142, "m")][0]["form"] == "ʃoŋ"


def test_items_143_149_complete_physical_page_52():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_52B)}
    assert cells[(143, "b")]["visible_base"] == "nok"
    assert cells[(143, "0")]["form"] == "bari / gʰor"
    assert cells[(144, "l")]["form"] == "dokor"
    assert cells[(144, "l")]["group"] == "1 | 3"
    assert cells[(144, "0")]["form"] == "dɔrdʒa"
    assert cells[(145, "r")]["status"] == "blank"
    assert cells[(145, "c")]["form"] == "kʰokai̯ duar"
    assert cells[(145, "b")]["visible_base"] == "kʰokai̯ dukat"
    assert cells[(145, "q")]["form"] == "kokri dʰonda"
    assert cells[(146, "b")]["form"] == "nukʰaraŋ"
    assert cells[(146, "c")]["form"] == "nukʰuraŋ"
    assert cells[(146, "c")]["group"] == "1 | 3"
    assert cells[(146, "r")]["form"] == "nukʰraŋ"
    assert cells[(146, "r")]["group"] == "1 | 3"
    assert cells[(146, "0")]["form"] == "tʃʰad / tʃal"
    assert cells[(147, "r")]["form"] == "kʰadʒa"
    assert cells[(148, "b")]["form"] == "balus"
    assert cells[(148, "c")]["form"] == "baluʃ"
    assert cells[(149, "0")]["form"] == "kɔmbol"


def test_items_150_157_preserve_direct_page_53_left_column_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_53)}
    assert cells[(150, "0")]["form"] == "aŋti"
    assert cells[(150, "b")]["form"] == "antʰi"
    assert cells[(151, "q")]["form"] == "ʃoka"
    assert cells[(151, "b")]["form"] == "soka"
    assert {cells[(152, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert cells[(153, "b")]["visible_base"] == "pantʃak"
    assert cells[(153, "q")]["form"] == "paŋkur"
    assert cells[(154, "l")]["visible_base"] == "lɛkkʰa"
    assert cells[(154, "0")]["form"] == "kagodʒ"
    assert cells[(155, "l")]["form"] == "sɨlʃimi"
    assert cells[(155, "0")]["form"] == "ʃutʃ"
    assert cells[(156, "q")]["form"] == "kintiŋ"
    assert cells[(156, "q")]["group"] == "1 | 2"
    assert cells[(157, "b")]["visible_base"] == "nohɛk"
    assert cells[(157, "l")]["visible_base"] == "nogɛk"
    assert cells[(157, "0")]["form"] == "dʒʰaru"


def test_items_158_167_complete_physical_page_53():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_53B)}
    assert cells[(158, "b")]["form"] == "kortʃali"
    assert cells[(158, "c")]["form"] == "kortʃila"
    assert {cells[(159, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert cells[(160, "b")]["form"] == "hatur"
    assert cells[(160, "0")]["form"] == "haturi"
    assert cells[(161, "b")]["form"] == "wakɛŋ"
    assert cells[(161, "0")]["form"] == "kutʰar"
    assert cells[(162, "q")]["form"] == "dʰonuk"
    assert cells[(163, "q")]["form"] == "tir"
    assert cells[(164, "b")]["form"] == "tʃɛwal"
    assert cells[(164, "l")]["form"] == "gutʰini"
    assert cells[(165, "b")]["form"] == "war"
    assert cells[(165, "l")]["form"] == "wal"
    assert cells[(166, "b")]["visible_base"] == "tʰappala"
    assert cells[(166, "m")]["form"] == "tʰapʰra"
    assert cells[(166, "l")]["form"] == "tʰapra"
    assert cells[(166, "0")]["form"] == "tʃʰai̯"
    assert cells[(167, "l")]["form"] == "walkʰu"
    assert cells[(167, "b")]["form"] == "warkʰu"
    assert cells[(167, "0")]["form"] == "dʰõa̯"
    assert cells[(167, "q")]["form"] == "dʰumʌ"


def test_items_168_176_preserve_direct_page_54_left_column_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_54)}
    assert cells[(168, "b")]["form"] == "mombati"
    assert cells[(169, "b")]["form"] == "nau̯ka"
    assert cells[(169, "r")]["form"] == "nou̯ka"
    assert cells[(170, "b")]["form"] == "lam"
    assert cells[(170, "l")]["form"] == "ram"
    assert cells[(170, "0")]["form"] == "rasta / ʃorok"
    assert {cells[(171, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert cells[(172, "b")]["form"] == "lai̯"
    assert cells[(172, "l")]["form"] == "rɛʔɛŋ"
    assert cells[(173, "b")]["form"] == "pʰai̯na"
    assert cells[(173, "l")]["form"] == "pʰina"
    assert cells[(174, "b")]["visible_base"] == "tʃap"
    assert cells[(174, "0")]["form"] == "darano"
    assert cells[(175, "q")]["form"] == "moʃoŋ"
    assert cells[(175, "0")]["form"] == "bɔʃa"
    assert {cells[(176, code)]["status"] for code in "0bclmqr"} == {"not_used"}


def test_items_177_183_complete_physical_page_54():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_54B)}
    assert cells[(177, "b")]["form"] == "bɛrai̯"
    assert cells[(177, "0")]["form"] == "hãta"
    assert cells[(177, "q")]["form"] == "lʌdʒʌmlai̯"
    assert cells[(178, "r")]["status"] == "blank"
    assert cells[(178, "l")]["form"] == "piu̯"
    assert cells[(179, "b")]["form"] == "daŋ"
    assert cells[(179, "0")]["form"] == "dʰoka"
    assert cells[(180, "l")]["status"] == "blank"
    assert cells[(180, "b")]["form"] == "tʃasum"
    assert cells[(180, "r")]["form"] == "taʃum"
    assert cells[(180, "m")]["form"] == "gatʰilni"
    assert cells[(180, "q")]["form"] == "gurei̯ aʃik"
    assert cells[(181, "q")]["form"] == "hatrei̯"
    assert cells[(182, "b")]["form"] == "tʃai̯"
    assert cells[(182, "0")]["form"] == "dɛkʰa"
    assert cells[(183, "m")]["form"] == "naniʔ"
    assert cells[(183, "q")]["form"] == "natʰim"


def test_items_184_191_preserve_direct_page_55_left_column_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_55)}
    assert cells[(184, "b")]["form"] == "ʃam"
    assert cells[(184, "q")]["form"] == "sam"
    assert cells[(184, "0")]["form"] == "opɛkkʰa kɔra"
    assert cells[(185, "b")]["visible_base"] == "kʰɛt"
    assert cells[(185, "l")]["visible_base"] == "kʰɛp"
    assert cells[(185, "0")]["form"] == "kãda"
    assert cells[(186, "b")]["form"] == "lum"
    assert cells[(186, "l")]["form"] == "rɨm"
    assert cells[(187, "b")]["visible_base"] == "rot"
    assert cells[(187, "0")]["form"] == "ʃiddʰo kɔra"
    assert cells[(188, "b")]["form"] == "ʃaʔ"
    assert cells[(188, "m")]["visible_base"] == "ʃat"
    assert cells[(188, "l")]["form"] == "saʔ"
    assert cells[(189, "0")]["form"] == "pani kʰawa"
    assert cells[(190, "l")]["form"] == "riŋʔ"
    assert cells[(190, "b")]["form"] == "gai̯"
    assert cells[(191, "b")]["visible_base"] == "kat"
    assert cells[(191, "q")]["visible_base"] == "kak"
    assert cells[(191, "0")]["form"] == "kamrano"


def test_items_192_199_complete_physical_page_55():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_55B)}
    assert cells[(192, "l")]["form"] == "mɨnei̯"
    assert cells[(192, "q")]["form"] == "mimin"
    assert cells[(192, "0")]["form"] == "haʃa"
    assert cells[(193, "l")]["form"] == "bal"
    assert cells[(193, "b")]["form"] == "bar"
    assert {cells[(194, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert cells[(195, "l")]["form"] == "tiŋ"
    assert cells[(195, "b")]["form"] == "tʰarman"
    assert cells[(196, "b")]["visible_base"] == "wandatlai̯"
    assert cells[(196, "l")]["form"] == "awan"
    assert cells[(197, "l")]["form"] == "dʒɨu̯"
    assert cells[(197, "b")]["form"] == "dʒu"
    assert cells[(198, "b")]["visible_base"] == "dʒamuk nuk"
    assert cells[(198, "l")]["visible_base"] == "dʒumaŋ nuk"
    assert cells[(198, "0")]["form"] == "ʃopno dɛkʰa"
    assert cells[(199, "b")]["form"] == "banai̯"
    assert cells[(199, "m")]["form"] == "tʰari"
    assert cells[(199, "q")]["form"] == "tuɛr nak"


def test_items_200_206_preserve_direct_page_56_left_column_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_56)}
    assert cells[(200, "b")]["form"] == "kam lau̯"
    assert cells[(200, "l")]["form"] == "kam pii̯"
    assert cells[(200, "0")]["form"] == "kadʒ kɔra"
    assert cells[(201, "b")]["form"] == "kʰɛl"
    assert cells[(201, "0")]["form"] == "kʰɛla"
    assert cells[(201, "l")]["form"] == "kʰɛlai̯"
    assert cells[(202, "m")]["form"] == "biʃii̯"
    assert cells[(202, "l")]["form"] == "biʃi"
    assert cells[(202, "0")]["form"] == "natʃa"
    assert cells[(203, "b")]["visible_base"] == "tʰuk"
    assert cells[(203, "l")]["visible_base"] == "tʰip"
    assert cells[(203, "q")]["form"] == "dʌmpʰai̯"
    assert cells[(204, "b")]["form"] == "pai̯"
    assert cells[(204, "l")]["form"] == "pai̯tao̯"
    assert cells[(204, "0")]["visible_base"] == "tola"
    assert cells[(205, "b")]["form"] == "tʃur"
    assert cells[(205, "l")]["visible_base"] == "ʃipʰak"
    assert cells[(205, "m")]["form"] == "ʃikdou̯"
    assert cells[(205, "q")]["form"] == "dʰɛkar hao̯"
    assert cells[(206, "l")]["visible_base"] == "bɨt"
    assert cells[(206, "b")]["visible_base"] == "but"
    assert cells[(206, "0")]["form"] == "ʈana"


def test_items_207_213_complete_physical_page_56():
    cells = defaultdict(list)
    for row in read_tsv(CELLS_56B):
        cells[(int(row["item"]), row["site_code"])].append(row)
    assert cells[(207, "l")][0]["status"] == "blank"
    assert cells[(207, "c")][0]["form"] == "kʰa"
    assert cells[(207, "b")][0]["form"] == "ha"
    assert cells[(207, "0")][0]["form"] == "bãdʰa"
    assert cells[(208, "b")][0]["visible_base"] == "tʃʰit"
    assert cells[(208, "l")][0]["visible_base"] == "ʃɛʃɛt"
    assert cells[(208, "0")][0]["form"] == "motʃʰa"
    assert cells[(209, "m")][0]["visible_base"] == "tok"
    assert cells[(209, "q")][0]["form"] == "bakai̯"
    assert cells[(209, "l")][0]["form"] == "banai̯"
    assert cells[(210, "b")][0]["form"] == "pʰuŋ"
    assert cells[(210, "0")][0]["form"] == "ʃɛlai̯ kɔra"
    assert cells[(211, "q")][0]["form"] == "gin"
    assert cells[(211, "m")][0]["visible_base"] == "ʃuʃut"
    assert [row["form"] for row in cells[(212, "b")]] == ["tilu"]
    assert [row["group"] for row in cells[(212, "b")]] == ["1 | 3"]
    assert cells[(212, "l")][0]["form"] == "tei̯ru"
    assert cells[(213, "b")][0]["form"] == "kʰan"
    assert cells[(213, "0")][0]["form"] == "kaʈa"
    assert cells[(213, "q")][0]["form"] == "handok"


def test_items_214_221_preserve_direct_page_57_left_column_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_57)}
    assert cells[(214, "b")]["form"] == "ʃau̯"
    assert cells[(214, "l")]["form"] == "sau̯"
    assert cells[(214, "0")]["form"] == "porano"
    assert cells[(215, "q")]["form"] == "kinnai̯"
    assert cells[(216, "b")]["form"] == "pʰal"
    assert cells[(216, "0")]["form"] == "bikri kɔra"
    assert cells[(217, "l")]["form"] == "ʃakʰau̯"
    assert cells[(217, "q")]["form"] == "tʃur"
    assert cells[(218, "b")]["form"] == "boja bar"
    assert cells[(218, "l")]["form"] == "tʰolai̯ (bal)"
    assert cells[(218, "0")]["form"] == "mittʰa bɔla"
    assert cells[(219, "b")]["visible_base"] == "nak"
    assert cells[(219, "0")]["form"] == "nɛa"
    assert cells[(219, "q")]["form"] == "laŋ"
    assert cells[(220, "b")]["form"] == "lakʰa"
    assert cells[(220, "l")]["form"] == "hin"
    assert cells[(220, "m")]["visible_base"] == "hut"
    assert {cells[(221, code)]["status"] for code in "0bclmqr"} == {"not_used"}


def test_items_222_228_complete_physical_page_57():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_57B)}
    assert cells[(222, "l")]["form"] == "tʰii̯"
    assert cells[(222, "b")]["form"] == "tʰi"
    assert cells[(222, "0")]["form"] == "mara dʒawa"
    assert cells[(223, "l")]["visible_base"] == "namnuk"
    assert cells[(223, "m")]["visible_base"] == "nɛmnuk"
    assert cells[(223, "b")]["form"] == "milai̯"
    assert cells[(223, "0")]["form"] == "bʰalobaʃa"
    assert cells[(224, "l")]["status"] == "blank"
    assert cells[(224, "q")]["form"] == "kantʃik"
    assert cells[(224, "0")]["form"] == "gʰrina kɔra"
    assert cells[(225, "c")]["form"] == "(gon)ʃa"
    assert cells[(225, "b")]["visible_base"] == "(got)ʃa"
    assert cells[(225, "0")]["form"] == "ek"
    assert cells[(226, "l")]["form"] == "(miŋ)nii̯ ʃa"
    assert cells[(226, "b")]["form"] == "dui̯"
    assert cells[(227, "m")]["form"] == "(go)tʰam"
    assert cells[(227, "b")]["form"] == "tin"
    assert cells[(228, "l")]["form"] == "(miŋ)bri"
    assert cells[(228, "q")]["form"] == "tʃar"


def test_item_229_completes_physical_page_57():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_57C)}
    assert cells[(229, "m")]["form"] == "(go)baŋa"
    assert cells[(229, "l")]["form"] == "(miŋ)baŋa"
    assert cells[(229, "b")]["form"] == "patʃ"


def test_items_230_237_preserve_direct_page_58_left_column_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_58)}
    assert cells[(230, "m")]["visible_base"] == "(go)korok"
    assert cells[(230, "l")]["visible_base"] == "(miŋ)korok"
    assert cells[(230, "b")]["form"] == "tʃʰoi̯"
    assert cells[(231, "m")]["form"] == "(go)ʃini"
    assert cells[(231, "l")]["form"] == "(miŋ)ʃɛnɛ"
    assert cells[(232, "l")]["visible_base"] == "(miŋ)tʃakik"
    assert cells[(232, "m")]["form"] == "gandanii̯"
    assert cells[(232, "0")]["form"] == "aʈ"
    assert cells[(233, "l")]["form"] == "(miŋ)skʰu"
    assert cells[(233, "b")]["form"] == "nɔi̯"
    assert cells[(233, "m")]["form"] == "gandanii̯ goi̯ʃa"
    assert cells[(234, "l")]["visible_base"] == "(miŋ)tʃɛgik"
    assert cells[(235, "0")]["form"] == "ægaro"
    assert cells[(235, "m")]["visible_base"] == "tʃei̯gik ʃa"
    assert cells[(236, "m")]["visible_base"] == "tʃei̯gik nii̯"
    assert cells[(237, "l")]["visible_base"] == "(miŋ)kʰolgik"
    assert cells[(237, "m")]["form"] == "kʰoltʃaŋʃa"


def test_items_238_246_complete_physical_page_58():
    rows = read_tsv(CELLS_58B)
    cells = {(int(row["item"]), row["site_code"]): row for row in rows}
    variants = defaultdict(list)
    for row in rows:
        variants[(int(row["item"]), row["site_code"])].append(row)
    assert cells[(238, "l")]["form"] == "radʒa"
    assert cells[(238, "b")]["form"] == "ʃɔ"
    assert cells[(239, "0")]["form"] == "hadʒar"
    assert {cells[(240, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert cells[(241, "q")]["form"] == "akui̯ʃa"
    assert cells[(241, "b")]["visible_base"] == "tɛp"
    assert cells[(241, "l")]["form"] == "tʃoi̯ʃan"
    assert [(row["status"], row["form"], row["visible_base"])
            for row in variants[(241, "r")]] == [
                ("attested", "akui̯ʃa", ""), ("ambiguous", "", "tɛp")
            ]
    assert cells[(242, "b")]["form"] == "tamti"
    assert cells[(242, "l")]["form"] == "panʃembia̯"
    assert cells[(242, "0")]["form"] == "ɔnɛk"
    assert [row["form"] for row in variants[(242, "r")]] == ["paŋa", "ɔnɛk"]
    assert cells[(243, "b")]["status"] == "blank"
    assert cells[(243, "c")]["visible_base"] == "bɛbak"
    assert cells[(243, "l")]["form"] == "dʒamai̯n"
    assert cells[(244, "c")]["visible_base"] == "matta"
    assert cells[(244, "l")]["visible_base"] == "tʃuŋ(bijok)"
    assert cells[(244, "m")]["form"] == "tʃuŋa"
    assert cells[(245, "r")]["form"] == "tukti"
    assert cells[(245, "q")]["form"] == "tamui̯"
    assert cells[(246, "l")]["visible_base"] == "rao̯twa"
    assert cells[(246, "b")]["form"] == "pilao"


def test_items_247_253_preserve_direct_page_59_left_column_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_59)}
    assert cells[(247, "b")]["form"] == "banda"
    assert cells[(247, "l")]["form"] == "suŋa"
    assert cells[(247, "0")]["form"] == "kʰato"
    assert cells[(248, "b")]["form"] == "liu"
    assert cells[(248, "l")]["form"] == "tʃirima"
    assert cells[(248, "q")]["form"] == "bʰar"
    assert cells[(249, "l")]["form"] == "tʃɛŋa"
    assert cells[(249, "r")]["form"] == "tʃɛŋni"
    assert cells[(249, "0")]["form"] == "halka"
    assert cells[(250, "b")]["form"] == "tʃadara"
    assert cells[(250, "c")]["form"] == "tʃadra"
    assert cells[(250, "0")]["form"] == "moʈa"
    assert cells[(251, "m")]["status"] == "blank"
    assert cells[(251, "l")]["form"] == "mɨl"
    assert cells[(251, "q")]["form"] == "tʃikon"
    assert {cells[(252, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert {cells[(253, code)]["status"] for code in "0bclmqr"} == {"not_used"}


def test_item_254_completes_page_59_left_column():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_59B)}
    assert cells[(254, "b")]["form"] == "doŋgor"
    assert cells[(254, "c")]["form"] == "duŋgur"
    assert cells[(254, "m")]["form"] == "tʰoʔwa"
    assert cells[(254, "l")]["visible_base"] == "tʰotwa"
    assert cells[(254, "0")]["form"] == "gobʰir"
    assert cells[(254, "q")]["form"] == "oŋor"


def test_items_255_261_complete_page_59_right_column():
    rows = read_tsv(CELLS_59C)
    cells = {(int(row["item"]), row["site_code"]): row for row in rows}
    variants = defaultdict(list)
    for row in rows:
        variants[(int(row["item"]), row["site_code"])].append(row)
    assert {cells[(255, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert cells[(256, "c")]["form"] == "tiptip"
    assert cells[(256, "m")]["form"] == "pʰiŋa"
    assert cells[(256, "b")]["form"] == "purno"
    assert cells[(256, "q")]["form"] == "pʰuŋni"
    assert cells[(256, "r")]["form"] == "ʃɛrni"
    assert cells[(257, "l")]["status"] == "blank"
    assert cells[(257, "b")]["form"] == "baba"
    assert cells[(257, "m")]["form"] == "kantra"
    assert cells[(257, "r")]["form"] == "kʰali"
    assert cells[(257, "q")]["form"] == "ɛra"
    assert cells[(258, "m")]["form"] == "okʰi"
    assert cells[(258, "b")]["form"] == "ukʰai̯"
    assert cells[(258, "l")]["form"] == "ukʰi"
    assert cells[(258, "0")]["form"] == "kʰidɛ laga"
    assert cells[(258, "q")]["form"] == "mai̯ hoi̯to"
    assert {cells[(259, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert cells[(260, "q")]["form"] == "ʃuma"
    assert cells[(260, "q")]["group"] == "1 | 2"
    assert len(variants[(260, "q")]) == 1
    assert cells[(260, "b")]["form"] == "ʃumni"
    assert cells[(260, "l")]["form"] == "sɨma"
    assert cells[(260, "0")]["form"] == "miʃti"
    assert cells[(261, "b")]["form"] == "hui̯ni"
    assert cells[(261, "c")]["form"] == "kʰui̯ni"
    assert cells[(261, "q")]["form"] == "hini"
    assert cells[(261, "l")]["form"] == "kʰɨja"
    assert cells[(261, "0")]["form"] == "ʈok"


def test_items_262_268_preserve_direct_page_60_left_column_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_60)}
    assert cells[(262, "l")]["form"] == "kʰa.a"
    assert cells[(262, "b")]["form"] == "kʰani"
    assert cells[(262, "0")]["form"] == "tita"
    assert cells[(262, "q")]["form"] == "ha.a"
    assert cells[(263, "b")]["form"] == "burni"
    assert cells[(263, "l")]["form"] == "wɛla"
    assert cells[(263, "0")]["form"] == "dʒʰal"
    assert cells[(264, "m")]["form"] == "minna"
    assert cells[(264, "q")]["form"] == "munni"
    assert cells[(265, "l")]["visible_base"] == "sawok"
    assert cells[(265, "q")]["form"] == "piʃa utrʌ"
    assert cells[(265, "b")]["form"] == "ʃau̯ni"
    assert cells[(265, "r")]["form"] == "sau̯ni"
    assert cells[(265, "0")]["form"] == "poʈʃa"
    assert cells[(266, "b")]["form"] == "dɛtʃiŋni"
    assert cells[(266, "0")]["form"] == "taɽataɽi"
    assert cells[(267, "l")]["form"] == "kʰaʃin"
    assert cells[(267, "b")]["visible_base"] == "tɛp"
    assert cells[(267, "0")]["form"] == "dʰirɛ dʰirɛ"
    assert cells[(268, "c")]["form"] == "gonʃɛn"
    assert cells[(268, "r")]["form"] == "gotsɛn"
    assert cells[(268, "b")]["form"] == "gunsɛn"
    assert cells[(268, "l")]["form"] == "apsan"


def test_items_269_275_complete_physical_page_60():
    rows = read_tsv(CELLS_60B)
    cells = {(int(row["item"]), row["site_code"]): row for row in rows}
    assert cells[(269, "l")]["form"] == "diŋtʰaŋ"
    assert cells[(269, "b")]["visible_base"] == "bɛgat"
    assert cells[(269, "0")]["form"] == "bʰinno"
    assert cells[(269, "q")]["form"] == "dʒudʌ"
    assert cells[(270, "m")]["form"] == "raʔna"
    assert cells[(270, "m")]["group"] == "1 | 2"
    assert cells[(270, "b")]["form"] == "ranni"
    assert cells[(270, "l")]["visible_base"] == "raʔnok"
    assert cells[(271, "m")]["form"] == "pei̯ʃia"
    assert cells[(271, "l")]["visible_base"] == "pei̯ʃok"
    assert cells[(271, "0")]["form"] == "bʰidʒa"
    assert cells[(272, "l")]["form"] == "duŋʔa"
    assert cells[(272, "c")]["form"] == "gumbarto"
    assert cells[(272, "q")]["form"] == "gɔrom"
    assert cells[(273, "l")]["visible_base"] == "tʃikka"
    assert cells[(273, "b")]["form"] == "tʃikni"
    assert cells[(273, "0")]["form"] == "tʰanɖa"
    assert cells[(274, "b")]["form"] == "pɛ.ɛn"
    assert cells[(274, "0")]["form"] == "bʰalo"
    assert cells[(275, "l")]["form"] == "nɛmʈʃa"
    assert cells[(275, "b")]["form"] == "natʰi"
    assert cells[(275, "r")]["form"] == "nattʰi"
    assert cells[(275, "0")]["form"] == "kʰarap"


def test_items_276_282_preserve_direct_page_61_left_column_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_61)}
    assert cells[(276, "b")]["form"] == "pidan"
    assert cells[(276, "0")]["form"] == "notun"
    assert cells[(277, "l")]["status"] == "blank"
    assert cells[(277, "q")]["form"] == "purʌn"
    assert cells[(278, "q")]["form"] == "bai̯ni"
    assert cells[(278, "l")]["form"] == "bai̯.oʔ"
    assert cells[(278, "0")]["form"] == "bʰaŋga"
    assert cells[(279, "l")]["form"] == "kʰambai̯"
    assert cells[(279, "q")]["form"] == "upurʌŋ"
    assert cells[(280, "r")]["form"] == "kama"
    assert cells[(280, "b")]["form"] == "kukma"
    assert cells[(280, "0")]["form"] == "nitʃɛ"
    assert cells[(281, "b")]["form"] == "dʒanni"
    assert cells[(281, "r")]["form"] == "dʒanu"
    assert cells[(282, "0")]["form"] == "katʃʰɛ"
    assert cells[(282, "q")]["form"] == "osorai̯"


def test_items_283_291_complete_physical_page_61():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_61B)}
    assert cells[(283, "b")]["form"] == "dʒagara"
    assert cells[(283, "0")]["form"] == "ɖan"
    assert cells[(284, "l")]["form"] == "dʒagiʃi"
    assert cells[(284, "b")]["form"] == "dɛbara"
    assert cells[(285, "b")]["visible_base"] == "pinak"
    assert cells[(285, "q")]["form"] == "nɛka"
    assert cells[(286, "l")]["visible_base"] == "boka/gippok"
    assert cells[(286, "b")]["visible_base"] == "pɛbok"
    assert cells[(286, "0")]["form"] == "ʃada"
    assert cells[(287, "b")]["form"] == "ʃakraŋ"
    assert cells[(287, "q")]["form"] == "raŋa"
    assert {cells[(288, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert {cells[(289, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert cells[(290, "0")]["form"] == "kɔkʰon"
    assert cells[(290, "q")]["form"] == "biʃumai̯"
    assert cells[(291, "b")]["form"] == "bɛʃoŋ"
    assert cells[(291, "m")]["form"] == "biʃaŋ"
    assert cells[(291, "0")]["form"] == "kotʰai̯"


def test_items_292_300_preserve_direct_page_62_left_column_readings():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_62)}
    assert cells[(292, "b")]["form"] == "tʃaŋ"
    assert cells[(292, "0")]["form"] == "kɛ"
    assert cells[(293, "b")]["form"] == "bita"
    assert cells[(293, "m")]["form"] == "atoŋ"
    assert {cells[(294, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert cells[(295, "l")]["form"] == "i̯jɛ"
    assert cells[(295, "b")]["form"] == "idʒa"
    assert cells[(295, "q")]["form"] == "ja.a"
    assert cells[(296, "q")]["form"] == "hoa"
    assert cells[(296, "b")]["form"] == "udʒa"
    assert cells[(296, "m")]["form"] == "uwɛ"
    assert cells[(297, "b")]["form"] == "idʒoroŋ"
    assert cells[(297, "0")]["form"] == "ɛgulo"
    assert cells[(298, "b")]["form"] == "udʒoroŋ"
    assert cells[(298, "l")]["form"] == "uraŋ"
    assert cells[(299, "q")]["form"] == "aŋ"
    assert cells[(300, "0")]["form"] == "tumi"


def test_items_301_307_complete_koch_wordlist():
    cells = {(int(row["item"]), row["site_code"]): row for row in read_tsv(CELLS_62B)}
    assert {cells[(301, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert cells[(302, "r")]["form"] == "odʒa"
    assert cells[(302, "b")]["form"] == "udʒa"
    assert cells[(302, "0")]["form"] == "ʃɛ"
    assert cells[(302, "q")]["form"] == "wa.a"
    assert {cells[(303, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert cells[(304, "b")]["form"] == "nindra"
    assert cells[(304, "m")]["form"] == "nanaŋ"
    assert cells[(304, "q")]["form"] == "nuŋ"
    assert cells[(305, "b")]["form"] == "nandra"
    assert cells[(305, "0")]["form"] == "tomra"
    assert {cells[(306, code)]["status"] for code in "0bclmqr"} == {"not_used"}
    assert cells[(307, "c")]["form"] == "odra"
    assert cells[(307, "b")]["form"] == "udara"
    assert cells[(307, "m")]["form"] == "itim"
    assert cells[(307, "q")]["form"] == "utupru"


def test_manifest_keeps_the_unreviewed_remainder_pending_with_source_ready():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["state"] == "manual_review_complete"
    chunks = {tuple(row["items"]): row for row in manifest["manual_chunks"]}
    assert chunks[(1, 13)]["source_image_sha256"] == (
        "88d00344a48875188a993df51cecd9eb2731c3af331eee9ec54a08486ee8c3f4"
    )
    assert chunks[(14, 18)]["source_image_sha256"] == (
        "9130020f377bfb4fe457bb9f566ce8a06dfa3bd2ab69c5e71a8da3cd85ef8d4a"
    )
    assert chunks[(29, 33)]["source_image_sha256"] == (
        "d126add19dfff95349d4ef0609a61863171bc1e5df4ab1e75b243a9ca3e897af"
    )
    assert manifest["items_reviewed"] == [1, 307]
    assert manifest["pending_items"] == []
    assert manifest["source_pdf_sha256"] == (
        "d1b2d597c16fd0338ad47d2bf031566192c5ff4e26a6651de14a228df681fc10"
    )
    assert manifest["source_pdf_pages"] == 91
    assert manifest["wordlist_render"]["physical_pages"] == [43, 62]
    assert manifest["wordlist_render"]["rendered_page_count"] == 20
    assert "legacy glyphs, OCR, PDF text" in manifest["policy"]
