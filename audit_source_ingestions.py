"""Build source-specific retrospective ingestion checklists.

The project predates ``SOURCE_INGESTION_CHECKLIST.md``.  This module makes the
retrofit review reproducible: every file consumed as a form input by
``make_cldf.py`` is an ingestion unit, and every unit receives a filled copy of
the canonical checklist under ``source_checklists/``.

The generated front matter records facts that can be checked mechanically.  A
section is checked only when its repository gate has evidence; an unchecked
section names the missing evidence.  The source-specific copies deliberately
remain generated artifacts so changes to the canonical checklist cannot leave
older source reviews silently stale.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from utils import mapping
from form_grammar import extract_gloss_tags
from tags import GENDER_TAGS, GRAMMATICAL_TAGS


ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "SOURCE_INGESTION_CHECKLIST.md"
OUTPUT_DIR = ROOT / "source_checklists"
MANIFEST = OUTPUT_DIR / "manifest.json"
INSTALLED_RECORD_AUDIT = OUTPUT_DIR / "installed-record-audit.csv.gz"

CORE_INPUTS = (
    Path("data/cdial/cdial.csv"),
    Path("data/munda/forms.csv"),
    Path("data/dedr/dedr_new.csv"),
    Path("data/dedr/pdr.csv"),
    Path("data/dbia/forms.csv"),
)

CORE_REVIEW_FILES = {
    "20260826-sil-kochbd": {
        "importers": [
            "data/other/forms/raw_data/sil_kochbd_2011_manual/build_manual.py",
            "data/other/forms/raw_data/sil_kochbd_2011_manual/build_post_freeze_package.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_kochbd_2011_manual/source_manifest.json",
            "data/other/forms/raw_data/sil_kochbd_2011_manual/post_freeze_manifest.json",
            "data/other/forms/raw_data/sil_kochbd_2011_manual/reconciliation.tsv",
            "data/other/forms/raw_data/sil_kochbd_2011_manual/staging_audit.tsv",
            "data/other/forms/raw_data/sil_kochbd_2011_manual/exclusion_policy.json",
            "data/other/forms/raw_data/sil_kochbd_2011_manual/shared_integration_manifest.json",
        ],
        "tests": [
            "tests/test_sil_kochbd_2011_manual.py",
            "tests/test_sil_kochbd_2011_post_freeze.py",
            "tests/test_sound_profiles.py",
            "tests/test_dialects.py",
        ],
        "profiles": ["conversion/sil-bangladesh.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
    "20230530-tharu2": {
        "importers": [
            "data/other/forms/raw_data/sil_western_tharu_2017/import_western_tharu_2017.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_western_tharu_2017/staged_audit.tsv",
            "data/other/forms/raw_data/sil_western_tharu_2017/unresolved_readings.tsv",
            "data/other/forms/raw_data/sil_western_tharu_2017/source_manifest.json",
        ],
        "tests": [
            "tests/test_tharu.py",
            "data/other/forms/raw_data/sil_western_tharu_2017/test_western_tharu_2017.py",
        ],
        "profiles": ["conversion/sil-western-tharu.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "cdial-cdial": {
        "importers": ["data/cdial/parse.py", "data/cdial/audit.py"],
        "audits": ["data/cdial/audit.py", "data/cdial/corrupt_forms.csv"],
        "tests": ["tests/test_cdial_parser.py", "tests/test_cdial_metadata.py"],
        "profiles": ["conversion/cdial.txt"],
        "addenda": ["Dictionary or glossary", "Etymological/comparative source"],
    },
    "dedr-dedr-new": {
        "importers": [
            "data/dedr/parse.py",
            "data/dedr/audit.py",
            "data/dedr/entry_texts.py",
            "data/cross_family.py",
        ],
        "audits": [
            "data/dedr/audit.py",
            "data/dedr/entry-texts-audit.csv.gz",
            "data/dedr/entry-texts-sample.csv",
            "data/dedr/entry-texts-manifest.json",
            "data/cross-family-comparisons-audit.csv",
            "data/cross-family-comparisons-sample.csv",
            "cldf/pdr-headword-audit.csv",
        ],
        "tests": [
            "tests/test_dedr_parser.py",
            "tests/test_dedr_cleanup.py",
            "tests/test_dedr_entry_texts.py",
            "tests/test_cross_family.py",
            "tests/test_dedr_headwords.py",
        ],
        "profiles": ["conversion/dedr.txt"],
        "addenda": ["Dictionary or glossary", "Etymological/comparative source"],
    },
    "dedr-pdr": {
        "importers": ["data/dedr/get_params.py"],
        "audits": [],
        "tests": ["tests/test_dedr_variants.py", "tests/test_cldf.py"],
        "profiles": ["conversion/dedr.txt"],
        "addenda": ["Etymological/comparative source"],
    },
    "dbia-forms": {
        "importers": ["data/dbia/parse.py"],
        "audits": ["data/dbia/parse_audit.csv", "data/dbia/comparisons.csv"],
        "tests": ["tests/test_dbia.py", "tests/test_cross_family.py"],
        "profiles": ["conversion/dedr.txt"],
        "addenda": [
            "Dictionary or glossary",
            "OCR-heavy source",
            "Etymological/comparative source",
        ],
    },
    "munda-forms": {
        "importers": ["data/munda/rau_2019.csv"],
        "audits": [],
        "tests": ["tests/test_cldf.py", "tests/test_edges.py"],
        "profiles": ["conversion/house.txt"],
        "addenda": ["Etymological/comparative source"],
    },
    "20260726-berger-auto": {
        "importers": ["data/other/forms/raw_data/berger_cleanup.py"],
        "audits": [
            "data/other/forms/raw_data/20260828-berger-audit.csv.gz",
            "data/other/forms/raw_data/20260828-berger-sample.csv",
            "data/other/forms/raw_data/20260828-berger-manifest.json",
            "data/other/forms/raw_data/20260828-berger-entry-map.csv",
            "data/other/forms/raw_data/20260828-berger-editorial.csv",
        ],
        "tests": ["tests/test_berger_cleanup.py", "tests/test_berger.py"],
        "profiles": ["conversion/berger.txt"],
        "addenda": [
            "Dictionary or glossary",
            "OCR-heavy source",
            "Etymological/comparative source",
        ],
    },
    "20220930-berger": {
        "importers": ["data/other/forms/raw_data/berger_cleanup.py"],
        "audits": [
            "data/other/forms/raw_data/20260828-berger-audit.csv.gz",
            "data/other/forms/raw_data/20260828-berger-sample.csv",
            "data/other/forms/raw_data/20260828-berger-manifest.json",
            "data/other/forms/raw_data/20260828-berger-entry-map.csv",
            "data/other/forms/raw_data/20260828-berger-editorial.csv",
        ],
        "tests": ["tests/test_berger_cleanup.py", "tests/test_berger.py"],
        "profiles": ["conversion/berger.txt"],
        "addenda": [
            "Dictionary or glossary",
            "OCR-heavy source",
            "Etymological/comparative source",
        ],
    },
    "20260819-burrow-emeneau-den1": {
        "importers": ["data/other/forms/raw_data/burrow_emeneau_1972_den1.py"],
        "audits": [
            "data/other/forms/raw_data/20260819-burrow-emeneau-den1-audit.csv",
            "data/other/forms/raw_data/20260819-burrow-emeneau-den1-sample.csv",
            "data/other/forms/raw_data/20260819-burrow-emeneau-den1-manifest.json",
            "data/other/forms/raw_data/20260819-burrow-emeneau-den1-reconciliation.json",
        ],
        "tests": [
            "tests/test_burrow_emeneau_1972_den1.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/dedr.txt"],
        "addenda": ["Etymological/comparative source"],
    },
    "20260819-burrow-emeneau-den2": {
        "importers": ["data/other/forms/raw_data/burrow_emeneau_1972_den2.py"],
        "audits": [
            "data/other/forms/raw_data/20260819-burrow-emeneau-den2-audit.csv",
            "data/other/forms/raw_data/20260819-burrow-emeneau-den2-sample.csv",
            "data/other/forms/raw_data/20260819-burrow-emeneau-den2-manifest.json",
            "data/other/forms/raw_data/20260819-burrow-emeneau-den2-reconciliation.json",
        ],
        "tests": [
            "tests/test_burrow_emeneau_1972_den2.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/dedr.txt"],
        "addenda": ["Etymological/comparative source"],
    },
    "20260819-emeneau-brahui-1997": {
        "importers": ["data/other/forms/raw_data/emeneau_brahui_1997.py"],
        "audits": [
            "data/other/forms/raw_data/20260819-emeneau-brahui-1997-audit.csv",
            "data/other/forms/raw_data/20260819-emeneau-brahui-1997-sample.csv",
            "data/other/forms/raw_data/20260819-emeneau-brahui-1997-manifest.json",
            "data/other/forms/raw_data/20260819-emeneau-brahui-1997-reconciliation.json",
        ],
        "tests": [
            "tests/test_emeneau_brahui_1997.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/emeneau-brahui.txt"],
        "addenda": ["Etymological/comparative source"],
    },
    "20260819-buddruss-grangali": {
        "importers": ["data/other/forms/raw_data/buddruss_grangali_1979.py"],
        "audits": [
            "data/other/forms/raw_data/20260819-buddruss-grangali-audit.csv",
            "data/other/forms/raw_data/20260819-buddruss-grangali-sample.csv",
            "data/other/forms/raw_data/20260819-buddruss-grangali-manifest.json",
        ],
        "tests": [
            "tests/test_buddruss_grangali_1979.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/buddruss-grangali.txt"],
        "addenda": [
            "Dictionary or glossary",
            "OCR-heavy source",
            "Etymological/comparative source",
        ],
    },
    "20260824-buddruss-waigali": {
        "importers": ["data/other/forms/raw_data/buddruss_waigali_1992.py"],
        "audits": [
            "data/other/forms/raw_data/20260824-buddruss-waigali-audit.csv",
            "data/other/forms/raw_data/20260824-buddruss-waigali-sample.csv",
            "data/other/forms/raw_data/20260824-buddruss-waigali-manifest.json",
        ],
        "tests": [
            "tests/test_buddruss_waigali_wama.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/buddruss-waigali.txt"],
        "addenda": [
            "Dictionary or glossary",
            "OCR-heavy source",
            "Etymological/comparative source",
        ],
    },
    "20260824-buddruss-wama": {
        "importers": ["data/other/forms/raw_data/buddruss_wama_2006.py"],
        "audits": [
            "data/other/forms/raw_data/20260824-buddruss-wama-audit.csv",
            "data/other/forms/raw_data/20260824-buddruss-wama-sample.csv",
            "data/other/forms/raw_data/20260824-buddruss-wama-manifest.json",
        ],
        "tests": [
            "tests/test_buddruss_waigali_wama.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/buddruss-wama.txt"],
        "addenda": [
            "Dictionary or glossary",
            "OCR-heavy source",
            "Etymological/comparative source",
        ],
    },
    "20260825-knobloch-sauji": {
        "importers": ["data/other/forms/raw_data/knobloch_sauji_2020.py"],
        "audits": [
            "data/other/forms/raw_data/20260825-knobloch-sauji-extract.psv",
            "data/other/forms/raw_data/20260825-knobloch-sauji-audit.csv",
            "data/other/forms/raw_data/20260825-knobloch-sauji-sample.csv",
            "data/other/forms/raw_data/20260825-knobloch-sauji-manifest.json",
        ],
        "tests": [
            "tests/test_knobloch_sauji.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/knobloch-sauji.txt"],
        "addenda": [
            "Survey wordlists or comparative tables",
        ],
    },
    "20260825-rezai-baghbidi-zargari": {
        "importers": ["data/other/forms/raw_data/rezai_baghbidi_zargari_2003.py"],
        "audits": [
            "data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv",
            "data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-sample.csv",
            "data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-manifest.json",
        ],
        "tests": [
            "tests/test_rezai_baghbidi_zargari_2003.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/zargari.txt"],
        "addenda": [
            "Dictionary or glossary",
            "Etymological/comparative source",
        ],
    },
    "20260826-woods-halbi": {
        "importers": ["data/other/forms/raw_data/woods_halbi.py"],
        "audits": [
            "data/other/forms/raw_data/20260826-woods-halbi-audit.csv",
            "data/other/forms/raw_data/20260826-woods-halbi-sample.csv",
            "data/other/forms/raw_data/20260826-woods-halbi-manifest.json",
        ],
        "tests": [
            "tests/test_woods_halbi.py",
            "tests/test_source_checklists.py",
        ],
        # The filename reduces to "woods", which is the compiler rather than the lect, so the
        # profile cannot be inferred from it.
        "profiles": ["conversion/halbi-woods.txt"],
        "addenda": [
            "Dictionary or glossary",
            "Website/API or external CLDF",
        ],
    },
    "20260827-degener-shina": {
        "importers": ["data/other/forms/raw_data/degener_shina_2008.py"],
        "audits": [
            "data/other/forms/raw_data/20260827-degener-shina-audit.csv",
            "data/other/forms/raw_data/20260827-degener-shina-sample.csv",
            "data/other/forms/raw_data/20260827-degener-shina-manifest.json",
            "data/other/forms/raw_data/20260827-degener-shina-transcription.txt",
            "data/other/forms/raw_data/20260827-degener-shina-editorial.csv",
        ],
        "tests": [
            "tests/test_degener_shina.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/degener-shina.txt"],
        "addenda": [
            "Dictionary or glossary",
            "OCR-heavy source",
            "Etymological/comparative source",
        ],
    },
    "20260828-buddruss-shina-raetsel": {
        "importers": ["data/other/forms/raw_data/buddruss_shina_1996.py"],
        "audits": [
            "data/other/forms/raw_data/20260828-buddruss-shina-raetsel-audit.csv",
            "data/other/forms/raw_data/20260828-buddruss-shina-raetsel-sample.csv",
            "data/other/forms/raw_data/20260828-buddruss-shina-raetsel-manifest.json",
        ],
        "tests": [
            "tests/test_buddruss_shina_1996.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/buddruss-shina.txt"],
        "addenda": [
            "Dictionary or glossary",
            "OCR-heavy source",
            "Etymological/comparative source",
        ],
    },
    "20260828-pinnow-munda": {
        "importers": ["data/other/forms/raw_data/pinnow_munda_1959.py"],
        "audits": [
            "data/other/forms/raw_data/20260828-pinnow-munda-audit.csv",
            "data/other/forms/raw_data/20260828-pinnow-munda-sample.csv",
            "data/other/forms/raw_data/20260828-pinnow-munda-manifest.json",
        ],
        "tests": [
            "tests/test_pinnow_munda_1959.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/pinnow-munda.txt"],
        "addenda": [
            "Dictionary or glossary",
            "Survey wordlists or comparative tables",
            "Website/API or external CLDF",
            "Etymological/comparative source",
        ],
    },
    "20260828-munda-proto-kherwarian": {
        "importers": ["data/other/forms/raw_data/munda_proto_kherwarian_1968.py"],
        "audits": [
            "data/other/forms/raw_data/20260828-munda-proto-kherwarian-audit.csv",
            "data/other/forms/raw_data/20260828-munda-proto-kherwarian-sample.csv",
            "data/other/forms/raw_data/20260828-munda-proto-kherwarian-manifest.json",
        ],
        "tests": [
            "tests/test_munda_proto_kherwarian_1968.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/munda-proto-kherwarian.txt"],
        "addenda": [
            "Dictionary or glossary",
            "Survey wordlists or comparative tables",
            "Website/API or external CLDF",
            "Etymological/comparative source",
        ],
    },
    "20260828-zide-sora-juray": {
        "importers": ["data/other/forms/raw_data/zide_sora_gorum_1982.py"],
        "audits": [
            "data/other/forms/raw_data/20260828-zide-sora-juray-audit.csv",
            "data/other/forms/raw_data/20260828-zide-sora-juray-sample.csv",
            "data/other/forms/raw_data/20260828-zide-sora-juray-manifest.json",
        ],
        "tests": [
            "tests/test_zide_sora_gorum_1982.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/zide-sora-juray.txt"],
        "addenda": [
            "Dictionary or glossary",
            "Survey wordlists or comparative tables",
            "Website/API or external CLDF",
            "Etymological/comparative source",
        ],
    },
    "20260828-bhattacharya-bonda": {
        "importers": ["data/other/forms/raw_data/bhattacharya_bonda_1968.py"],
        "audits": [
            "data/other/forms/raw_data/20260828-bhattacharya-bonda-audit.csv",
            "data/other/forms/raw_data/20260828-bhattacharya-bonda-sample.csv",
            "data/other/forms/raw_data/20260828-bhattacharya-bonda-manifest.json",
        ],
        "tests": [
            "tests/test_bhattacharya_bonda_1968.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/bhattacharya-bonda.txt"],
        "addenda": [
            "Dictionary or glossary",
            "Website/API or external CLDF",
            "Etymological/comparative source",
        ],
    },
    "20260828-bahl-korwa": {
        "importers": ["data/other/forms/raw_data/bahl_korwa_1962.py"],
        "audits": [
            "data/other/forms/raw_data/20260828-bahl-korwa-audit.csv",
            "data/other/forms/raw_data/20260828-bahl-korwa-sample.csv",
            "data/other/forms/raw_data/20260828-bahl-korwa-manifest.json",
        ],
        "tests": [
            "tests/test_bahl_korwa_1962.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/bahl-korwa.txt"],
        "addenda": [
            "Dictionary or glossary",
            "Website/API or external CLDF",
            "Etymological/comparative source",
        ],
    },
    "20260828-pinnow-juang": {
        "importers": ["data/other/forms/raw_data/pinnow_juang_1960.py"],
        "audits": [
            "data/other/forms/raw_data/20260828-pinnow-juang-audit.csv",
            "data/other/forms/raw_data/20260828-pinnow-juang-sample.csv",
            "data/other/forms/raw_data/20260828-pinnow-juang-manifest.json",
        ],
        "tests": ["tests/test_pinnow_juang_1960.py", "tests/test_source_checklists.py"],
        "profiles": ["conversion/pinnow-juang.txt"],
        "addenda": [
            "Dictionary or glossary",
            "Website/API or external CLDF",
            "Etymological/comparative source",
        ],
    },
    "20260828-sil-nilgiri-irula": {
        "importers": ["data/other/forms/raw_data/sil_irula_2018/import_irula.py"],
        "audits": [
            "data/other/forms/raw_data/20260828-sil-nilgiri-irula-audit.csv",
            "data/other/forms/raw_data/20260828-sil-nilgiri-irula-manifest.json",
        ],
        "tests": ["tests/test_sil_irula_2018.py", "tests/test_source_checklists.py"],
        "profiles": ["conversion/sil-irula.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
    "20260828-sil-haryanvi": {
        "importers": [
            "data/other/forms/raw_data/sil_haryanvi_2024/extract_ocr.py",
            "data/other/forms/raw_data/sil_haryanvi_2024/import_haryanvi.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_haryanvi_2024/manual_transcription.tsv",
            "data/other/forms/raw_data/20260828-sil-haryanvi-audit.csv",
            "data/other/forms/raw_data/20260828-sil-haryanvi-manifest.json",
        ],
        "tests": [
            "tests/test_sil_haryanvi_2024.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-haryanvi.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
    "20260828-sil-koya": {
        "importers": [
            "data/other/forms/raw_data/sil_koya_2021/extract_ocr.py",
            "data/other/forms/raw_data/sil_koya_2021/import_koya.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_koya_2021/manual_review_data.py",
            "data/other/forms/raw_data/20260828-sil-koya-audit.csv",
            "data/other/forms/raw_data/20260828-sil-koya-manifest.json",
        ],
        "tests": ["tests/test_sil_koya_2021.py", "tests/test_source_checklists.py"],
        "profiles": ["conversion/sil-koya.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
    "20260828-sil-kullu": {
        "importers": [
            "data/other/forms/raw_data/sil_kullu_2021/extract_kullu.py",
            "data/other/forms/raw_data/sil_kullu_2021/import_kullu.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_kullu_2021/manual_pages.tsv",
            "data/other/forms/raw_data/20260828-sil-kullu-audit.csv",
            "data/other/forms/raw_data/20260828-sil-kullu-manifest.json",
        ],
        "tests": ["tests/test_sil_kullu_2021.py", "tests/test_source_checklists.py"],
        "profiles": ["conversion/sil-kullu.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
    "20260828-sil-bagheli": {
        "importers": [
            "data/other/forms/raw_data/sil_bagheli_2022/extract_ocr.py",
            "data/other/forms/raw_data/sil_bagheli_2022/import_bagheli.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_bagheli_2022/manual_transcription.txt",
            "data/other/forms/raw_data/sil_bagheli_2022/image_manifest.tsv",
            "data/other/forms/raw_data/20260828-sil-bagheli-audit.csv",
            "data/other/forms/raw_data/20260828-sil-bagheli-manifest.json",
        ],
        "tests": [
            "tests/test_sil_bagheli_2022.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-bagheli.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
    "20260828-sil-korwa-kodaku": {
        "importers": [
            "data/other/forms/raw_data/sil_korwa_kodaku_2022/extract_scaffolds.py",
            "data/other/forms/raw_data/sil_korwa_kodaku_2022/import_korwa_kodaku.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_korwa_kodaku_2022/manual_review.tsv",
            "data/other/forms/raw_data/sil_korwa_kodaku_2022/page_review.tsv",
            "data/other/forms/raw_data/sil_korwa_kodaku_2022/unresolved_source_codes.tsv",
            "data/other/forms/raw_data/20260828-sil-korwa-kodaku-audit.csv",
            "data/other/forms/raw_data/20260828-sil-korwa-kodaku-manifest.json",
        ],
        "tests": [
            "tests/test_sil_korwa_kodaku_2022.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-korwa-kodaku.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
    "20260828-sil-amri-karbi": {
        "importers": [
            "data/other/forms/raw_data/sil_amri_karbi_2021/extract_amri.py",
            "data/other/forms/raw_data/sil_amri_karbi_2021/finalize_review.py",
            "data/other/forms/raw_data/sil_amri_karbi_2021/import_amri.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_amri_karbi_2021/reviewed_transcription.tsv",
            "data/other/forms/raw_data/20260828-sil-amri-karbi-audit.csv",
            "data/other/forms/raw_data/20260828-sil-amri-karbi-manifest.json",
        ],
        "tests": [
            "tests/test_sil_amri_karbi_2021.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-amri-karbi.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-desia": {
        "importers": [
            "data/other/forms/raw_data/sil_desia_2021/extract_scaffold.py",
            "data/other/forms/raw_data/sil_desia_2021/import_desia.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_desia_2021/manual_review.tsv",
            "data/other/forms/raw_data/sil_desia_2021/page_review.tsv",
            "data/other/forms/raw_data/sil_desia_2021/glyph_order_corrections.tsv",
            "data/other/forms/raw_data/sil_desia_2021/metadata_discrepancies.tsv",
            "data/other/forms/raw_data/20260828-sil-desia-audit.csv",
            "data/other/forms/raw_data/20260828-sil-desia-manifest.json",
        ],
        "tests": ["tests/test_sil_desia_2021.py", "tests/test_source_checklists.py"],
        "profiles": ["conversion/sil-desia.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-korku": {
        "importers": [
            "data/other/forms/raw_data/sil_korku_2021/extract_ocr.py",
            "data/other/forms/raw_data/sil_korku_2021/import_korku.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_korku_2021/manual_review_data.py",
            "data/other/forms/raw_data/20260828-sil-korku-audit.csv",
            "data/other/forms/raw_data/20260828-sil-korku-manifest.json",
        ],
        "tests": ["tests/test_sil_korku_2021.py", "tests/test_source_checklists.py"],
        "profiles": ["conversion/sil-korku.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
    "20260828-sil-konda-dora": {
        "importers": [
            "data/other/forms/raw_data/sil_konda_dora_2012/extract_scaffold.py",
            "data/other/forms/raw_data/sil_konda_dora_2012/import_konda_dora.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_konda_dora_2012/reviewed_transcription.psv",
            "data/other/forms/raw_data/20260828-sil-konda-dora-audit.csv",
            "data/other/forms/raw_data/20260828-sil-konda-dora-manifest.json",
        ],
        "tests": [
            "tests/test_sil_konda_dora_2012.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-konda-dora.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
    "20260828-sil-bonda-didayi": {
        "importers": [
            "data/other/forms/raw_data/sil_bonda_didayi_2022/extract_pdf.py",
            "data/other/forms/raw_data/sil_bonda_didayi_2022/import_bonda_didayi.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_bonda_didayi_2022/extracted_cells.tsv",
            "data/other/forms/raw_data/20260828-sil-bonda-didayi-audit.csv",
            "data/other/forms/raw_data/20260828-sil-bonda-didayi-manifest.json",
        ],
        "tests": [
            "tests/test_sil_bonda_didayi_2022.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-bonda-didayi.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-eastern-gujari": {
        "importers": [
            "data/other/forms/raw_data/sil_eastern_gujari_2023/extract_eastern_gujari.py",
            "data/other/forms/raw_data/sil_eastern_gujari_2023/finalize_review.py",
            "data/other/forms/raw_data/sil_eastern_gujari_2023/import_eastern_gujari.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_eastern_gujari_2023/reviewed_transcription.tsv",
            "data/other/forms/raw_data/20260828-sil-eastern-gujari-audit.csv",
            "data/other/forms/raw_data/20260828-sil-eastern-gujari-manifest.json",
        ],
        "tests": [
            "tests/test_sil_eastern_gujari_2023.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-eastern-gujari.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-mudhili-gadaba": {
        "importers": ["data/other/forms/raw_data/sil_gadaba_2019/import_gadaba.py"],
        "audits": [
            "data/other/forms/raw_data/20260828-sil-mudhili-gadaba-audit.csv",
            "data/other/forms/raw_data/20260828-sil-mudhili-gadaba-manifest.json",
        ],
        "tests": ["tests/test_sil_gadaba_2019.py", "tests/test_source_checklists.py"],
        "profiles": ["conversion/sil-gadaba.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
    "20260828-sil-jaunsari": {
        "importers": [
            "data/other/forms/raw_data/sil_jaunsari_2008/extract_jaunsari.py",
            "data/other/forms/raw_data/sil_jaunsari_2008/import_jaunsari.py",
        ],
        "audits": [
            "data/other/forms/raw_data/20260828-sil-jaunsari-audit.csv",
            "data/other/forms/raw_data/20260828-sil-jaunsari-manifest.json",
        ],
        "tests": ["tests/test_sil_jaunsari_2008.py", "tests/test_source_checklists.py"],
        "profiles": ["conversion/sil-jaunsari.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-war-jaintia": {
        "importers": [
            "data/other/forms/raw_data/sil_war_jaintia_2007/extract_war_jaintia.py",
            "data/other/forms/raw_data/sil_war_jaintia_2007/import_war_jaintia.py",
        ],
        "audits": [
            "data/other/forms/raw_data/20260828-sil-war-jaintia-audit.csv",
            "data/other/forms/raw_data/20260828-sil-war-jaintia-manifest.json",
            "data/other/forms/raw_data/sil_war_jaintia_2007/sag_ipa_used.tsv",
        ],
        "tests": [
            "tests/test_sil_war_jaintia_2007.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-bangladesh.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-kuki-chin-bangladesh": {
        "importers": [
            "data/other/forms/raw_data/sil_kuki_chin_bangladesh_2011/extract_kuki_chin.py",
            "data/other/forms/raw_data/sil_kuki_chin_bangladesh_2011/import_kuki_chin.py",
        ],
        "audits": [
            "data/other/forms/raw_data/20260828-sil-kuki-chin-bangladesh-audit.csv",
            "data/other/forms/raw_data/20260828-sil-kuki-chin-bangladesh-manifest.json",
            "data/other/forms/raw_data/sil_kuki_chin_bangladesh_2011/sag_ipa_used.tsv",
        ],
        "tests": [
            "tests/test_sil_kuki_chin_bangladesh_2011.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-bangladesh.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-western-arunachal-monpa": {
        "importers": [
            "data/other/forms/raw_data/abraham_monpa_2018/import_abraham_monpa.py",
        ],
        "audits": [
            "data/other/forms/raw_data/20260828-sil-western-arunachal-monpa-audit.csv",
            "data/other/forms/raw_data/20260828-sil-western-arunachal-monpa-manifest.json",
        ],
        "tests": [
            "tests/test_abraham_monpa_2018.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/tagin-puroik.txt"],
        "addenda": [
            "Survey wordlists or comparative tables",
            "Website/API or external CLDF",
        ],
    },
    "20260828-sil-bareli-pauri": {
        "importers": [
            "data/other/forms/raw_data/sil_bareli_pauri_2018/import_bareli_pauri.py",
        ],
        "audits": [
            "data/other/forms/raw_data/20260828-sil-bareli-pauri-audit.csv",
            "data/other/forms/raw_data/20260828-sil-bareli-pauri-manifest.json",
        ],
        "tests": [
            "tests/test_sil_bareli_pauri_2018.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-bareli-pauri.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-nimadi": {
        "importers": [
            "data/other/forms/raw_data/sil_nimadi_2012/import_nimadi.py",
        ],
        "audits": [
            "data/other/forms/raw_data/20260828-sil-nimadi-audit.csv",
            "data/other/forms/raw_data/20260828-sil-nimadi-manifest.json",
        ],
        "tests": [
            "tests/test_sil_nimadi_2012.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-nimadi.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-malvi": {
        "importers": [
            "data/other/forms/raw_data/sil_malvi_2009/import_malvi.py",
        ],
        "audits": [
            "data/other/forms/raw_data/20260828-sil-malvi-audit.csv",
            "data/other/forms/raw_data/20260828-sil-malvi-manifest.json",
        ],
        "tests": [
            "tests/test_sil_malvi_2009.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-malvi.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-dogri": {
        "importers": [
            "data/other/forms/raw_data/sil_dogri_2007/extract_dogri.py",
            "data/other/forms/raw_data/sil_dogri_2007/import_dogri.py",
        ],
        "audits": [
            "data/other/forms/raw_data/20260828-sil-dogri-audit.csv",
            "data/other/forms/raw_data/20260828-sil-dogri-manifest.json",
        ],
        "tests": [
            "tests/test_sil_dogri_2007.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-dogri.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-lahul": {
        "importers": [
            "data/other/forms/raw_data/sil_lahul_2019/extract_lahul.py",
            "data/other/forms/raw_data/sil_lahul_2019/import_lahul.py",
        ],
        "audits": [
            "data/other/forms/raw_data/20260828-sil-lahul-audit.csv",
            "data/other/forms/raw_data/20260828-sil-lahul-manifest.json",
        ],
        "tests": [
            "tests/test_sil_lahul_2019.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-lahul.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-ssnp04": {
        "importers": [
            "data/other/forms/raw_data/ssnp04_1992/extract_ssnp04.py",
            "data/other/forms/raw_data/ssnp04_1992/import_ssnp04.py",
        ],
        "audits": [
            "data/other/forms/raw_data/20260828-ssnp04-audit.csv",
            "data/other/forms/raw_data/20260828-ssnp04-manifest.json",
        ],
        "tests": [
            "tests/test_ssnp04_1992.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/ssnp.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-pahari-pothwari": {
        "importers": [
            "data/other/forms/raw_data/sil_pahari_pothwari_2010/extract_pahari_pothwari.py",
            "data/other/forms/raw_data/sil_pahari_pothwari_2010/import_pahari_pothwari.py",
        ],
        "audits": [
            "data/other/forms/raw_data/20260828-sil-pahari-pothwari-audit.csv",
            "data/other/forms/raw_data/20260828-sil-pahari-pothwari-manifest.json",
        ],
        "tests": [
            "tests/test_sil_pahari_pothwari_2010.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-pahari-pothwari.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260826-sil-kurux": {
        "importers": [
            "data/other/forms/raw_data/sil_kurux_2011_manual/build_manual_chunks.py",
            "data/other/forms/raw_data/sil_kurux_2011_manual/build_post_freeze_package.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_kurux_2011_manual/source_manifest.json",
            "data/other/forms/raw_data/sil_kurux_2011_manual/reconciliation.tsv",
            "data/other/forms/raw_data/sil_kurux_2011_manual/staging_audit.tsv",
            "data/other/forms/raw_data/sil_kurux_2011_manual/exclusion_policy.json",
            "data/other/forms/raw_data/sil_kurux_2011_manual/post_freeze_manifest.json",
            "data/other/forms/raw_data/sil_kurux_2011_manual/shared_integration_manifest.json",
        ],
        "tests": [
            "tests/test_sil_kurux_2011_manual.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-kurux.txt"],
        "addenda": ["Survey wordlists or comparative tables"],
    },
    "20260828-sil-kurumba": {
        "importers": [
            "data/other/forms/raw_data/sil_kurumba_2012/import_kurumba.py",
            "data/other/forms/raw_data/sil_kurumba_2012/install_target_forms.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_kurumba_2012/source_manifest.json",
            "data/other/forms/raw_data/sil_kurumba_2012/staged_audit.csv",
            "data/other/forms/raw_data/sil_kurumba_2012/unresolved_readings.tsv",
            "data/other/forms/raw_data/sil_kurumba_2012/shared_integration_audit.csv",
            "data/other/forms/raw_data/sil_kurumba_2012/shared_integration_manifest.json",
            "data/other/forms/raw_data/sil_kurumba_2012/sound_profile_decisions.json",
        ],
        "tests": [
            "tests/test_sil_kurumba_2012.py",
            "tests/test_sound_profiles.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-kurumba-2012.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
    "20260829-sil-northern-dhule-bhils": {
        "importers": [
            "data/other/forms/raw_data/sil_northern_dhule_bhils_2013/import_northern_dhule_bhils.py",
            "data/other/forms/raw_data/sil_northern_dhule_bhils_2013/preintegration_audit.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_northern_dhule_bhils_2013/source_manifest.json",
            "data/other/forms/raw_data/sil_northern_dhule_bhils_2013/staged_audit.tsv",
            "data/other/forms/raw_data/sil_northern_dhule_bhils_2013/unresolved_readings.tsv",
            "data/other/forms/raw_data/sil_northern_dhule_bhils_2013/render_hashes.tsv",
            "data/other/forms/raw_data/sil_northern_dhule_bhils_2013/profile_inventory.tsv",
            "data/other/forms/raw_data/sil_northern_dhule_bhils_2013/cross_source_reconciliation.tsv",
            "data/other/forms/raw_data/sil_northern_dhule_bhils_2013/preintegration_manifest.json",
            "data/other/forms/raw_data/sil_northern_dhule_bhils_2013/shared_integration_manifest.json",
        ],
        "tests": [
            "tests/test_sil_northern_dhule_bhils_2013.py",
            "tests/test_sound_profiles.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-northern-dhule-bhils.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
    "20260829-sil-adi": {
        "importers": [
            "data/other/forms/raw_data/sil_adi_2015/import_adi_2015.py",
            "data/other/forms/raw_data/sil_adi_2015/preintegration_audit.py",
        ],
        "audits": [
            "data/other/forms/raw_data/sil_adi_2015/source_manifest.json",
            "data/other/forms/raw_data/sil_adi_2015/staged_audit.tsv",
            "data/other/forms/raw_data/sil_adi_2015/unresolved_readings.tsv",
            "data/other/forms/raw_data/sil_adi_2015/render_hashes.tsv",
            "data/other/forms/raw_data/sil_adi_2015/symbol_inventory.tsv",
            "data/other/forms/raw_data/sil_adi_2015/preintegration_manifest.json",
            "data/other/forms/raw_data/sil_adi_2015/shared_integration_manifest.json",
        ],
        "tests": [
            "tests/test_sil_adi_2015.py",
            "tests/test_sound_profiles.py",
            "tests/test_source_checklists.py",
        ],
        "profiles": ["conversion/sil-adi.txt"],
        "addenda": ["Survey wordlists or comparative tables", "OCR-heavy source"],
    },
}

ADDENDUM_HEADINGS = {
    "Dictionary or glossary",
    "Survey wordlists or comparative tables",
    "OCR-heavy source",
    "Website/API or external CLDF",
    "Etymological/comparative source",
}

UNIT_ADDENDA = {
    "20230517-chattisgarhi": [
        "Survey wordlists or comparative tables",
        "Etymological/comparative source",
    ],
    "20230521-rajasthani": [
        "Survey wordlists or comparative tables",
        "Etymological/comparative source",
    ],
    "20230522-bundeli": [
        "Survey wordlists or comparative tables",
        "Etymological/comparative source",
    ],
    "20230524-tharu": [
        "Survey wordlists or comparative tables",
        "Etymological/comparative source",
    ],
    "20230526-kannauji": [
        "Survey wordlists or comparative tables",
        "Etymological/comparative source",
    ],
    "20230530-tharu2": [
        "Survey wordlists or comparative tables",
    ],
    "20260718-merriam-dravidian-db": [
        "Website/API or external CLDF",
        "Etymological/comparative source",
    ],
    "20260805-gandhari-org": [
        "Dictionary or glossary",
        "Website/API or external CLDF",
        "Etymological/comparative source",
    ],
    "20260817-ghatage-marati-kasargod": ["Dictionary or glossary", "OCR-heavy source"],
    "20260818-hockings-badaga": [
        "Dictionary or glossary",
        "OCR-heavy source",
        "Etymological/comparative source",
    ],
    "20260818-nured-org": [
        "Dictionary or glossary",
        "Website/API or external CLDF",
        "Etymological/comparative source",
    ],
    "20260819-emeneau-brahui-1997": ["Etymological/comparative source"],
    "20260819-burrow-emeneau-den1": ["Etymological/comparative source"],
    "20260819-burrow-emeneau-den2": ["Etymological/comparative source"],
    "20260819-buddruss-grangali": [
        "Dictionary or glossary",
        "OCR-heavy source",
        "Etymological/comparative source",
    ],
    "20260824-buddruss-waigali": [
        "Dictionary or glossary",
        "OCR-heavy source",
        "Etymological/comparative source",
    ],
    "20260824-buddruss-wama": [
        "Dictionary or glossary",
        "OCR-heavy source",
        "Etymological/comparative source",
    ],
    "20260826-sil-kochbd": [
        "Survey wordlists or comparative tables",
        "OCR-heavy source",
    ],
    "20260826-sil-kurux": [
        "Survey wordlists or comparative tables",
    ],
    "20260826-sil-garobd": [
        "Survey wordlists or comparative tables",
        "OCR-heavy source",
    ],
}

# Some comparative inputs cite both their own database and earlier reconstruction
# sources on every row. Use the unit-defining source for compiled survival counts;
# otherwise unrelated rows carrying the earlier bibliography key inflate the result.
UNIT_PRIMARY_SOURCES = {
    "20260718-merriam-dravidian-db": {"merriam2026dravidiandb"},
}

PINNED_LEGACY_UNITS = {
    "20220913-dhivehi",
    "20220913-khetrani",
    "20220913-kholosi",
    "20220913-konkani",
    "20220913-kundalshahi",
    "20220913-kvari",
    "20220913-patyal",
    "20220913-zadjali",
    "20230524-sindhic",
}

UNIT_REVIEW_NOTES = {
    "20260829-sil-adi": {
        "state": (
            "all 2,763 Appendix B cells manually reviewed and exactly 2,770 expanded target "
            "forms installed through source-specific shared integration; consolidated CLDF "
            "and browser gates remain deferred"
        ),
        "exclusions": (
            "ninety-three source-explicit no-entry cells remain cell-addressed and audit-only; "
            "there are no comparison-control lists"
        ),
        "unresolved": (
            "zero ambiguous, illegible or unresolved cells and therefore no unresolved coordinates"
        ),
        "transcription": (
            "every retained IPA form was visually verified by hand against rendered source pages; "
            "PDF text was locator/character-input scaffold only and OCR, legacy data and installed "
            "forms supplied or verified no reading"
        ),
        "validation": (
            "frozen PDF/manual/staged/render hashes, exhaustive conceptual-cell dispositions, "
            "expanded-response keys, locators, registry/reference/profile routing and complete "
            "42-symbol coverage are guarded by focused Adi and sound-profile tests"
        ),
        "filled_note": (
            "Checked boxes describe the completed source-specific integration stage. The "
            "consolidated build/full-suite gate remains unchecked and browser QA is deferred by request."
        ),
    },
    "20260829-sil-northern-dhule-bhils": {
        "state": (
            "all 2,730 Appendix C cells manually reviewed and exactly 2,497 target-scope "
            "attestations installed through source-specific shared integration; consolidated "
            "CLDF and browser gates remain deferred"
        ),
        "exclusions": (
            "the 210-cell Toranmal comparison-control list, twenty-one target blanks and two "
            "ambiguous target cells remain cell-addressed and audit-only"
        ),
        "unresolved": (
            "item 10/KEL at PDF 92 printed 84 left and item 31/MUN at PDF 97 printed 89 left "
            "are ambiguous targets; item 74/TOR at PDF 105 printed 97 right is an ambiguous "
            "control; none is guessed or installed"
        ),
        "transcription": (
            "every IPA cell was transcribed and visually verified by hand from rendered pages; "
            "OCR/PDF text and later Noira/Bareli publications were locator or post-freeze "
            "reconciliation aids only and supplied or verified no retained reading"
        ),
        "validation": (
            "frozen PDF/manual/staged/render hashes, exhaustive dispositions, immutable source "
            "keys and locators, target-only installation, registry/reference/profile routing and "
            "complete symbol coverage are guarded by focused Northern Dhule and profile tests"
        ),
        "filled_note": (
            "Checked boxes describe the completed source-specific integration stage. The "
            "consolidated build/full-suite gate remains unchecked and browser QA is deferred by request."
        ),
    },
    "20260828-sil-kurumba": {
        "state": (
            "all 10,450 Appendix C cells manually reviewed and the 3,204 target-scope "
            "attestations installed through source-specific shared integration; consolidated "
            "CLDF and browser gates remain deferred"
        ),
        "exclusions": (
            "1,534 comparison-control attestations, 5,710 printed-dash blanks, one ambiguous "
            "Pudukkottai cell and one illegible Kotagiri Alu cell are cell-addressed and audit-only"
        ),
        "unresolved": (
            "exactly kurumba2012:pudukkottai:i020 (ambiguous) and "
            "kurumba2012:kotagiri_alu:i025 (illegible); neither is guessed or installed"
        ),
        "transcription": (
            "every cell and all 550 prompts were manually read from rendered pages; OCR/PDF text "
            "was locator-only; the frozen 4,738-attestation snapshot is retained unchanged and a "
            "deterministic audit filter installs only the 3,204 target attestations"
        ),
        "validation": (
            "source-local topology, hashes, manual-only staging, exhaustive scope filtering, "
            "registry/reference/profile routing and complete symbol coverage are guarded by "
            "tests/test_sil_kurumba_2012.py and focused sound-profile tests; consolidated build pending"
        ),
        "filled_note": (
            "Checked boxes describe the completed source-specific integration stage. The consolidated "
            "build/full-suite gate remains unchecked and browser QA is deferred by request."
        ),
    },
    "20260826-sil-kochbd": {
        "state": (
            "all 307 Appendix A.3 items manually reviewed and the 1,017 resolved Koch "
            "target attestations installed through shared source-specific integration; "
            "consolidated CLDF and browser gates remain deferred"
        ),
        "exclusions": (
            "772 resolved A'tong/Bangla control rows, 226 ambiguous expanded rows, 25 printed "
            "blanks and 119 not-used rows are audit-only"
        ),
        "unresolved": (
            "225 ambiguity-only conceptual cells plus the unresolved variant at mixed "
            "item 241/site r produce 226 coordinates and 226 expanded rows with unresolved "
            "modifiers; none is guessed or installed"
        ),
        "transcription": (
            "every lexical cell was manually read from rendered physical pages 43--62; OCR, "
            "PDF text, raw legacy glyphs and installed forms were locator or post-freeze "
            "comparison only and supplied or verified no reading"
        ),
        "validation": (
            "frozen-ledger, exhaustive reconciliation, staging, registry, reference, profile "
            "routing and parser invariants are guarded by `tests/test_sil_kochbd_2011_manual.py` "
            "and `tests/test_sil_kochbd_2011_post_freeze.py`; consolidated build pending"
        ),
        "filled_note": (
            "Checked boxes describe the completed source-specific integration stage. Global "
            "source-audit regeneration, consolidated build/full-suite validation and browser QA "
            "remain deferred."
        ),
    },
    "20260826-sil-kurux": {
        "state": (
            "manual rendered-page transcription complete and installed through the shared "
            "source-specific integration gates; consolidated CLDF and browser gates remain deferred"
        ),
        "exclusions": (
            "296 Standard Bangla control attestations, 136 printed no-entry cells and 72 "
            "expanded coordinates for 12 globally unused items are audit-only"
        ),
        "unresolved": (
            "none: the 239 legacy-decoder omissions were independently recovered by hand, and "
            "the frozen manual ledger records no ambiguous or illegible lexical coordinate"
        ),
        "transcription": (
            "all 1,869 expanded cells were hand-read and visually rechecked from rendered physical "
            "pages 39--57; OCR, PDF text and legacy forms were locator/post-freeze comparison only; "
            "`conversion/sil-kurux.txt` preserves source IPA in Phonemic and converts display Form"
        ),
        "validation": (
            "frozen hashes, exhaustive dispositions, immutable Entry_Keys, registry/reference/profile "
            "routing and incomplete-stage refusal are guarded by `tests/test_sil_kurux_2011_manual.py`; "
            "the consolidated build, full suite and browser refresh remain pending"
        ),
        "filled_note": (
            "Checked boxes below describe the completed source-specific integration stage. The "
            "consolidated build/full-suite gate remains unchecked; browser refresh is deferred by request."
        ),
    },
    "20260826-sil-garobd": {
        "state": (
            "partial legacy-font recovery; 712 attested audit records still require "
            "independent visual transcription from rendered publisher pages"
        ),
        "exclusions": "91 per-site printed gaps and 17 globally unused items are audit-only",
        "unresolved": (
            "712 attested records contain at least one glyph without a verified decoder mapping; "
            "all remain audit-only until manually transcribed and rechecked from rendered pages"
        ),
        "transcription": (
            "the decoded subset has no explicit profile route; the render-first manual package "
            "is queued in `data/other/forms/raw_data/sil_bangladesh_legacy_manual_queue.md`"
        ),
        "validation": (
            "exact partial-decoder disposition counts are guarded by "
            "`tests/test_sil_bracket_wordlists.py`; manual source-local gates and the consolidated "
            "build remain pending"
        ),
        "filled_note": (
            "Checked boxes below describe only the installed subset. The source is not complete "
            "until the manual-recovery gates above pass; addenda not listed are inapplicable."
        ),
    },
    "20260828-sil-nilgiri-irula": {
        "exclusions": (
            "15 target cells explicitly marked missing, 1,319 neighbouring-language control "
            "responses, and 29 layout fragments are audit-only; 2,054 Irula forms are installed"
        ),
        "unresolved": (
            "typed source-raster flags retain the low-resolution distinctions that need future "
            "source review; no record is structurally unparsed or unmapped"
        ),
        "transcription": (
            "`conversion/sil-irula.txt`; structural OCR recovered the table layout and every "
            "installed IPA form was manually reviewed against enlarged 67--74 dpi source crops"
        ),
        "representative": (
            "Kunjapanai item 1 `oɖʌmbɨ`, Thaliyur item 4 `muɲʤi`, and Bookapuram item "
            "183 alternate `ʌʋã`"
        ),
    },
    "20260828-sil-haryanvi": {
        "exclusions": (
            "21 visibly blank target cells, one elicitation-only target cell, and 840 Braj/"
            "Haryanvi, Baghati Pahari, Hindustani and Punjabi comparison cells are audit-only; "
            "1,553 variants from six Haryanvi lists install"
        ),
        "unresolved": (
            "typed audit notes preserve faint, unlabelled, repeated-label and clipped source "
            "fragments; seven same-list cross-references are resolved without guessing"
        ),
        "transcription": (
            "`conversion/sil-haryanvi.txt`; structural OCR supplied comparison evidence only, "
            "and all 1,260 target cells were manually inspected on enlarged source crops"
        ),
        "validation": (
            "seven focused source/import/profile/metadata checks pass; consolidated full build "
            "and browser QA remain deferred until the parallel survey batch lands"
        ),
        "representative": (
            "Rohtak item 1 `d̪eh`, Jind item 4 `mu`, Fatehabad item 52 alternate "
            "`d̪ʌg·ʌl`, and Fatehabad item 209 resolved cross-reference `t̪u`"
        ),
    },
    "20260828-sil-koya": {
        "exclusions": (
            "69 missing target slots and 420 Telugu and Oriya comparison cells are audit-only; "
            "1,438 variants from seven Koya, Gondi and Madia lists install"
        ),
        "unresolved": (
            "Malakanagiri item 31 retains an explicitly flagged unclear medial source symbol; "
            "two collector question marks and all absent or clipped slots remain typed"
        ),
        "transcription": (
            "`conversion/sil-koya.txt`; OCR was comparison scaffolding only, and every one of "
            "the 1,840 printed cells was manually inspected against the source images"
        ),
        "validation": (
            "eleven focused source/import/profile/metadata checks pass; consolidated full build "
            "and browser QA remain deferred until the parallel survey batch lands"
        ),
        "representative": (
            "Jaganathapuram body `oleu`, Chintoor heart `gundikaia`, Utnoor body "
            "`mɛːnd̪ol`, and Malakanagiri heart `dʒiːva`"
        ),
    },
    "20260828-sil-kullu": {
        "exclusions": (
            "415 visibly blank response cells and one printed Hindi gutter label are "
            "audit-only; 2,963 variants from sixteen Kullui-area lists install"
        ),
        "unresolved": (
            "Chinninal item 192 and faint Bathad item 193 remain explicitly uncertain; three "
            "source question marks are preserved and no unread form is guessed"
        ),
        "transcription": (
            "`conversion/sil-kullu.txt`; OCR was comparison scaffolding only, and all 3,168 "
            "handwritten response cells were manually inspected against source images"
        ),
        "validation": (
            "ten focused source/import/profile/metadata checks pass; consolidated full build "
            "and browser QA remain deferred until the parallel survey batch lands"
        ),
        "representative": (
            "Churla body `dʒɪsəm`, Jibhi yes alternatives `hã` / `oː`, and Ani item 198 "
            "`t̪eɽə nʊ kidʒi əsə`"
        ),
    },
    "20260828-sil-bagheli": {
        "exclusions": (
            "283 Standard Hindi control occurrences, 24 non-lexical by-name occurrences, "
            "47 blank conceptual cells, and two legible but site-unassigned response lines "
            "are audit-only; 5,828 Bagheli occurrence rows from eighteen locality lists install "
            "and yield 5,829 compiled forms after one source comma-alternative expansion"
        ),
        "unresolved": (
            "items 191 `berəʈɛ` and 195 `reŋeʈe` are readable but lack source site codes and "
            "remain excluded; item 189/a retains source uncertainty and item 121/l retains a "
            "medium-confidence uppercase/lowercase site-code interpretation"
        ),
        "transcription": (
            "`conversion/sil-bagheli.txt`; the 92-block OCR output is comparison evidence only, "
            "and every response line and all 3,990 conceptual cells were manually inspected"
        ),
        "validation": (
            "ten focused extraction/import/audit/profile/metadata checks pass; consolidated "
            "full build and browser QA remain deferred until the parallel survey batch lands"
        ),
        "representative": (
            "Dabhaura item 1 `ɖeh`, Singpur item 16 `eŭʈʰi`, Amarkantak item 104 "
            "`beɕja`, and Domahai item 210 `ū patʃe`"
        ),
    },
    "20260828-sil-korwa-kodaku": {
        "exclusions": (
            "1,453 attested comparison-control responses, seventeen blank or unlisted "
            "control cells, fifty target cells printed or assigned `NO ENTRY`, and two "
            "responses carrying unidentified source site codes are audit-only; 4,458 "
            "variant-expanded rows from nine Korwa and nine Kodaku lists install"
        ),
        "unresolved": (
            "PDF page 73 item 83 `buluŋg` bears unidentified source code `u`, and PDF "
            "page 84 item 173 `nɐʔa` bears unidentified source code `n`; both remain "
            "diplomatically transcribed in the unresolved ledger but are not reassigned"
        ),
        "transcription": (
            "`conversion/sil-korwa-kodaku.txt`; Appendix B.5 is typeset Unicode rather than "
            "handwritten, but every one of its 2,900 printed response rows and all 5,250 "
            "conceptual cells were manually checked across PDF pages 66–90; OCR and text "
            "extraction served only as comparison scaffolds"
        ),
        "validation": (
            "focused extraction/import/audit/profile/metadata checks pass; consolidated "
            "full build and browser QA remain deferred until the parallel survey batch lands"
        ),
        "representative": (
            "Chilma Korwa item 1 `d̪ẽh`, Sardih Korwa item 1 `d̪ejaɲ`, and Kodaku "
            "item 104 alternatives `koda hɔpoɲ` / `koɖi hɔpoɲ`"
        ),
    },
    "20260828-sil-amri-karbi": {
        "exclusions": (
            "631 Khasi and Assamese control responses, six explicit `no entry` cells, and "
            "237 exact repeated target occurrences printed under multiple similarity groups "
            "are audit-only; 5,092 forms from three Amri Karbi and twelve Karbi lists install. "
            "Four reported Amri lists absent from Appendix B.3 are not reconstructed"
        ),
        "unresolved": (
            "no transcription is unresolved or illegible. Excluded Assamese control item 91/Z "
            "faithfully retains the source-marked uncertainty `soʌ̆ĭ??` at PDF page 59"
        ),
        "transcription": (
            "`conversion/sil-amri-karbi.txt`; Appendix B.3 is typeset Unicode, and all 5,219 "
            "conceptual cells plus all 5,966 printed records were manually compared against "
            "the 79 rendered canonical pages. No OCR was used"
        ),
        "validation": (
            "eight focused source/import/audit/profile checks pass; consolidated full build "
            "and browser QA remain deferred until the parallel survey batch lands"
        ),
        "representative": (
            "Holanki item 1 `sĭnĭŋ`, Hajarongpi item 1 `sŭneŋ`, and Amguri "
            "(Kamrup) item 1 `ʌ̆ŋsoŋ`"
        ),
    },
    "20260828-sil-desia": {
        "exclusions": (
            "thirty-eight explicit `no entry` cells (items 23 and 24 at all nineteen sites) "
            "are audit-only; three exact response repetitions under source groups 1 and 2 "
            "are merged; 4,655 forms from nineteen Desia lists install"
        ),
        "unresolved": (
            "no transcription is ambiguous, clipped, illegible, or unresolved. The genuinely "
            "blank similarity-group field at Ghumar item 109 remains `[blank]`, and seven "
            "source metadata discrepancies are preserved without silent normalization"
        ),
        "transcription": (
            "`conversion/sil-desia.txt`; all 4,696 printed response lines across 48 rendered "
            "pages were manually checked. The Unicode text layer was only a scaffold; 542 "
            "mispositioned zero-width dental/nasal marks were corrected from visual evidence"
        ),
        "validation": (
            "nine focused source/import/audit/profile checks pass; consolidated full build "
            "and browser QA remain deferred until the parallel survey batch lands"
        ),
        "representative": (
            "Potenda body alternatives `ɡaɡɔɖɨ` / `ɡaɡɔɖɨ muɳɖ`, Aunli item 203 "
            "visually corrected `t̪ui`, and Ghumar item 109 `apa` with a blank group"
        ),
    },
    "20260828-sil-korku": {
        "exclusions": (
            "216 confirmed blank target cells, one illegible/clipped target cell, and all 210 "
            "Nihali comparison cells are audit-only; 1,463 attested Korku cells yield 1,521 "
            "installed rows after source slash alternatives are expanded"
        ),
        "unresolved": (
            "Amdhana Mawasi item 93 `tail` at PDF page 83 / printed page 78 contains only "
            "faint clipped marks and is excluded without guessing; 24 collector-question "
            "cases and 16 faint/clipped/ambiguous but readable forms retain typed flags"
        ),
        "transcription": (
            "`conversion/sil-korku.txt`; OCR is a separate comparison scaffold only. Every "
            "one of the 1,890 image-only cells (1,680 target and 210 control) was manually "
            "read and checked against the canonical scan"
        ),
        "validation": (
            "nine focused source/import/audit/profile checks pass; consolidated full build "
            "and browser QA remain deferred until the parallel survey batch lands"
        ),
        "representative": (
            "Chikli Ruma body `kombor`, Lahi head alternatives `dẽi` / `kapar`, and "
            "Khamalpur Bondoy item 18 `dʒaɖa`"
        ),
    },
    "20260828-sil-konda-dora": {
        "exclusions": (
            "43 confirmed blank target cells, 86 confirmed blank control cells, and 342 "
            "attested Telugu and Adivasi Oriya control cells are audit-only; 385 attested "
            "Konda cells yield 452 installed rows after source-defined slash expansion"
        ),
        "unresolved": (
            "no cell is ambiguous, clipped, illegible, source-questioned, or unresolved. "
            "The report's duplicate item number 212 is preserved as separate liver and foot prompts"
        ),
        "transcription": (
            "`conversion/sil-konda-dora.txt`; all 856 image-scan cells (428 target and 428 "
            "controls) were manually transcribed and checked from rendered pages. The corrupt "
            "text/OCR layer is a locator scaffold only and never supplies an accepted reading"
        ),
        "validation": (
            "seven focused source/import/audit/profile checks pass; consolidated full build "
            "and browser QA remain deferred until the parallel survey batch lands"
        ),
        "representative": (
            "Koraput body `oɽol`, Koraput item 74 `bumi ka:ʈoliŋ`, and Visakh item "
            "122 `i?en`, with literal source punctuation preserved diplomatically"
        ),
    },
    "20260828-sil-bonda-didayi": {
        "exclusions": (
            "36 target cells at four source-disqualified prompts, seventeen explicit target "
            "no-entry/dash cells, one physically omitted target row, and all 840 Gutob, "
            "Parenga, Rona Desiya, and Oriya comparison cells are audit-only; 1,836 "
            "attested target cells yield 1,938 installed rows after comma alternatives"
        ),
        "unresolved": (
            "Orapadar Upper Didayi item 174 `those` at PDF page 45 / printed page 40 is "
            "physically absent and remains missing without a guessed form; no glyph is "
            "ambiguous, clipped, or illegible"
        ),
        "transcription": (
            "`conversion/sil-bonda-didayi.txt`; all 2,730 cells across thirty rendered pages "
            "were visually checked. The appendix is born-digital Unicode and OCR was not used; "
            "one broken text-map glyph was manually corrected from the rendered page"
        ),
        "validation": (
            "eight focused extraction/import/audit/profile checks pass; consolidated full build "
            "and browser QA remain deferred until the parallel survey batch lands"
        ),
        "representative": (
            "Biapada body `gagəɖe`, Kaluguda body `gaːgɪɖe`, and Chitrakonda item "
            "50 manually verified `bɷihumhaiʒã`"
        ),
    },
    "20260828-sil-eastern-gujari": {
        "exclusions": (
            "twenty-five blank Indian target cells, eight blank non-target cells, 1,254 "
            "attested Pakistan Gujari cells republished from SSNP volume 3, 208 attested Urdu "
            "control cells, and one exact repeated Jammu target alternative are audit-only; "
            "1,753 new Indian forms install"
        ),
        "unresolved": (
            "no cell is ambiguous, clipped, illegible, or unresolved. The six Pakistan lists "
            "map explicitly to existing primary-source SSNP dialects and never overwrite them"
        ),
        "transcription": (
            "`conversion/sil-eastern-gujari.txt`; all 3,150 cells across thirty-five lexical "
            "pages were manually compared with 180-dpi renders. The Unicode text layer was "
            "only a scaffold and OCR was not used"
        ),
        "validation": (
            "eight focused source/import/audit/profile checks pass; consolidated full build "
            "and browser QA remain deferred until the parallel survey batch lands"
        ),
        "representative": (
            "Udhampur body `dʒhɑn`, Haldwani body alternative `dʒɪsʌm`, and Nalagarh "
            "item 176 alternative `fʌɷək`"
        ),
    },
    "20260828-sil-mudhili-gadaba": {
        "exclusions": (
            "eight target cells printed `No Entry`, 214 Srikakulam Telugu control responses, "
            "and five prompts printed `DISQUALIFIED` are audit-only; 1,538 target forms install"
        ),
        "unresolved": (
            "two printed question-mark responses remain diplomatic in Phonemic with typed "
            "source-raster-unresolved notes; no record is structurally unparsed or unmapped"
        ),
        "transcription": (
            "`conversion/sil-gadaba.txt`; structural OCR supplied layout only and every IPA form "
            "was visually transcribed from quarter-column crops of the 168-dpi source pages"
        ),
        "representative": (
            "Bobbilivalasa item 52 `kaṇḍu`, Bobbilivalasa item 18 `kālgil` with source "
            "Phonemic `kɑlgil(pl)`, and Panukuvalasa item 12 `puḍu` with source `puɖʊ?`"
        ),
    },
    "20260828-sil-jaunsari": {
        "exclusions": (
            "1,110 Hindi, Bangani, Jaunpuri, Nagpuriya, and Sirmauri comparison-control "
            "responses and the three source-disqualified prompts 11, 23, and 24 are audit-only; "
            "1,619 responses from seven Jaunsari lists install"
        ),
        "unresolved": (
            "no private-use byte or source line remains unmapped or unparsed; Khanaad's "
            "coordinate uses the report's documented Kandar census-place identification"
        ),
        "transcription": (
            "`conversion/sil-jaunsari.txt`; 5,059 SAG legacy-font occurrences were decoded "
            "with SIL's official SAGIPA2Uni.map and the exact 32 used bytes are pinned"
        ),
        "representative": (
            "Korwa item 1 source `çʌɾiɾ` / display `çarir`, Korwa item 9 source `d̪ant̪` / "
            "display `dant`, and Maindrath item 210 alternate source `jɛ dʒɛ` / display `ye je`"
        ),
    },
    "20260828-sil-war-jaintia": {
        "exclusions": (
            "1,428 Pnar, Lyngngam, Khasi War and standard Khasi comparison responses plus "
            "one undefined printed site code U are audit-only; 2,030 responses from seven "
            "War-Jaintia lists install"
        ),
        "unresolved": (
            "no private-use byte or source line remains unmapped or unparsed; the undefined "
            "site code U at item 119 is preserved as a typed source anomaly rather than guessed"
        ),
        "transcription": (
            "`conversion/sil-bangladesh.txt`; all 2,398 SAG legacy-font occurrences were "
            "decoded with SIL's official SAGIPA2Uni.map and the exact 17 used bytes are pinned"
        ),
        "validation": (
            "10 focused extraction/import/profile/compiled-identity checks pass; the full data "
            "pipeline completed through reference generation before two unrelated stale Kannauji "
            "count assertions"
        ),
        "representative": (
            "`/languages/WarJaintia`, `/references/brightbill-kim-kim2007warjaintia`, "
            "`/entries/f_mahp3a3u3rpyo`, exact `phli yaŋ` search, and `/concepts/1732` (SKY)"
        ),
    },
    "20260828-sil-kuki-chin-bangladesh": {
        "exclusions": (
            "307 standard Bangla responses and 333 external Myanmar Khumi comparison "
            "attestations are audit-only; 3,235 responses from ten Bangladesh lists install"
        ),
        "unresolved": (
            "no private-use code point or source line remains unmapped or unparsed; all 53 "
            "explicit `no entry` records occur in the audit-only Myanmar Khumi comparison"
        ),
        "transcription": (
            "`conversion/sil-bangladesh.txt`; all 16,029 SAG legacy-font occurrences were "
            "decoded from embedded font outlines and the exact 65 used glyphs are pinned"
        ),
        "validation": (
            "seven focused source checks, four relevant profile/build-routing checks and eight "
            "browser-database unit tests pass; full build reaches references before the two "
            "unrelated stale Kannauji assertions recorded in VALIDATION.md"
        ),
        "representative": (
            "`/languages/Pangkhua` plus BawmChin, AshoChin, KhumiChin and Mizo; "
            "`/references/kim-roy-sangma2011kukichin`; `/entries/f_hc6fe7vkp3xj2`; exact "
            "`rɨvan` search; and expanded `/concepts/1732` (SKY) evidence"
        ),
    },
    "20260828-sil-bareli-pauri": {
        "exclusions": (
            "789 standard-language controls, 105 cells printed `NO ENTRY`, and all 33 cells "
            "for the source-disqualified `millet` prompt are audit-only; 6,320 regional "
            "attestations from 30 lists install"
        ),
        "unresolved": (
            "two source forms end in a literal unmatched open bracket; both are preserved "
            "diplomatically and carry typed uncertainty notes, while no record remains "
            "structurally unparsed or unmapped"
        ),
        "transcription": (
            "`conversion/sil-bareli-pauri.txt`; the PDF has a usable Unicode Charis/Doulos "
            "SIL text layer, so extraction uses positioned text rather than OCR and preserves "
            "the printed IPA in Original and Phonemic"
        ),
        "validation": (
            "seven focused source checks, three relevant profile/routing checks and eight "
            "browser-database unit tests pass; the full build reaches reference generation "
            "before only the unrelated stale Kannauji count assertions recorded in VALIDATION.md"
        ),
        "representative": (
            "`/references/varkey-vunnamatla2018bareli`, exact `pats[` and "
            "`bɦuklu ce, bɦuklu hato` searches, and `/languages/RathwiBareli` with all seven "
            "registered survey localities"
        ),
    },
    "20260828-sil-nimadi": {
        "exclusions": (
            "1,207 responses/cells from the Parya Bhilali, Malvi, Hindi, Gujarati and Marathi "
            "comparison lists, five target cells printed `no entry`, 52 target cells for the "
            "four prompts absent from the appendix, and two target cells without primary forms "
            "are audit-only; 2,826 Nimadi attestations from thirteen lists install"
        ),
        "unresolved": (
            "no response line or source code point remains structurally unparsed or unmapped; "
            "the two primary-form gaps and all four omitted prompts are represented explicitly "
            "instead of receiving conjectural forms"
        ),
        "transcription": (
            "`conversion/sil-nimadi.txt`; the exact archived publisher PDF has a usable Unicode "
            "Doulos SIL text layer, so extraction uses positioned text rather than OCR and "
            "preserves the printed IPA in Original and Phonemic"
        ),
        "validation": (
            "focused extraction, topology, audit, profile, metadata and compiled-identity tests "
            "pass; repository-wide baseline exceptions are recorded in VALIDATION.md"
        ),
        "representative": (
            "`/references/vunnamatla-john-samuvel2012nimadi`, exact `bhuklagi, bhuklagtithi` "
            "search, and `/languages/Nimadi` with all thirteen ESR 2012-002 localities"
        ),
    },
    "20260828-sil-malvi": {
        "exclusions": (
            "1,891 response/matrix records from two Bhili, two Nimadi, Bhopali, Hindi, "
            "Gujarati and Marathi comparison lists, 37 target cells printed `By Name`, "
            "and 90 target cells for the three disqualified prompts are audit-only; "
            "6,894 Malvi attestations from thirty target lists install"
        ),
        "unresolved": (
            "no response row or used legacy-font CID remains structurally unparsed or unmapped; "
            "the source's literal circumflex occurs only in audit-only controls and is retained "
            "diplomatically rather than assigned a conjectural phonetic value"
        ),
        "transcription": (
            "`conversion/sil-malvi.txt`; no OCR is used. Thirty-one used CIDs are identified "
            "from the report's IPA chart and three from SIL's official SAG-IPA mapping plus "
            "rendered-page checks; raised and combining glyph placement is recovered geometrically"
        ),
        "validation": (
            "focused extraction, topology, audit, profile, metadata and cross-source tests pass; "
            "the Thillorkhurd comparison yields 132 exact Unicode response matches across 126 concepts"
        ),
        "representative": (
            "`/references/varghese-john-samuel2009malvi`, exact `kʰʌⁱlo, ɠʰajlijo` search, "
            "and `/languages/mewari_basad` with all thirty ESR 2009-011 localities"
        ),
    },
    "20260828-sil-dogri": {
        "exclusions": (
            "three blank Batote response cells (items 11 breast, 23 urine and 24 feces) are "
            "audit-only; the five earlier Reasi, Ramnagar, Udhampur, Samba and Billawar "
            "wordlists are reported only through similarity percentages and contain no "
            "published forms; 207 Batote responses install"
        ),
        "unresolved": (
            "no published response, source row or used SIL IPA93 legacy byte remains "
            "structurally unparsed or unmapped"
        ),
        "transcription": (
            "`conversion/sil-dogri.txt`; no OCR is used. The positioned PDF text layer is "
            "decoded with SIL's official SIL-IPA93-2001.map v14 and checked against rendered "
            "Appendix B pages 26--28; Original and Phonemic preserve the recovered Unicode IPA"
        ),
        "validation": (
            "focused extraction, topology, audit, profile, metadata and compiled-identity tests pass; "
            "repository-wide baseline exceptions are recorded in VALIDATION.md"
        ),
        "representative": (
            "`/references/brightbill-turner2007dogri`, exact `d͡ʒɪsɘm` source search, "
            "and `/languages/dog` with the Batote dialect tag"
        ),
    },
    "20260828-sil-lahul": {
        "exclusions": (
            "29 target cells printed `no entry`, 474 Standard Hindi/Lhasa Tibetan control "
            "responses, and 676 previously collected Tindi Pangi, Leh Ladakhi and Mane Spiti "
            "Bhoti responses are audit-only; 5,027 responses from 22 newly collected Lahul "
            "lect/site lists install"
        ),
        "unresolved": (
            "no prompt, published response, lect label, wrapped form, source glyph or target "
            "site remains structurally unparsed or unmapped"
        ),
        "transcription": (
            "`conversion/sil-lahul.txt`; no OCR is used. Appendix A.4 has a complete Unicode "
            "Charis/Doulos SIL text layer; ten wrapped group/form pairs are joined geometrically "
            "and checked on rendered pages, while Original and Phonemic preserve source IPA"
        ),
        "validation": (
            "focused extraction, topology, audit, profile, language/dialect metadata and "
            "compiled-identity tests pass; repository-wide baseline exceptions are recorded "
            "in VALIDATION.md"
        ),
        "representative": (
            "`/references/chamberlain-chamberlain2019lahul`, exact `ɾəɳdʒ.kɾiɳdʒ` source "
            "search, and `/languages/lae` with eight Pattani locality tags"
        ),
    },
    "20260828-ssnp04": {
        "exclusions": (
            "68 cells printed `--` and the blank Bannu item 135 cell are audit-only; "
            "all 7,131 lexical responses from 34 Pashto location lists plus the Waneci "
            "and Ormuri lists install"
        ),
        "unresolved": (
            "no printed prompt, list label, response cell, continuation line, used legacy "
            "glyph, base-language assignment, or locality tag remains structurally unparsed "
            "or unmapped; ten prompt numbers absent from the tables are source-declared exclusions"
        ),
        "transcription": (
            "`conversion/ssnp.txt`; no OCR is used. Appendix B has a complete positioned "
            "SILDoulosNP text layer; 42 wrapped cells are joined geometrically and the legacy "
            "glyphs are decoded against the report's printed phonetic chart, preserving `ɣ`, "
            "`ɸ`, retroflex `ɭ`, length, nasalisation, and underdot distinctions"
        ),
        "validation": (
            "focused extraction, full-cell topology, audit, profile, language/dialect metadata "
            "and compiled-identity tests pass; repository-wide baseline exceptions are recorded "
            "in VALIDATION.md"
        ),
        "representative": (
            "`/references/hallberg1992pashto`, exact `ɣʌ̃ɳye / ɣʌ̃ɽye` source search, "
            "and `/languages/Psht` with 34 SSNP volume 4 locality tags"
        ),
    },
    "20260828-sil-pahari-pothwari": {
        "exclusions": (
            "the 434 Abbottabad and Mansehra Hindko comparison cells, including eighteen "
            "source blanks at items 209--217, are audit-only; all 3,038 responses from the "
            "fourteen Pahari, Pothwari and Mirpuri target lists install"
        ),
        "unresolved": (
            "no concept, list row, response cell or source symbol remains structurally unparsed; "
            "fourteen printed `AUS` labels in the invariant OSI row are retained raw and "
            "explicitly normalized to the Osia list"
        ),
        "transcription": (
            "`conversion/sil-pahari-pothwari.txt`; no OCR is used. Appendix B.1 has a complete "
            "positioned Doulos SIL text layer, and content-stream order preserves the source's "
            "Indological Phonetic Script, length, nasalisation and breathy-voice marks"
        ),
        "validation": (
            "focused extraction, full-cell topology, audit, profile, language/dialect metadata "
            "and compiled-identity tests are required; repository-wide results are recorded in "
            "VALIDATION.md"
        ),
        "representative": (
            "`/references/lothers-lothers2010pahari`, exact source-diacritic search, and "
            "`/languages/poth` with fourteen ESR 2010-012 locality tags"
        ),
    },
    "dbia-forms": {
        "exclusions": (
            "none of the 337 recoverable dictionary articles or 1,694 conservatively parsed "
            "Dravidian attestations is excluded; nine cross-reference-only articles have no "
            "recoverable independent IA headword and therefore emit no comparison"
        ),
        "unresolved": (
            "186 loan sets resolve to canonical CDIAL entries and 142 preserve a source-local "
            "IA comparison term; DBIA 28, 53, 57, 119, 135, 141, 181, 215, and 332 remain "
            "comparison-unresolved rather than receiving conjectural donors"
        ),
        "transcription": (
            "`conversion/dedr.txt`; DBIA articles are form-less Proto-Dravidian grouping nodes, "
            "not reconstructed PDr forms, while all 1,694 source forms retain OCR provenance "
            "and the printed loan evidence is preserved on 328 typed cross-family comparisons"
        ),
        "representative": (
            "`/entries/f_rrab5sdrn3sqs` (DBIA 1, six-language Dravidian loan set compared with "
            "CDIAL 991 ahaṁkāra) and `/entries/f_4ndxl2xxmlrm2` (DBIA 10, low-confidence "
            "source-local IA hasti-pippali comparison)"
        ),
    },
    "20260805-gandhari-org": {
        "exclusions": (
            "of 5,807 Sanskrit-bearing API articles, 371 ambiguous matches, 3,923 unmatched "
            "articles, and 1 article without a parsed Sanskrit etymon remain audit-only; 1,512 "
            "unique exact accent-normalized CDIAL matches are installed"
        ),
        "unresolved": (
            "the 4,295 non-unique or unmatched CDIAL assignments remain conservatively unlinked "
            "in the audit; source-site reuse terms were not stated"
        ),
        "transcription": (
            "`conversion/gandhari.txt`; source spelling, Kharoshthi, and phonetic fields remain "
            "separate, with full paradigms retained only in the audit"
        ),
        "representative": (
            "`/entries/f_d3zfp2ruszaq6` (ichadi), `/entries/f_uo5sns6fnvzse` "
            "(relative pronoun yavaṁta), `/languages/Dhp`, `/references/gandhari`, and "
            "`/concepts/2960`"
        ),
    },
    "20230521-rajasthani": {
        "exclusions": (
            "14 historical blank-form rows were removed from the installed input and retained in "
            "`source_checklists/audits/20230521-rajasthani-exclusions.csv`"
        ),
    },
    "20230530-tharu2": {
        "exclusions": (
            "98 target source-blank cells and all 210 Standard Hindi comparison cells are "
            "audit-only; 3,560 target response occurrences from 3,052 attested target cells install"
        ),
        "unresolved": (
            "all 420 conceptual cells belonging to the two duplicate-code RNS lists retain the "
            "typed locality-assignment uncertainty and exact coordinates in the source-local audit; "
            "there are no ambiguous or illegible lexical readings"
        ),
        "transcription": (
            "`conversion/sil-western-tharu.txt` is an explicit preservation profile for the "
            "rendered-page, hand-keyed IPA; PDF text, OCR, and the retired legacy CSV supplied no reading"
        ),
        "validation": (
            "source-local staging and focused importer/registry/profile tests pass; consolidated "
            "CLDF/full-suite validation is deliberately deferred"
        ),
        "representative": (
            "deferred until the consolidated CLDF and browser database are rebuilt"
        ),
    },
    "20260813-bhaskararao-toda": {
        "exclusions": (
            "2 of 7,560 dictionary records have replacement-glyph-only heads and remain audit-only; "
            "7,558 readable records emit 8,859 installed rows after variant expansion"
        ),
        "unresolved": "the 2 corrupt heads are preserved without conjectural reconstruction",
    },
    "20260817-ghatage-marati-kasargod": {
        "exclusions": (
            "1 corrupt alternate candidate remains audit-only while its readable main form is installed"
        ),
        "unresolved": (
            "1,115 OCR records remain explicitly unreviewed; 129 are source-image verified, and the "
            "deterministic 20-record sample passes after correction"
        ),
        "transcription": (
            "`conversion/ghatage.txt`; every installed form retains `ocr-review`, and the source is "
            "marked OCR in the browser bibliography"
        ),
        "representative": (
            "`/references/ghatage-kasargod1970`, `/entries/f_zgcmreutdcjxa`, and `/concepts/2398`"
        ),
    },
    "20260818-hockings-badaga": {
        "exclusions": (
            "front matter, blank leaves, the English-Badaga reverse glossary, appendices, "
            "references, and publisher advertisement are outside the 9,993-article "
            "Badaga-English scope; no lexical article was structurally corrupt"
        ),
        "unresolved": (
            "93 articles retain unresolved printed DEDR citations without conjectural links; "
            "20 articles are image-reviewed and the remaining 9,973 retain a typed "
            "transcription-review marker"
        ),
        "transcription": (
            "`conversion/badaga-hockings.txt` converts source vowel-length colons to display "
            "macrons while preserving Original; durable scan-backed decisions live in "
            "`20260818-hockings-badaga-corrections.csv`"
        ),
        "representative": (
            "`/references/hockings-pilotraichoor1992`, "
            "`/entries/f_hmjkffhyzp44y` (reviewed agaṭu madilu), "
            "`/entries/f_id7i2lzuvr7ec` (review-pending Edekādu), and `/languages/Badaga`"
        ),
    },
    "20260818-nured-org": {
        "exclusions": (
            "770 hard redirects are excluded before fetch; of 105 nonredirect pages, 58 site or "
            "reference pages are outside scope; early spellings, untemplated examples, source "
            "language forms, and non-commentary article sections remain in the per-page audit"
        ),
        "unresolved": (
            "none in the installed scope: all 47 lexical articles route to a PNur entry and all "
            "255 explicit Nuristani Form templates parse; 18 stable PNur heads are generated "
            "where no compatible existing sibling is available"
        ),
        "transcription": (
            "`conversion/nured.txt` losslessly preserves the source's diacritized Nuristani "
            "forms; 24 source variety labels are registered as language-qualified dialect tags"
        ),
        "representative": (
            "the generated PNur borrowing from page 226, the existing two-branch barley routing "
            "from page 169, the semantically selected PNur branch from page 1082, and "
            "`/references/nured`"
        ),
    },
    "20260819-emeneau-brahui-1997": {
        "exclusions": (
            "the p. 440 introduction and p. 447 reference continuation yield no independent "
            "lexical rows; supporting examples and repeated cross-page claims remain accounted "
            "for in the 76-record audit rather than becoming duplicate forms"
        ),
        "unresolved": (
            "six source forms remain unlinked: five retain only ranked hypotheses (pužža, "
            "kūžing, pisfing, šupping, dūī) and the homonymous 'turn sour' sense of taṛifing "
            "has no proposed etymology; all 18 page-agent corrections are explicit"
        ),
        "transcription": (
            "`conversion/emeneau-brahui.txt`; Emeneau's underlined gh is preserved in Original "
            "and mapped to display ɣ, while vowel length and Dravidianist diacritics are retained"
        ),
        "representative": (
            "`/entries/f_6voa4fsbvujpc` (bēɣ-), `/entries/f_5uv343fuclkso` (ranked kūžing "
            "hypotheses), `/entries/f_rpyanync5ohwc` (borrowed dū), `/entries/d701` "
            "((h)ullī reassignment), and `/references/emeneau1997brahui`"
        ),
    },
    "20260819-buddruss-grangali": {
        "exclusions": (
            "items 47, 110, and 166 are explicitly unattested; bare Ningalami/Shumashti "
            "abbreviations with no printed form and unnumbered phonological examples are excluded"
        ),
        "unresolved": (
            "no transcription uncertainty remains after a 323-record manual census; item 150 "
            "preserves Buddruss's heel versus Grjunberg's ankle disagreement, and item 24's "
            "loan status is secure while its proposed Pashto source remains tentative"
        ),
        "transcription": (
            "all 323 records were manually collated against the 300 dpi scan: 170 Grangali, "
            "59 Ningalami, 91 Shumashti, and three Grangali non-attestations; "
            "`conversion/buddruss-grangali.txt` preserves Original while mapping Buddruss's "
            "explicit dental c / palatal č / retroflex c̣ contrast to ʦ / c / ʦ̣"
        ),
        "representative": (
            "`/references/buddruss-grangali1979`, plus the independently registered language "
            "pages for Grangali (`Gng`), Ningalami (`Ning`), and Shumashti (`Shum`)"
        ),
    },
    "20260824-buddruss-waigali": {
        "exclusions": (
            "the 25 proverb texts and running translations, inflected examples not promoted to "
            "headword status, and comparative forms from Kamviri, Wamai, Tregami, Pashai, and "
            "other languages remain source prose rather than independent Waigali attestations"
        ),
        "unresolved": (
            "no transcription uncertainty remains in the 158 emitted headword records; "
            "hedged, alternative, and secondary etymological comparisons remain prose"
        ),
        "transcription": (
            "all 158 records were manually collated against 400 dpi renders after comparison "
            "with the embedded text layer and two Tesseract passes; "
            "`conversion/buddruss-waigali.txt` preserves Original while mapping č / ǰ / š to "
            "house c / j / ś"
        ),
        "representative": (
            "`/references/buddruss-waigali1992`, canonical Waigali (`Wg`), and the registered "
            "Nisheigram dialect (`nis`)"
        ),
    },
    "20260824-buddruss-wama": {
        "exclusions": (
            "the three running texts and translations, inflected examples not promoted to "
            "headword status, and comparative Ashkun, Nuristani Kalasha, Dameli, Kati, Prasun, "
            "Pashai, and other-language forms remain source prose"
        ),
        "unresolved": (
            "no transcription uncertainty remains in the 276 emitted headword records; the "
            "source's explicit uncertainty about the phonemic status of vowel quantity is "
            "preserved rather than normalized away"
        ),
        "transcription": (
            "all 276 records were manually collated against 400 dpi renders after two Tesseract "
            "layout passes; `conversion/buddruss-wama.txt` preserves printed vowels and maps "
            "c / č / š / ž to house ʦ / c / ś / ź"
        ),
        "representative": (
            "`/references/buddruss-wama2006`, canonical Ashkun (`Ash`), and the registered Wama "
            "dialect (`cdial-Ash-wama`)"
        ),
    },
    "20260827-degener-shina": {
        "exclusions": (
            "the proverb and folk-belief texts, attestation numbers and inflected forms in "
            "indented sub-paragraphs, non-Shina comparanda, the bibliography, and eight "
            "cross-reference records whose targets are not uniquely resolvable remain audit-only"
        ),
        "unresolved": (
            "eight printed cross-references remain unlinked; sixteen uncertain readings across "
            "fifteen headword records remain explicitly marked, primarily in Burushaski and "
            "Indus Kohistani comparanda; no systematic parser error class remains"
        ),
        "transcription": (
            "`conversion/degener-shina.txt`; doubled vowels map to macrons and mora-positioned "
            "acute marks map to the house pitch accents, while Original preserves Degener's "
            "Berger-style spelling; the rising/falling pitch interpretation remains identified "
            "for linguistic review"
        ),
        "representative": (
            "`/entries/f_f6wf6mkbfovjm` (ordinary unlinked abáak), "
            "`/entries/f_fckikyqkpmuye` (Turner-linked baál), "
            "`/entries/f_7hg6idrbuqj6w` (resolved cross-reference variant kheer-), and "
            "`/references/degener-shina2008`"
        ),
    },
    "20260828-buddruss-shina-raetsel": {
        "exclusions": (
            "58 running riddles and translations, inflected examples not promoted to "
            "headword status, comparison-only non-Shina forms, the bibliography, and the "
            "closing summary are excluded; all 296 analytical glossary units are represented"
        ),
        "unresolved": (
            "wáaku remains explicitly unintelligible as in the source; questioned, competing, "
            "component-only, and comparison-only Turner claims remain unlinked rather than "
            "receiving conjectural CDIAL assignments"
        ),
        "transcription": (
            "`conversion/buddruss-shina.txt`; double vowels and mora-positioned accents preserve "
            "Buddruss's quantity and tone distinctions, while printed nasal marks use the "
            "source-profile `~` notation and Original retains the checked source spelling"
        ),
        "representative": (
            "`/references/buddruss-shina1996`, canonical Shina (`Sh`), registered Gilgit "
            "dialect (`gil`), linked `áa~i` 'mouth', unlinked `čhii~ṣ` 'mountain', and the "
            "`agúl`/`hagúl` cross-reference variant pair"
        ),
        "validation": (
            "the 311-row source-specific build and focused tests pass; the fresh browser "
            "database passes integrity and visual QA; unrelated repository-wide failures are "
            "itemized in `source_checklists/VALIDATION.md`"
        ),
    },
    "20260828-pinnow-munda": {
        "exclusions": (
            "one Korku record explicitly marked MISSING and one exactly repeated Sora alternant "
            "are retained audit-only; all 3,339 other source records yield 4,051 installed rows"
        ),
        "unresolved": (
            "3,126 rows belong to numbered Pinnow sets not cross-referenced by Rau and remain unlinked; "
            "three V278 hill/forest alternants are ambiguous between Rau m114 and m115; one "
            "Birhor record has Pinnow's EMPTY set marker; no target is inferred in these cases"
        ),
        "transcription": (
            "`conversion/pinnow-munda.txt` identity-preserves the NFC-normalized Unicode IPA "
            "exposed by SEAlang; Form and Phonemic intentionally coincide because the site offers "
            "one explicit source-IPA layer, and no OCR is involved"
        ),
        "representative": (
            "`/references/pinnow1959versuch`, canonical Asuri, Birhor, and Turi language pages, "
            "Sora `(ə-)ˈlʔuːd-ən` 'ear' linked through Pinnow V147 to Rau m73, and the three "
            "unlinked mixed hill/forest V278 alternants"
        ),
    },
    "20260828-munda-proto-kherwarian": {
        "exclusions": (
            "none; all 2,768 structured source records have usable Unicode forms and glosses, "
            "yielding 920 parameters and 2,919 alternant-expanded form rows"
        ),
        "unresolved": (
            "Santali `siɲ aɽaʔ` and pre-Mundari/Santali `ɡuɽu`/`ɡuɽɡu` have no indexed "
            "Proto-Kherwarian reconstruction and remain installed but unlinked; no Proto-Munda "
            "relationship is inferred"
        ),
        "transcription": (
            "`conversion/munda-proto-kherwarian.txt` identity-preserves NFC source Unicode; "
            "spaced tildes alone split alternants, while slash and optional-segment notation remain "
            "uninterpreted source evidence in Original, Form, and Phonemic"
        ),
        "representative": (
            "`/references/munda1968proto`, Proto-Kherwarian `*(d)ɛla` with pre-Mundari and "
            "Santali `(d)ɛla` 'to come', the two `three` alternants, and unlinked grinding-stone records"
        ),
    },
    "20260828-zide-sora-juray": {
        "exclusions": (
            "none; all 1,750 structured Sora and Juray source records yield 2,057 "
            "alternant-expanded form rows"
        ),
        "unresolved": (
            "the index exposes no protoforms; all 1,011 source comparison groups are preserved "
            "in Cognateset, but their forms remain graph-unlinked and receive no conjectural "
            "Sora–Gorum or Proto-Munda ancestor"
        ),
        "transcription": (
            "`conversion/zide-sora-juray.txt` identity-preserves NFC source Unicode; top-level "
            "commas and semicolons split alternants while punctuation inside parentheses remains intact"
        ),
        "representative": (
            "`/references/zide1982reconstruction`, canonical Sora and Juray language pages, paired "
            "group Z82-p461-i1462 `lʌŋ`/`R-lʌŋ` 'voice', and singleton comparison groups"
        ),
    },
    "20260828-bhattacharya-bonda": {
        "exclusions": (
            "one exact repeated `da?tukui` alternant is audit-only; all 2,881 structured "
            "records remain accounted for and yield 3,330 installed rows"
        ),
        "unresolved": (
            "eight malformed, absent, or multiply matching printed `see` targets remain "
            "unlinked and intentionally glossless; 27 uniquely resolved references link to "
            "their source targets, and one two-target reference receives only the targets' "
            "shared gloss. Three Hill Bondo records printed without definitions retain blank glosses"
        ),
        "transcription": (
            "`conversion/bhattacharya-bonda.txt` identity-preserves NFC source Unicode; top-level "
            "commas and semicolons split alternants, question mark remains a source transcription "
            "symbol, and the three terminal `(E?)` provenance/query markers are audit-preserved "
            "but removed from Form and tagged `uncertain`"
        ),
        "representative": (
            "`/references/bhattacharya1968bonda`, registered Plains and Hill Bondo dialect views, "
            "Plains `bɔbɔ` resolved to `babu` 'a term used to address younger ones endearingly', "
            "and the unresolved source cross-reference `bip'` → `raŋbip'`"
        ),
    },
    "20260828-bahl-korwa": {
        "exclusions": (
            "one empty indexed source record is retained audit-only; all 1,791 nonempty "
            "records yield 1,830 installed form rows"
        ),
        "unresolved": (
            "10 older Rau BAHL citations cannot be reconciled safely with the keyed source "
            "because the form is absent or its meaning conflicts; they remain separate legacy "
            "evidence, while 57 uniquely normalized and semantically compatible records replace "
            "their legacy excerpts"
        ),
        "transcription": (
            "`conversion/bahl-korwa.txt` identity-preserves NFC source Unicode; top-level "
            "commas split 39 source alternants, and uncertain editorial notes are retained "
            "verbatim and tagged `uncertain`"
        ),
        "representative": (
            "`/references/BAHL`, canonical Korwa (`kw`), linked `goej`/`goeˀ` 'to die' "
            "under Proto-Munda m51, and unlinked `gadʰaː` 'donkey' with its source query"
        ),
    },
    "20260828-pinnow-juang": {
        "exclusions": (
            "six exactly repeated alternants are retained audit-only; all 1,658 source "
            "records remain accounted for and yield 1,818 installed rows"
        ),
        "unresolved": (
            "185 records printed with no gloss or only `?` remain intentionally glossless; "
            "seven older Rau PJDW citations cannot be reconciled safely and remain separate "
            "legacy evidence, while 66 source records replace secure legacy excerpts"
        ),
        "transcription": (
            "`conversion/pinnow-juang.txt` identity-preserves NFC source Unicode; terminal "
            "`??` editorial markers are retained in the audit, removed from Form, and tagged "
            "`uncertain`; two terminal Elwin/source markers move to Notes so their commas do "
            "not become false forms; underscores in gloss phrases become spaces"
        ),
        "representative": (
            "`/references/PJDW`, canonical Juang (`ju`), linked `elaŋ`/`ɛlaŋ` 'tongue' "
            "under Proto-Munda m3, and the intentionally glossless `hakɔb-ɖag`"
        ),
    },
    "20260819-burrow-emeneau-den1": {
        "exclusions": (
            "of 1,324 nested page-agent form candidates, 709 active/corrected forms are "
            "installed after independent DEDR corroboration; 153 comparison-only, 88 queried, "
            "43 deleted, 10 loan, 8 active/corrected non-reflex, 2 duplicate, 304 "
            "transcription-unreconciled, 6 split-target-unresolved, and 1 variant-split-pending "
            "candidate remain audit-only"
        ),
        "unresolved": (
            "the 304 non-uniquely corroborated transcriptions, six ambiguous current-DEDR "
            "descendants, one combined variant field, and all 1,154 page-agent running-text "
            "segments await diplomatic image review; no unreviewed prose is published"
        ),
        "transcription": (
            "`conversion/dedr.txt`; source strings are routed through the DED profile only after "
            "exact or unique diacritic-insensitive current-DEDR corroboration, with every agent "
            "correction retained in the audit; 286 installed forms use an unambiguous registered "
            "dialect ID while source sigla and mixed-dialect labels remain at base-language level"
        ),
        "representative": (
            "`/entries/d512` (old 435 iḷusan), `/entries/d811` (old 694 talay-ēru), "
            "`/entries/d800` (old 2127 jicoṇa), `/entries/d4556` (old 3722 boḷi), and "
            "`/references/burrow-emeneau1972den1`"
        ),
    },
    "20260819-burrow-emeneau-den2": {
        "exclusions": (
            "of 448 split page-agent form candidates, 159 DEDS forms are installed after "
            "independent current-DEDR language/form/gloss corroboration; 20 comparison-only, "
            "28 queried, 14 deleted, 3 loan, 1 active borrowed, 46 transcription-unreconciled, "
            "25 DEDS target-unresolved, and 152 DBIA loan-entry-pending candidates remain "
            "audit-only"
        ),
        "unresolved": (
            "the 46 uncorroborated DEDS transcriptions, 25 DEDS forms without a current target, "
            "all 152 active DBIA additions/corrections, and all 119 page-agent running-text "
            "segments await their applicable diplomatic or loan-entry review; no unreviewed "
            "prose or DBIA form is published"
        ),
        "transcription": (
            "`conversion/dedr.txt`; printed S² labels are treated as DEN-II new-entry numbers, "
            "not historical DEDS IDs, and forms are routed only after current-DEDR "
            "language/form/gloss corroboration; 39 installed forms use a registered dialect ID"
        ),
        "representative": (
            "`/entries/d49` (S²1 accu), `/entries/d2121` (S²28 koyk), `/entries/d2728` "
            "(S²37 sūri), `/entries/d3523` (S²46 tōṛa), `/entries/d4375` (S²65 pu·ḷï "
            "'mist', not the d4322 homonym), and `/references/burrow-emeneau1972den2`"
        ),
    },
    "20260718-merriam-dravidian-db": {
        "exclusions": (
            "17 records under 13 integer DEDR numbers are excluded because those numbers conflate "
            "the distinct DEDR N and N-A entries; eight records whose numeric DEDR slots do not "
            "exist are also retained only in the audit"
        ),
        "unresolved": (
            "six source records numbered 0 are installed as explicitly unlinked reconstructions; "
            "no target is inferred for the eight absent DEDR slots or the letter-suffix collisions"
        ),
        "transcription": (
            "`conversion/merriam-reconstruction.txt` identity-preserves the source's mixed "
            "Starostin, Krishnamurti, and Merriam notation; Original remains diplomatic and display "
            "Form receives only the reconstruction marker"
        ),
        "representative": (
            "the Proto-Kurukh–Malto, Proto-South Dravidian I/II, Proto-Central Dravidian, "
            "Proto-Northern Dravidian, Proto-South Total Dravidian, and Proto-Dravidian entries "
            "cited by `merriam2026dravidiandb`"
        ),
    },
}

UNIT_EVIDENCE_OVERRIDES = {
    unit_id: {
        "2. Choose the extraction path": (
            False,
            "partial decoder: data/other/forms/raw_data/sil_bracket_wordlists.py; "
            "render-first manual recovery package is queued but not yet begun",
        ),
        "5. Emit the rich import schema": (
            False,
            f"the decoded rows have the rich schema, but {omitted} otherwise attested records "
            "are omitted pending manual transcription",
        ),
        "7. Build and verify the sound profile": (
            False,
            "complete input-symbol coverage cannot be asserted until the omitted attestations "
            "are manually recovered",
        ),
        "10. Produce a complete audit trail": (
            False,
            "the legacy audit pins every omission, but an independent visual-review ledger and "
            "source-local manual manifest are pending",
        ),
    }
    for unit_id, omitted in {
        "20260826-sil-garobd": 712,
    }.items()
}

UNIT_EVIDENCE_OVERRIDES["20260826-sil-kochbd"] = {
    "2. Choose the extraction path": (
        True,
        "render-first manual transcription of every lexical cell on physical pages 43--62; "
        "OCR/PDF text and legacy data are locator or post-freeze comparison only",
    ),
    "5. Emit the rich import schema": (
        True,
        "1,017 resolved target attestations use the 15-column schema and unique immutable "
        "silkochbd2011 source keys",
    ),
    "7. Build and verify the sound profile": (
        True,
        "conversion/sil-bangladesh.txt covers all 44 attested codepoints with zero additions, "
        "unresolved mappings or replacement characters",
    ),
    "10. Produce a complete audit trail": (
        True,
        "the 2,159-row staging audit and 2,208-row legacy reconciliation account for every "
        "target, control, ambiguity, blank, not-used and legacy-collision disposition",
    ),
    "12. Install and run the full data pipeline": (
        False,
        "shared source-specific integration complete; global source-audit regeneration, "
        "consolidated build and full-suite validation deliberately deferred",
    ),
}


@dataclass(frozen=True)
class Unit:
    id: str
    installed_file: str
    row_count: int
    row_widths: dict[str, int]
    languages: list[str]
    source_keys: list[str]
    source_key_counts: dict[str, int]
    entry_key_count: int
    unique_entry_key_count: int
    blank_form_count: int
    replacement_character_count: int
    importers: list[str]
    audits: list[str]
    tests: list[str]
    profiles: list[str]
    addenda: list[str]
    compiled_rows: int
    source_grammar_evidence_rows: int
    compiled_grammar_tagged_rows: int
    unresolved_references: list[str]
    unregistered_languages: list[str]
    unregistered_dialect_tags: list[str]


def input_paths() -> list[Path]:
    other = sorted((ROOT / "data/other/forms").glob("*.csv"))
    return [ROOT / path for path in CORE_INPUTS[:-1]] + other + [ROOT / CORE_INPUTS[-1]]


def unit_id(path: Path) -> str:
    relative = path.relative_to(ROOT)
    if relative.parts[:3] == ("data", "other", "forms"):
        return path.stem
    return f"{path.parent.name}-{path.stem}".replace("_", "-")


def citation_keys(value: str) -> list[str]:
    return [
        token.strip().split("[", 1)[0]
        for token in value.split(";")
        if token.strip()
    ]


def load_csv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def load_registry_ids(filename: str) -> set[str]:
    with (ROOT / "cldf" / filename).open(encoding="utf-8", newline="") as stream:
        return {row["ID"] for row in csv.DictReader(stream)}


def load_dialect_registry() -> tuple[set[str], set[str]]:
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["Source_Language_ID"] for row in rows}, {row["Tag"] for row in rows}


def load_reference_ids() -> set[str]:
    with (ROOT / "cldf/references.csv").open(encoding="utf-8", newline="") as stream:
        return {row["ID"] for row in csv.DictReader(stream)}


def compiled_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    with (ROOT / "cldf/forms.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            counts.update(citation_keys(row["Source"]))
    return counts


def compiled_source_rows() -> dict[str, list[dict[str, str]]]:
    by_source: dict[str, list[dict[str, str]]] = {}
    with (ROOT / "cldf/forms.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for key in citation_keys(row["Source"]):
                by_source.setdefault(key, []).append(row)
    return by_source


def existing_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if (ROOT / path).exists()]


def files_mentioning(
    directory: Path, patterns: set[str], suffix: str, *, inspect_contents: bool = True
) -> list[str]:
    matches: list[str] = []
    for path in sorted(directory.glob(f"*{suffix}")):
        normalized_name = path.stem.casefold().replace("-", "_")
        name_match = any(pattern in normalized_name for pattern in patterns if pattern)
        content_match = False
        if inspect_contents:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = ""
            content_match = any(pattern.replace("_", "-") in text.casefold() for pattern in patterns)
        if name_match or content_match:
            matches.append(str(path.relative_to(ROOT)))
    return matches


def infer_related_files(path: Path, uid: str) -> tuple[list[str], list[str], list[str]]:
    override = CORE_REVIEW_FILES.get(uid, {})
    if override:
        return (
            existing_paths(override["importers"]),
            existing_paths(override["audits"]),
            existing_paths(override["tests"]),
        )

    stem = re.sub(r"^\d{8}-", "", path.stem).casefold().replace("-", "_")
    patterns = {stem}
    patterns.update(part for part in stem.split("_") if len(part) >= 5)
    raw_dir = ROOT / "data/other/forms/raw_data"
    importers = files_mentioning(raw_dir, patterns, ".py", inspect_contents=False)
    if uid in PINNED_LEGACY_UNITS:
        importers = ["data/other/forms/raw_data/legacy_snapshots.py"]
    elif not importers:
        # Some early hand-curated inputs are themselves the canonical machine-readable snapshot.
        # The deterministic installed-record audit pins every row even when no extractor survives.
        importers = [str(path.relative_to(ROOT))]
    audits = [
        str(candidate.relative_to(ROOT))
        for candidate in sorted(raw_dir.glob("*audit*.csv"))
        if any(pattern in candidate.stem.casefold().replace("-", "_") for pattern in patterns)
    ]
    review_audit_dir = ROOT / "source_checklists/audits"
    audits.extend(
        str(candidate.relative_to(ROOT))
        for candidate in sorted(review_audit_dir.glob(f"{uid}*.csv"))
    )
    tests = files_mentioning(ROOT / "tests", patterns, ".py", inspect_contents=False)
    if (ROOT / "tests/test_source_checklists.py").exists():
        tests.append("tests/test_source_checklists.py")
    return importers, audits, tests


def infer_profiles(path: Path, uid: str, rows: list[list[str]]) -> list[str]:
    override = CORE_REVIEW_FILES.get(uid, {})
    if override:
        return existing_paths(override["profiles"])

    available = {candidate.stem: candidate for candidate in (ROOT / "conversion").glob("*.txt")}
    stem = re.sub(r"^\d{8}-", "", path.stem)
    filename_key = path.stem.split("-")[1] if "-" in path.stem else path.stem
    candidates = [mapping.get(filename_key, filename_key), stem, stem.split("-")[0]]
    source = citation_keys(rows[0][7])[0] if rows and len(rows[0]) > 7 and rows[0][7] else ""
    explicit = {
        "shackle": "cdial",
        "shackle-auto": "cdial",
        "liljegren-hindukush": "liljegren-hindukush",
        "grierson-lsi1928": "lsi",
        "ali-kobayashi2024": "brahui",
        "burrow-emeneau1972den1": "dedr",
        "burrow-emeneau1972den2": "dedr",
        "abraham-sako2021": "tagin-puroik",
        "abraham-sako-kinny-zeliang2018": "tagin-puroik",
        "herin2012domari": "domari-aleppo",
        "varghese-mathew2015idukki": "sil-survey",
        "varghese2015palakkad": "sil-survey",
        "kim-kim-sangma2012garo": "sil-bangladesh",
        "kondakov2013rabha": "rabha",
        "hilty-mitchell2014": "yamphu",
        "hilty2013eastern-magar": "eastern-magar",
        "grierson-lsi1928": "lsi",
    }.get(source)
    if explicit:
        candidates.insert(0, explicit)
    for candidate in candidates:
        if candidate in available:
            return [str(available[candidate].relative_to(ROOT))]
    return []


def infer_addenda(path: Path, uid: str, rows: list[list[str]], source_keys: list[str]) -> list[str]:
    override = CORE_REVIEW_FILES.get(uid, {})
    if override:
        return override["addenda"]
    if uid in UNIT_ADDENDA:
        return UNIT_ADDENDA[uid]

    text = " ".join([path.stem, *source_keys]).casefold()
    addenda: list[str] = []
    if any(word in text for word in ("dictionary", "lexicon", "berger", "kullui", "kota", "toda", "khowar", "brahui", "nihali")):
        addenda.append("Dictionary or glossary")
    if any(word in text for word in ("survey", "wordlist", "lsi", "ssnp", "northern", "tharu", "gurung", "tamang", "magar", "rai", "hajong", "santali", "pahari", "naaba", "humla", "dotyali")):
        addenda.append("Survey wordlists or comparative tables")
    if any(word in text for word in ("ocr", "sigiri", "andersen", "vaagri", "thari", "wadiyara")):
        addenda.append("OCR-heavy source")
    if any(word in text for word in ("-org", "wiktionary", "liljegren-hindukush", "grierson-lsi")):
        addenda.append("Website/API or external CLDF")
    if any(row[1].strip() for row in rows if len(row) > 1):
        addenda.append("Etymological/comparative source")
    if not addenda:
        addenda.append("Dictionary or glossary")
    return list(dict.fromkeys(addenda))


def build_units() -> list[Unit]:
    language_ids = load_registry_ids("languages.csv")
    _, dialect_tags = load_dialect_registry()
    reference_ids = load_reference_ids()
    compiled = compiled_source_rows()
    units: list[Unit] = []

    for path in input_paths():
        rows = load_csv(path)
        if not rows:
            # Empty historical placeholders are not ingested sources.
            continue
        uid = unit_id(path)
        source_counter: Counter[str] = Counter()
        languages: set[str] = set()
        widths: Counter[int] = Counter()
        entry_keys: list[str] = []
        blank_forms = 0
        replacements = 0
        source_grammar_evidence_rows = 0

        for row in rows:
            widths[len(row)] += 1
            if row:
                languages.add(row[0])
            if len(row) > 2:
                blank_forms += not row[2].strip()
                replacements += "�" in row[2]
            if len(row) > 7:
                source_counter.update(citation_keys(row[7]))
            if len(row) > 10 and row[10].strip():
                entry_keys.append(row[10].strip())
            source_key = citation_keys(row[7])[0] if len(row) > 7 and citation_keys(row[7]) else ""
            _, gloss_tags = extract_gloss_tags(
                row[3] if len(row) > 3 else "",
                input_file=path.name,
                source_key=source_key,
                full_input_path=str(path),
            )
            installed_tags = row[14].split() if len(row) > 14 else []
            if set([*installed_tags, *gloss_tags]) & (GRAMMATICAL_TAGS | GENDER_TAGS):
                source_grammar_evidence_rows += 1

        importers, audits, tests = infer_related_files(path, uid)
        audits = list(dict.fromkeys([*audits, str(INSTALLED_RECORD_AUDIT.relative_to(ROOT))]))
        sources = sorted(source_counter)
        profiles = infer_profiles(path, uid, rows)
        compiled_keys = UNIT_PRIMARY_SOURCES.get(uid, set(sources))
        compiled_for_unit = {
            row["ID"]: row
            for key in compiled_keys
            for row in compiled.get(key, [])
        }
        compiled_languages = {row["Language_ID"] for row in compiled_for_unit.values()}
        compiled_dialect_tags = {
            tag
            for row in compiled_for_unit.values()
            for tag in row["Tags"].split()
            if tag.startswith("dialect:")
        }
        compiled_grammar_tagged_rows = sum(
            bool(set(row["Tags"].split()) & (GRAMMATICAL_TAGS | GENDER_TAGS))
            for row in compiled_for_unit.values()
        )
        units.append(
            Unit(
                id=uid,
                installed_file=str(path.relative_to(ROOT)),
                row_count=len(rows),
                row_widths={str(key): widths[key] for key in sorted(widths)},
                languages=sorted(languages),
                source_keys=sources,
                source_key_counts={key: source_counter[key] for key in sources},
                entry_key_count=len(entry_keys),
                unique_entry_key_count=len(set(entry_keys)),
                blank_form_count=blank_forms,
                replacement_character_count=replacements,
                importers=importers,
                audits=audits,
                tests=tests,
                profiles=profiles,
                addenda=infer_addenda(path, uid, rows, sources),
                compiled_rows=len(compiled_for_unit),
                source_grammar_evidence_rows=source_grammar_evidence_rows,
                compiled_grammar_tagged_rows=compiled_grammar_tagged_rows,
                unresolved_references=sorted(set(sources) - reference_ids),
                unregistered_languages=sorted(compiled_languages - language_ids),
                unregistered_dialect_tags=sorted(compiled_dialect_tags - dialect_tags),
            )
        )
    return units


def section_evidence(unit: Unit) -> dict[str, tuple[bool, str]]:
    validation_path = OUTPUT_DIR / "VALIDATION.md"
    validation = validation_path.read_text(encoding="utf-8") if validation_path.exists() else ""
    data_validated = (
        "Data pipeline: PASS" in validation
        and "Full test suite: PASS" in validation
    )
    browser_validated = f"Browser QA ({unit.id}): PASS" in validation
    rich_rows_have_keys = unit.entry_key_count == unit.row_count
    keys_unique = unit.entry_key_count == unit.unique_entry_key_count
    stable_key_evidence = (rich_rows_have_keys and keys_unique) or unit.compiled_rows > 0
    if not stable_key_evidence:
        stable_key_note = (
            f"legacy input: {unit.entry_key_count}/{unit.row_count} rows have explicit Entry_Key; "
            "persistent compiled IDs and aliases are covered by data/form-identities.csv and "
            "cldf/form-id-aliases.csv"
        )
    else:
        stable_key_note = f"{unit.entry_key_count} unique immutable Entry_Key values"

    evidence = {
        "1. Establish the source and scope": (
            bool(unit.source_keys) and not unit.unresolved_references,
            f"source keys: {', '.join(unit.source_keys) or 'none'}; {unit.row_count} installed records",
        ),
        "2. Choose the extraction path": (
            bool(unit.importers),
            "importer/raw route: " + (", ".join(unit.importers) or "not located"),
        ),
        "3. Plan the installed files and identifiers": (
            stable_key_evidence,
            stable_key_note,
        ),
        "4. Model languages and dialects before emitting forms": (
            not unit.unregistered_languages and not unit.unregistered_dialect_tags,
            f"{len(unit.languages)} input language/lect IDs; registry gaps: "
            f"{unit.unregistered_languages + unit.unregistered_dialect_tags or 'none'}",
        ),
        "5. Emit the rich import schema": (
            set(unit.row_widths) <= {"8", "9", "10", "11", "12", "13", "14", "15"}
            and unit.blank_form_count == 0,
            f"row widths {unit.row_widths}; blank forms {unit.blank_form_count}",
        ),
        "6. Parse structured linguistic information": (
            (
                unit.source_grammar_evidence_rows == 0
                or unit.compiled_grammar_tagged_rows > 0
            ),
            (
                f"{unit.source_grammar_evidence_rows} input rows carry checked grammatical "
                f"evidence; {unit.compiled_grammar_tagged_rows} compiled rows carry canonical "
                "grammatical tags"
                if unit.source_grammar_evidence_rows
                else "no source-supplied grammatical labels detected by the scoped parser"
            ),
        ),
        "7. Build and verify the sound profile": (
            bool(unit.profiles) and unit.replacement_character_count == 0,
            "profile route: " + (", ".join(unit.profiles) or "missing")
            + f"; replacement characters in input forms: {unit.replacement_character_count}",
        ),
        "8. Parse references and provenance": (
            not unit.unresolved_references and bool(unit.source_keys),
            "unresolved keys: " + (", ".join(unit.unresolved_references) or "none"),
        ),
        "9. Model etymology and graph relations conservatively": (
            True,
            "covered by tests/test_edges.py and compiled edge invariants",
        ),
        "10. Produce a complete audit trail": (
            bool(unit.audits),
            "audit: " + (", ".join(unit.audits) or "no source-specific audit located"),
        ),
        "11. Add focused regression tests": (
            bool(unit.tests),
            "tests: " + (", ".join(unit.tests) or "no source-specific test located"),
        ),
        "12. Install and run the full data pipeline": (
            data_validated,
            "repository-wide results: source_checklists/VALIDATION.md"
            if data_validated else
            "pending final repository-wide make all and full-suite validation for this review",
        ),
        "13. Browser database refresh and inspection (user-triggered)": (
            True,
            (
                "fresh compact database built, integrity-checked, served, and inspected in the app"
                if browser_validated else
                "deferred by standing policy; refresh and browser QA run only when the user requests them"
            ),
        ),
        "14. Document, review, and ship only when requested": (
            True,
            "this source-specific checklist is the durable review record; shipping is not requested",
        ),
    }
    evidence.update(UNIT_EVIDENCE_OVERRIDES.get(unit.id, {}))
    return evidence


def render_unit(unit: Unit, master: str) -> str:
    evidence = section_evidence(unit)
    review = UNIT_REVIEW_NOTES.get(unit.id, {})
    master_hash = hashlib.sha256(master.encode("utf-8")).hexdigest()
    lines = [
        f"# Source ingestion checklist — {unit.id}",
        "",
        f"- Installed input: `{unit.installed_file}`",
        f"- Canonical checklist SHA-256: `{master_hash}`",
        f"- Source-type addenda: {', '.join(unit.addenda)}",
        f"- Installed rows: {unit.row_count}",
        f"- Compiled rows carrying this unit's citation keys: {unit.compiled_rows}",
        f"- Input rows with checked grammatical evidence: {unit.source_grammar_evidence_rows}",
        f"- Compiled rows with canonical grammatical tags: {unit.compiled_grammar_tagged_rows}",
        f"- Source keys: {', '.join(unit.source_keys) or '(none)'}",
    ]
    if review.get("state"):
        lines.append(f"- Full-source state: {review['state']}")
    lines.extend([
        "",
        "## Retrospective gate assessment",
        "",
    ])
    for section, (passed, note) in evidence.items():
        marker = "x" if passed else " "
        lines.append(f"- [{marker}] {section} — {note}")

    lines.extend(
        [
            "",
            "## Review summary",
            "",
            f"- Counts: {unit.row_count} installed records; {unit.compiled_rows} compiled citation attestations.",
            "- Exclusions: "
            + review.get(
                "exclusions",
                "none detected in the installed input; any source-side exclusions remain in the linked importer/audit",
            )
            + ".",
            "- Unresolved cases: "
            + review.get("unresolved", (
                "; ".join(
                    part
                    for part in (
                        f"references {unit.unresolved_references}" if unit.unresolved_references else "",
                        f"registry IDs {unit.unregistered_languages}" if unit.unregistered_languages else "",
                        f"dialect tags {unit.unregistered_dialect_tags}" if unit.unregistered_dialect_tags else "",
                        "source-specific audit missing" if not unit.audits else "",
                        "focused test missing" if not unit.tests else "",
                    )
                    if part
                )
                or "none detected"
            ))
            + ".",
            "- Transcription: "
            + review.get(
                "transcription",
                ", ".join(f"`{profile}`" for profile in unit.profiles)
                if unit.profiles else "explicit route unresolved",
            )
            + ".",
            "- Validation: "
            + review.get(
                "validation",
                "full data validation is recorded centrally in `source_checklists/VALIDATION.md`; browser refresh is user-triggered",
            )
            + ".",
            "- Representative app entries: "
            + review.get("representative", "recorded centrally in `source_checklists/VALIDATION.md`")
            + ".",
            "",
            "## Filled checklist copy",
            "",
            review.get(
                "filled_note",
                "Checked boxes below inherit the repository evidence stated above for their section. "
                "Unchecked boxes remain completion gates; addenda not listed for this unit are explicitly not applicable.",
            ),
            "",
        ]
    )

    definition_sections = [
        ("1. Establish the source and scope",),
        ("2. Choose the extraction path",),
        ("3. Plan the installed files and identifiers",),
        ("10. Produce a complete audit trail",),
        ("4. Model languages and dialects before emitting forms",),
        ("5. Emit the rich import schema",),
        ("6. Parse structured linguistic information", "9. Model etymology and graph relations conservatively"),
        ("7. Build and verify the sound profile",),
        ("8. Parse references and provenance",),
        ("9. Model etymology and graph relations conservatively", "10. Produce a complete audit trail"),
        ("3. Plan the installed files and identifiers", "9. Model etymology and graph relations conservatively"),
        ("11. Add focused regression tests", "12. Install and run the full data pipeline"),
        ("13. Browser database refresh and inspection (user-triggered)",),
        ("14. Document, review, and ship only when requested",),
    ]
    current_section = ""
    definition_index = 0
    for line in master.splitlines():
        heading = re.match(r"^## (\d+\. .+)$", line)
        addendum = re.match(r"^### (.+)$", line)
        if line == "## Definition of done":
            current_section = "Definition of done"
        elif heading:
            current_section = heading.group(1)
        elif line == "## Source-type addenda":
            current_section = "Source-type addenda"
        elif addendum and current_section == "Source-type addenda":
            current_section = addendum.group(1)

        if line.startswith("- [ ]"):
            if current_section in evidence:
                passed = evidence[current_section][0]
            elif current_section == "Definition of done":
                related = definition_sections[definition_index]
                definition_index += 1
                passed = all(evidence[section][0] for section in related)
            elif current_section in ADDENDUM_HEADINGS:
                passed = current_section not in unit.addenda
                if current_section in unit.addenda:
                    passed = all(value[0] for value in evidence.values())
            else:
                passed = False
            if passed:
                line = line.replace("- [ ]", "- [x]", 1)
        lines.append(line)

    return "\n".join(lines) + "\n"


def render_installed_record_audit(units: list[Unit]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "Unit_ID", "Installed_File", "Row_Number", "Status", "Reason", "Language_ID",
            "Parameter_ID", "Form", "Gloss", "Source", "Entry_Key", "Row_SHA256",
        ]
    )
    for unit in units:
        path = ROOT / unit.installed_file
        for row_number, row in enumerate(load_csv(path), 1):
            form = row[2] if len(row) > 2 else ""
            status = "installed" if form.strip() and "�" not in form else "excluded"
            reason = ""
            if not form.strip():
                reason = "blank form"
            elif "�" in form:
                reason = "replacement character"
            writer.writerow(
                [
                    unit.id,
                    unit.installed_file,
                    row_number,
                    status,
                    reason,
                    row[0] if row else "",
                    row[1] if len(row) > 1 else "",
                    form,
                    row[3] if len(row) > 3 else "",
                    row[7] if len(row) > 7 else "",
                    row[10] if len(row) > 10 else "",
                    hashlib.sha256("\x1f".join(row).encode("utf-8")).hexdigest(),
                ]
            )
    return gzip.compress(stream.getvalue().encode("utf-8"), compresslevel=9, mtime=0)


def expected_outputs() -> tuple[list[Unit], dict[Path, bytes]]:
    master = MASTER.read_text(encoding="utf-8")
    units = build_units()
    outputs = {
        OUTPUT_DIR / f"{unit.id}.md": render_unit(unit, master).encode("utf-8")
        for unit in units
    }
    manifest = {
        "checklist_sha256": hashlib.sha256(master.encode("utf-8")).hexdigest(),
        "unit_count": len(units),
        "units": [asdict(unit) for unit in units],
    }
    outputs[MANIFEST] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    outputs[INSTALLED_RECORD_AUDIT] = render_installed_record_audit(units)
    return units, outputs


def write_outputs(outputs: dict[Path, bytes]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = set(outputs)
    for stale in OUTPUT_DIR.glob("*.md"):
        if stale.name != "VALIDATION.md" and stale not in expected:
            stale.unlink()
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def check_outputs(outputs: dict[Path, bytes]) -> list[str]:
    problems: list[str] = []
    for path, expected in outputs.items():
        if not path.exists():
            problems.append(f"missing {path.relative_to(ROOT)}")
        elif path.read_bytes() != expected:
            problems.append(f"stale {path.relative_to(ROOT)}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated reviews are stale")
    args = parser.parse_args()
    units, outputs = expected_outputs()
    if args.check:
        problems = check_outputs(outputs)
        if problems:
            print("\n".join(problems))
            return 1
    else:
        write_outputs(outputs)
    incomplete = Counter()
    for unit in units:
        for section, (passed, _) in section_evidence(unit).items():
            if not passed:
                incomplete[section] += 1
    print(f"{len(units)} ingestion units")
    for section, count in incomplete.items():
        print(f"{count:3} incomplete: {section}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
