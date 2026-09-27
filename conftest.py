"""Keep superseded source tests as provenance without collecting them as live tests.

These four files were archived alongside their pilot importers. Their original
relative-path assumptions are invalid at the archival location. Current full-source
coverage is exercised by the corresponding tests/*_full_stage.py modules.
"""

collect_ignore = [
    "data/other/forms/raw_data/bailey_bhalesi_1908/legacy-pilot/test_bailey_bhalesi_1908.py",
    "data/other/forms/raw_data/bailey_padari_1908/legacy-pilot/test_bailey_padari_1908.py",
    "data/other/forms/raw_data/grierson_malvi_rangri_1908/legacy-pilot/test_grierson_malvi_rangri_1908.py",
    "data/other/forms/raw_data/grierson_suketi_1916/legacy-pilot/test_grierson_suketi_1916.py",
]
