import csv
import json
import re
from collections import Counter
from pathlib import Path

from etymology_assignments import read_assignments


ROOT = Path(__file__).parents[1]
ANALYSIS = ROOT / "data/other/analysis/nihali-provisional"
MARKER = "Nihali provisional 2026"

# The reviewed September 2026 cohort is fixed to these five lexical sources.
# Later attestations (e.g. Zoller 2023) do not acquire its provisional hypotheses.
STUDY_SOURCES = {
    "nagaraja2014", "mundlay1996", "bhattacharya1957",
    "varghesekumar2015noira", "konow1906",
}


def is_study_attestation(row):
    sources = {part.split("[", 1)[0].strip() for part in row["Source"].split(";")}
    return (
        row["Language_ID"] == "Ni" and row["Status"] != "entry"
        and not row["ID"].startswith("nihprov-")
        and bool(sources & STUDY_SOURCES)
    )


def dicts(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_audit_covers_every_attested_nihali_record_once():
    with (ROOT / "cldf/forms.csv").open(encoding="utf-8", newline="") as stream:
        targets = [row for row in csv.DictReader(stream) if is_study_attestation(row)]
    audit = dicts(ANALYSIS / "nihali-provisional-etymology-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    # The reviewed cohort remains fixed. The later optional-length expansion adds
    # 22 short readings of those attestations, not new provisional hypotheses.
    assert len(targets) == 4321
    assert len(audit) == 4299
    audited_ids = {row["Form_ID"] for row in audit}
    with (ROOT / "data/other/forms/20260817-nagaraja-nihali-wiktionary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        short_keys = {row[10] for row in csv.reader(stream) if row[10].endswith(":short")}
    assert len(short_keys) == 22
    paired_keys = short_keys | {key.removesuffix(":short") for key in short_keys}
    with (ROOT / "cldf/form-source-keys.csv").open(encoding="utf-8", newline="") as stream:
        legacy_keys = {row["Legacy_ID"]: row["Source_Key"] for row in csv.DictReader(stream)
                       if row["Source_Key"] in paired_keys}
    with (ROOT / "cldf/form-id-aliases.csv").open(encoding="utf-8", newline="") as stream:
        keyed_ids = {legacy_keys[row["Legacy_ID"]]: row["Form_ID"] for row in csv.DictReader(stream)
                     if row["Legacy_ID"] in legacy_keys}
    short_ids = {keyed_ids[key] for key in short_keys}
    assert len(short_ids) == 22 and not (short_ids & audited_ids)
    assert {row["ID"] for row in targets} == audited_ids | short_ids
    with (ROOT / "cldf/edges.csv").open(encoding="utf-8", newline="") as stream:
        short_edges = [row for row in csv.DictReader(stream) if row["Child_ID"] in short_ids]
    assert len(short_edges) == 22
    assert {row["Child_ID"] for row in short_edges} == short_ids
    assert Counter(row["Kind"] for row in short_edges) == {"variant": 21, "borrowed": 1}
    assert all(row["Rank"] == "1" for row in short_edges)
    assert all(row["Parent_ID"] in audited_ids for row in short_edges if row["Kind"] == "variant")
    # he(ː)la already has the explicit CDIAL borrowing recorded in the reviewed
    # audit; both length readings retain that claim instead of adding a new one.
    long_id = keyed_ids["nagaraja2014-wiktionary:716"]
    long_audit = next(row for row in audit if row["Form_ID"] == long_id)
    borrowed = next(row for row in short_edges if row["Kind"] == "borrowed")
    assert borrowed["Child_ID"] == keyed_ids["nagaraja2014-wiktionary:716:short"]
    assert borrowed["Parent_ID"] == long_audit["Parent_ID"] == "14158"
    assert all(
        row["Parent_ID"] and row["Method"] and row["Confidence"]
        and row["Lexeme_ID"].startswith("nilex-") and int(row["Lexeme_Size"]) >= 1
        for row in audit
    )
    assert len({row["Lexeme_ID"] for row in audit}) == summary["lexeme_clusters"]
    assert set(row["Method"] for row in audit) == {
        "existing-curated", "source-resolved", "source-proxy",
        "cluster-resolved", "cluster-proxy", "manual-resolved",
        "manual-rejected", "manual-deferred", "residue-proxy",
    }


def test_every_computational_lead_has_an_explicit_manual_decision():
    audit = dicts(ANALYSIS / "nihali-provisional-etymology-audit.csv")
    separated = dicts(ANALYSIS / "nihali-computational-candidate-review.csv")
    ties = dicts(ANALYSIS / "nihali-low-margin-candidate-review.csv")
    transparent = dicts(ANALYSIS / "nihali-transparent-loan-review.csv")
    source_parents = dicts(ANALYSIS / "nihali-source-parent-review.csv")
    reviews = separated + ties + transparent + source_parents
    reviewed = [row for row in audit if row["Manual_Decision"]]
    assert len(separated) == 95
    assert len(ties) == 23
    assert len(transparent) == 25
    assert len(source_parents) == 19
    assert len(reviews) == 162
    assert {row["Lexeme_ID"] for row in reviews} == {row["Lexeme_ID"] for row in reviewed}
    assert Counter(row["Decision"] for row in reviews) == {
        "accept": 115, "reject": 29, "defer": 18,
    }
    assert all(row["Manual_Rationale"] for row in reviewed)
    assert Counter(row["Manual_Trigger"] for row in reviewed) == {
        "separated-candidate": sum(
            row["Lexeme_ID"] in {item["Lexeme_ID"] for item in separated} for row in reviewed
        ),
        "low-margin-family-tie": sum(
            row["Lexeme_ID"] in {item["Lexeme_ID"] for item in ties} for row in reviewed
        ),
        "transparent-cultural-loan": sum(
            row["Lexeme_ID"] in {item["Lexeme_ID"] for item in transparent}
            for row in reviewed
        ),
        "source-attributed-parent": sum(
            row["Lexeme_ID"] in {item["Lexeme_ID"] for item in source_parents}
            for row in reviewed
        ),
    }


def test_manual_candidate_adjudication_controls_external_linking():
    audit = dicts(ANALYSIS / "nihali-provisional-etymology-audit.csv")
    candidates = [row for row in audit if row["Manual_Decision"]]
    accepted = [row for row in candidates if row["Manual_Decision"] == "accept"]
    unresolved = [
        row for row in candidates
        if row["Manual_Decision"] in {"reject", "defer"}
        and row["Manual_Trigger"] != "source-attributed-parent"
    ]
    rejected_source_parents = [
        row for row in candidates
        if row["Manual_Decision"] == "reject"
        and row["Manual_Trigger"] == "source-attributed-parent"
    ]
    assert accepted and unresolved
    assert all(
        row["Method"] in {"manual-resolved", "existing-curated"}
        and row["Stratum"] != "Nihali residue"
        and not row["Parent_ID"].startswith("nihprov-")
        for row in accepted
    )
    assert all(
        row["Method"] in {"manual-rejected", "manual-deferred"}
        and row["Stratum"] == "Nihali residue"
        and row["Parent_ID"].startswith("nihprov-")
        and row["Parent_Language_ID"] == "Ni"
        and row["Alternatives"]
        for row in unresolved
    )
    assert all(
        row["Method"] == "source-proxy"
        and row["Stratum"] != "Nihali residue"
        and row["Parent_ID"].startswith("nihprov-")
        and row["Parent_Language_ID"] != "Ni"
        and row["Alternatives"]
        for row in rejected_source_parents
    )
    assert {row["Lexeme_ID"] for row in rejected_source_parents} == {
        "nilex-b0b1515862b206", "nilex-a34fe81ddfde3e", "nilex-7ab032de2d6baf",
        "nilex-628503046f43ca", "nilex-be5d4c0fbc974d", "nilex-c9926cb626bfd8",
        "nilex-8e68e203a2bea0", "nilex-55279a4979286c", "nilex-607fccef561832",
        "nilex-231eaf8a04d419", "nilex-05b1049435e1c4",
    }


def test_resolved_numbered_source_citations_route_to_the_cited_head():
    audit = dicts(ANALYSIS / "nihali-provisional-etymology-audit.csv")
    assignments = {
        row["Form_ID"]: row["Etymon_ID"]
        for row in read_assignments()
        if row["Rank"] == "1" and row["Status"] == "accepted"
    }
    upstream_only = set()
    for row in audit:
        if row["Method"] not in {"source-resolved", "manual-resolved"}:
            continue
        citations = {
            ("d" if kind.upper().startswith("DED") else "") + number
            for kind, number in re.findall(
                r"(CDIAL|DEDR|DED(?:\(S(?:, N)?\))?)\s*(?:#|no\.?|entry)?\s*(\d+)",
                row["Original_Etymology"], re.I,
            )
        }
        if citations and row["Parent_ID"] not in citations:
            assert any(assignments.get(citation) == row["Parent_ID"] for citation in citations)
            upstream_only.add(row["Lexeme_ID"])
    assert upstream_only == {
        "nilex-59c8c62db3b966", "nilex-c8e54a068d1408", "nilex-9fb6183f085c90",
    }


def test_transparent_cultural_loans_do_not_inflate_the_residue():
    audit = dicts(ANALYSIS / "nihali-provisional-etymology-audit.csv")
    forms = {row["ID"]: row for row in dicts(ROOT / "cldf/forms.csv")}
    by_form = {row["Form"]: row for row in audit}
    expected = {
        "gʰedyāl": ("4413", "Indo-Aryan"),
        "gor": ("4182", "Indo-Aryan"),
        "sikret": ("f_4ncc2xhiswbc2", "Indo-Aryan+English"),
        "lampṭāki": ("f_76ko65ts3oqb6", "Korku+English"),
        "tāmku": ("f_6xgtgfkwwwaus", "Indo-Aryan"),
    }
    for form, (parent_id, stratum) in expected.items():
        row = by_form[form]
        assert row["Method"] == "manual-resolved"
        assert row["Parent_ID"] == parent_id
        assert row["Stratum"] == stratum
        assert row["Manual_Decision"] == "accept"
        assert row["Manual_Rationale"]

    # Lock the selected parent surfaces, not just opaque IDs: this catches a stale or mistyped
    # durable ID that happens to exist but denotes an unrelated word.
    expected_parent_forms = {
        "gʰedyāl": "*gʰaṭītāḍa", "gor": "guḍá", "sikret": "cigarette",
        "sagai": "sʌgay", "voḍi": "vardʰaki", "lampṭāki": "lamp",
        "vorāri pakkā": "worari", "sajā": "Hsatyás", "dʰokā": "*dʰrōkṣa",
        "bhūt": "bʰūtá", "mircʰā": "marīca", "jiv": "jīvá",
        "maidān jāgā": "maidān", "sikāri": "śikārī", "hubehu": "hū-ba-hū",
        "tʰorā": "*stōkaḍ-", "dhankar": "*dʰaŋga", "rangā": "raŋga",
        "barsādo dino": "varṣārātri", "tisrā din": "tr̩tī́ya",
        "moṭʰā din": "*mōṭṭa-", "rojoka": "rōz", "rojoko": "rōz",
        "cimni-tel": "cʰimney", "bhītarkē": "*bʰiyantara",
    }
    transparent = dicts(ANALYSIS / "nihali-transparent-loan-review.csv")
    lexemes = {
        row["Lexeme_ID"]: row
        for row in dicts(ANALYSIS / "nihali-provisional-lexeme-audit.csv")
    }
    assert {
        lexemes[row["Lexeme_ID"]]["Representative_Form"]: forms[row["Parent_ID"]]["Form"]
        for row in transparent
    } == expected_parent_forms


def test_lexeme_audit_is_one_row_per_conservative_cluster():
    audit = dicts(ANALYSIS / "nihali-provisional-etymology-audit.csv")
    lexemes = dicts(ANALYSIS / "nihali-provisional-lexeme-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(lexemes) == summary["lexeme_clusters"]
    assert {row["Lexeme_ID"] for row in lexemes} == {row["Lexeme_ID"] for row in audit}
    assert sum(int(row["Record_Count"]) for row in lexemes) == 4299
    assert all(row["Stratum"] for row in lexemes)


def test_generated_proxy_display_forms_have_balanced_notation():
    with (ANALYSIS / "20260901-nihali-provisional.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        proxies = list(csv.reader(stream))
    assert len(proxies) == 2743
    assert all(len(row) == 5 for row in proxies)
    external = [row for row in proxies if row[1] != "Ni"]
    assert all(
        form.count("(") == form.count(")")
        and form.count("[") == form.count("]")
        and not form.endswith((",", ";", ":", "/", "("))
        and "\ufffd" not in form
        for _proxy_id, _language_id, form, _gloss, _source in external
    )
    by_id = {row[0]: row[2] for row in proxies}
    assert by_id["nihprov-024e39aa3d7520"] == "*jhapp-"
    assert by_id["nihprov-0ac08fb134a636"] == "cili"
    assert by_id["nihprov-0f76afeb770feb"] == "(h)iŋgàn"
    assert by_id["nihprov-8ec40874e4ae28"] == "ghaʈa(w)"


def test_report_reconciles_strict_and_sensitivity_denominators():
    report = (ANALYSIS / "REPORT.md").read_text(encoding="utf-8")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    strict_residue = summary["lexeme_strata"]["Nihali residue"]
    clearer_variants = summary["global_variant_sensitivity_assessments"]["variant"]
    qualified = summary["global_variant_sensitivity_assessments"]["qualified"]
    assert f"**{summary['records']:,} attested Nihali database records**" in report
    assert f"{strict_residue:,} (34.6%) remain in the Nihali residue" in report
    assert (
        f"reduce the strict residue from {strict_residue:,} to "
        f"{strict_residue - clearer_variants:,} clusters"
    ) in report
    assert (
        f"including qualified cases gives a broad sensitivity floor of "
        f"{strict_residue - clearer_variants - qualified:,}"
    ) in report
    assert (
        f"It yields {summary['core_residue_root_hypotheses']} provisional root hypotheses "
        f"across {summary['core_residue_root_concepts']} concepts"
    ) in report
    assert "| Nihali residue | 674 | 267 | 249 | 47 | 12 | 116 |" in report


def test_core_vocabulary_slice_is_fixed_and_variant_sensitive():
    core = dicts(ANALYSIS / "nihali-core-vocabulary-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(core) == summary["core_audit_rows"] == 292
    assert summary["core_mapping_exclusions"] == 14
    assert summary["core_concepts_covered"] == 91
    assert Counter(row["Stratum"] for row in core) == Counter(summary["core_strata"])
    assert summary["core_strata"]["Nihali residue"] == 157
    assert all(row["Concepts"] and row["Lexeme_ID"].startswith("nilex-") for row in core)
    sensitivity = [row for row in core if row["Sensitivity_Stratum"]]
    assert len(sensitivity) == summary["core_variant_sensitivity_rows"] == 66
    assert summary["core_sensitivity_strata"]["Nihali residue"] == 91
    assert all(
        row["Stratum"] == "Nihali residue"
        and row["Sensitivity_Stratum"] != "Nihali residue"
        and row["Sensitivity_Reference_Lexeme_ID"].startswith("nilex-")
        and row["Sensitivity_Rationale"]
        for row in sensitivity
    )
    reviewed_cross_concept = {
        row["Lexeme_ID"]: row["Sensitivity_Reference_Lexeme_ID"]
        for row in sensitivity
        if row["Lexeme_ID"] in {
            "nilex-41ddde15aa40aa", "nilex-64a1bea96cb527",
            "nilex-70d62a9bf460dc", "nilex-fd12572b3fbfe1",
        }
    }
    assert reviewed_cross_concept == {
        "nilex-41ddde15aa40aa": "nilex-72e0dc18e79d31",
        "nilex-64a1bea96cb527": "nilex-ef6408ce97d724",
        "nilex-70d62a9bf460dc": "nilex-716f18e4d72e4e",
        "nilex-fd12572b3fbfe1": "nilex-937fc03150af6f",
    }


def test_whole_lexicon_variant_sensitivity_is_exhaustively_reviewed():
    rows = dicts(ANALYSIS / "nihali-global-variant-sensitivity-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(rows) == summary["global_variant_sensitivity_rows"] == 205
    assert len({row["Lexeme_ID"] for row in rows}) == 205
    assert Counter(row["Review_Set"] for row in rows) == {
        "core-basic-vocabulary": 66,
        "global-full-lexicon": 139,
    }
    assert Counter(row["Assessment"] for row in rows) == Counter(
        summary["global_variant_sensitivity_assessments"]
    ) == Counter({"variant": 172, "qualified": 32, "reject": 1})
    assert all(
        row["Reference_Stratum"] not in {"Nihali residue", "Other", ""}
        and row["Reference_Lexeme_ID"].startswith("nilex-")
        and row["Rationale"]
        and "does not alter" in row["Interpretation"]
        for row in rows
    )


def test_residue_expressions_with_contact_components_are_not_counted_as_pure_roots():
    rows = dicts(ANALYSIS / "nihali-residue-contact-component-review.csv")
    lexemes = {
        row["Lexeme_ID"]: row
        for row in dicts(ANALYSIS / "nihali-provisional-lexeme-audit.csv")
    }
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(rows) == summary["residue_contact_component_review_rows"] == 23
    assert len({row["Lexeme_ID"] for row in rows}) == 23
    assert Counter(row["Assessment"] for row in rows) == Counter(
        summary["residue_contact_component_assessments"]
    ) == Counter({
        "transparent-component": 18,
        "qualified-component": 4,
        "source-gloss-conflict": 1,
    })
    assert all(
        lexemes[row["Lexeme_ID"]]["Stratum"] == "Nihali residue"
        and row["Contact_Component"] and row["Layer"]
        and row["Confidence"] in {"high", "medium", "low"}
        and row["Rationale"]
        for row in rows
    )
    conflict = [row for row in rows if row["Assessment"] == "source-gloss-conflict"]
    assert [(row["Form"], row["Gloss"]) for row in conflict] == [
        ("katʰarnāk", "beautiful")
    ]


def test_core_concept_profile_counts_each_basic_meaning_once():
    rows = dicts(ANALYSIS / "nihali-core-concept-origin-profile.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(rows) == summary["core_concept_profile_rows"] == 91
    assert len({row["Concept"] for row in rows}) == 91
    assert Counter(row["Profile_Class"] for row in rows) == Counter(
        summary["core_concept_profile_classes"]
        ) == Counter({
            "contact-only-mixed-family": 31, "residue-plus-contact": 22,
        "residue-only": 17, "contact-only-single-family": 21,
    })
    family_counts = {
        family: sum(family in row["Contact_Families"].split("+") for row in rows)
        for family in ("Korku", "Munda", "Indo-Aryan", "Dravidian")
    }
    assert family_counts == summary["core_concept_family_involvement"] == {
        "Korku": 46, "Munda": 20, "Indo-Aryan": 42, "Dravidian": 13,
    }
    assert sum(row["Residue_Present"] == "yes" for row in rows) == 39
    assert sum(int(row["Residual_Root_Hypotheses"]) for row in rows) == 56
    residue_only = [row for row in rows if row["Profile_Class"] == "residue-only"]
    assert Counter(row["Best_Residual_Replication"] for row in residue_only) == Counter({
        "very strong": 12, "strong": 3, "single-source": 1, "moderate": 1,
    })
    assert sum(row["Any_Multi_Source_Residual_Root"] == "yes" for row in residue_only) == 16
    assert sum(row["Early_Residual_Root"] == "yes" for row in residue_only) == 14
    assert {
        row["Concept"] for row in residue_only
        if row["Any_Multi_Source_Residual_Root"] == "no"
    } == {"WALK"}
    mixed = [row for row in rows if row["Profile_Class"] == "residue-plus-contact"]
    assert sum(row["Any_Multi_Source_Residual_Root"] == "yes" for row in mixed) == 11
    assert all("counts the concept once" in row["Interpretation"] for row in rows)


def test_effective_core_residue_is_collapsed_to_reviewed_root_hypotheses():
    roots = dicts(ANALYSIS / "nihali-core-residue-root-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(roots) == summary["core_residue_root_concepts"] == 39
    assert sum(int(row["Cluster_Count"]) for row in roots) == 91
    assert sum(int(row["Root_Group_Count"]) for row in roots) == (
        summary["core_residue_root_hypotheses"]
    ) == 56
    assert sum(int(row["Cluster_Count"]) > 1 for row in roots) == (
        summary["core_residue_multi_cluster_reviews"]
    ) == 23
    assert all(
        1 <= int(row["Root_Group_Count"]) <= int(row["Cluster_Count"])
        and row["Groups"] and row["Rationale"]
        for row in roots
    )

    inventory = dicts(ANALYSIS / "nihali-core-residue-root-inventory.csv")
    assert len(inventory) == 56
    assert len({row["Root_ID"] for row in inventory}) == 56
    assert Counter(row["Replication_Grade"] for row in inventory) == Counter(
        summary["core_residue_root_replication"]
    ) == Counter({"single-source": 27, "very strong": 16, "strong": 9, "moderate": 4})
    assert sum(row["Early_Source_Attested"] == "yes" for row in inventory) == (
        summary["core_residue_root_early_attested"]
    ) == 33
    assert all(
        row["Representative_Form"] and row["Forms"] and row["Glosses"]
        and row["Grouping_Rationale"] and "not evidence" in row["Interpretation"]
        for row in inventory
    )


def test_closed_class_diagnostic_is_complete_and_explicitly_non_reconstructive():
    rows = dicts(ANALYSIS / "nihali-closed-class-diagnostic-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(rows) == summary["closed_class_rows"] == 97
    assert len({(row["Concept"], row["Lexeme_ID"]) for row in rows}) == 97
    assert Counter(row["Effective_Stratum"] for row in rows) == Counter(
        summary["closed_class_effective_strata"]
    )
    assert Counter(row["Effective_Stratum"] for row in rows if row["Domain"] == "pronoun") == (
        Counter({
            "Nihali residue": 17, "Korku+Munda": 2, "Korku": 1,
            "Other": 1, "Dravidian": 1,
        })
    )
    assert Counter(
        row["Effective_Stratum"] for row in rows if row["Domain"] == "low numeral"
    )["Nihali residue"] == 1
    assert all("not thereby evidence" in row["Interpretation"] for row in rows)


def test_cross_source_residue_has_a_complete_stability_register():
    replicated = dicts(ANALYSIS / "nihali-replicated-residue-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(replicated) == summary["replicated_residue_rows"] == 140
    assert Counter(row["Replication_Grade"] for row in replicated) == Counter(
        summary["replicated_residue_grades"]
    ) == Counter({"moderate": 100, "strong": 28, "very strong": 12})
    assert sum(row["Core_Effective_Residue"] == "yes" for row in replicated) == (
        summary["replicated_effective_core_residue"]
    ) == 25
    assert all(
        int(row["Source_Count"]) >= 2 and row["Lexical_Sources"]
        and row["Interpretation"] and 0 <= float(row["Minimum_Form_Similarity"]) <= 1
        for row in replicated
    )


def test_layer_replication_is_documentary_not_chronological():
    rows = dicts(ANALYSIS / "nihali-contact-layer-replication-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert [row["Layer"] for row in rows] == [
        "Nihali residue", "Korku", "Munda", "Indo-Aryan", "Dravidian",
    ]
    assert {
        row["Layer"]: (int(row["Total_Clusters"]), int(row["Multi_Source_Clusters"]))
        for row in rows
    } == {
            "Nihali residue": (1133, 140), "Korku": (982, 323),
            "Munda": (153, 51), "Indo-Aryan": (1306, 348), "Dravidian": (143, 43),
    }
    assert all("not the age" in row["Interpretation"] for row in rows)
    assert set(summary["layer_replication"]) == {row["Layer"] for row in rows}


def test_source_normalized_profile_controls_dictionary_size_without_claiming_chronology():
    rows = dicts(ANALYSIS / "nihali-source-normalized-profile.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    by_source = {row["Lexical_Source"]: row for row in rows}
    expected = {
        "bhattacharya1957": (396, 250, 146, 133, 38, 149, 24),
        "konow1906": (183, 62, 121, 30, 3, 33, 11),
        "mundlay1996": (1687, 1075, 612, 436, 86, 751, 39),
        "nagaraja2014": (1735, 1388, 347, 770, 101, 742, 124),
        "varghesekumar2015noira": (220, 114, 106, 47, 11, 76, 17),
    }
    assert set(by_source) == set(expected) == set(summary["source_normalized_profile"])
    for source, values in expected.items():
        row = by_source[source]
        assert tuple(int(row[field]) for field in (
            "Lexeme_Clusters", "External_Clusters", "Residue_Clusters", "Korku_Clusters",
            "Munda_Clusters", "Indo_Aryan_Clusters", "Dravidian_Clusters",
        )) == values
        assert abs(float(row["External_Share"]) + float(row["Residue_Share"]) - 1) < .002
        assert "not chronological loan rates" in row["Interpretation"]


def test_cross_source_attribution_agreement_separates_silence_from_conflict():
    rows = dicts(ANALYSIS / "nihali-cross-source-attribution-agreement.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(rows) == summary["cross_source_agreement_rows"] == 701
    assert len({row["Lexeme_ID"] for row in rows}) == 701
    assert Counter(row["Agreement_Class"] for row in rows) == Counter(
        summary["cross_source_agreement_classes"]
    ) == Counter({
        "single-labelled-source": 298, "no-direct-label": 146,
        "multi-source-exact-agreement": 119,
        "multi-source-nested-compatible": 90, "multi-source-disjoint": 48,
    })
    assert all(int(row["Source_Count"]) >= 2 for row in rows)
    assert all(
        "Silence is not disagreement" in row["Interpretation"]
        and "agreement is not necessarily independent" in row["Interpretation"]
        for row in rows
    )
    assert all(
        int(row["Labelled_Source_Count"]) == 0 and not row["Labels_By_Source"]
        for row in rows if row["Agreement_Class"] == "no-direct-label"
    )


def test_family_attribution_replication_is_not_confused_with_lexeme_replication():
    rows = dicts(ANALYSIS / "nihali-family-attribution-replication.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    by_family = {row["Family"]: row for row in rows}
    expected = {
        "Korku": (982, 323, 0, 265, 58),
        "Munda": (153, 51, 0, 46, 5),
        "Indo-Aryan": (1306, 348, 3, 183, 162),
        "Dravidian": (143, 43, 3, 35, 5),
    }
    assert set(by_family) == set(expected) == set(summary["family_attribution_replication"])
    for family, values in expected.items():
        row = by_family[family]
        assert tuple(int(row[field]) for field in (
            "All_Labelled_Clusters", "Multi_Source_Clusters", "No_Direct_Family_Label",
            "One_Direct_Labelled_Source", "Two_Plus_Direct_Labelled_Sources",
        )) == values
        assert sum(int(row[field]) for field in (
            "No_Direct_Family_Label", "One_Direct_Labelled_Source",
            "Two_Plus_Direct_Labelled_Sources",
        )) == int(row["Multi_Source_Clusters"])
        assert "not proof of inheritance" in row["Interpretation"]


def test_structural_sensitivity_is_explicitly_non_genealogical():
    rows = dicts(ANALYSIS / "nihali-grambank-structural-sensitivity.csv")
    assert len(rows) == 13
    assert {row["Glottocode"] for row in rows} >= {
        "kork1243", "mund1320", "sant1410", "mara1378", "hind1269", "kusu1250",
    }
    assert {row["Grambank_Commit"] for row in rows} == {
        "37f73da55cf8b426c82383f46a972bc59ce6cf76"
    }
    assert all(
        0 <= float(row[metric]) <= 1
        for row in rows
        for metric in ("Raw_Agreement", "Cohens_Kappa", "Positive_Feature_Jaccard")
    )
    assert all("not a genealogical test" in row["Interpretation"] for row in rows)


def test_layer_category_profile_is_nonexclusive_and_cautious():
    rows = dicts(ANALYSIS / "nihali-layer-category-profile.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    by_layer = {row["Layer"]: row for row in rows}
    assert len(rows) == 5
    assert (int(by_layer["Nihali residue"]["Noun_Clusters"]),
            int(by_layer["Nihali residue"]["Verb_Clusters"])) == (267, 249)
    assert (int(by_layer["Indo-Aryan"]["Noun_Clusters"]),
            int(by_layer["Indo-Aryan"]["Verb_Clusters"])) == (533, 180)
    assert (int(by_layer["Korku"]["Noun_Clusters"]),
            int(by_layer["Korku"]["Verb_Clusters"])) == (391, 198)
    assert set(summary["layer_category_profile"]) == set(by_layer)
    assert all("cannot diagnose inheritance" in row["Interpretation"] for row in rows)


def test_layer_form_shape_profile_is_descriptive_not_a_family_classifier():
    rows = dicts(ANALYSIS / "nihali-layer-form-shape-profile.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    by_layer = {row["Layer"]: row for row in rows}
    expected = {
        "Nihali residue": (1133, 849, 296, 260, 182, 26),
        "Korku": (982, 710, 242, 243, 244, 34),
        "Munda": (153, 104, 36, 32, 33, 6),
        "Indo-Aryan": (1306, 930, 243, 267, 336, 34),
        "Dravidian": (143, 108, 48, 40, 18, 2),
    }
    assert set(by_layer) == set(expected) == set(summary["layer_form_shape_profile"])
    for layer, values in expected.items():
        row = by_layer[layer]
        assert tuple(int(row[field]) for field in (
            "Total_Clusters", "Final_Vowel_Count", "Multiword_Or_Compound_Count",
            "Retroflex_Count", "Aspiration_Count", "Nasalization_Count",
        )) == values
        assert float(row["Median_Folded_Length"]) == 5
        assert all(0 <= float(row[field]) <= 1 for field in (
            "Final_Vowel_Share", "Multiword_Or_Compound_Share", "Retroflex_Share",
            "Aspiration_Share", "Nasalization_Share",
        ))
        assert "not a genealogical test" in row["Interpretation"]


def test_resolved_contact_shape_audit_keeps_similarity_descriptive():
    shapes = dicts(ANALYSIS / "nihali-resolved-contact-shape-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(shapes) == summary["resolved_contact_shape_rows"] == 571
    assert Counter(row["Parent_Family"] for row in shapes) == Counter(
        summary["resolved_contact_parent_families"]
    ) == Counter({
        "Indo-Aryan": 422, "Dravidian": 98, "Munda": 33, "Other": 13,
        "English": 5,
    })
    assert Counter(row["Parent_Language"] for row in shapes) == Counter(
        summary["resolved_contact_parent_languages"]
    ) == Counter({
        "Indo-Aryan": 362, "Proto-Dravidian": 98, "Proto-Indo-Iranian": 58,
        "Proto-Kherwarian": 16, "Proto-Munda": 17, "Persian": 13, "English": 5,
        "Hindi-Urdu": 1, "Marathi": 1,
    })
    for family, expected in summary["resolved_contact_match_shapes"].items():
        assert Counter(
            row["Match_Shape"] for row in shapes if row["Parent_Family"] == family
        ) == Counter(expected)
    for family, expected in summary["resolved_contact_surface_match_shapes"].items():
        assert Counter(
            row["Surface_Match_Shape"] for row in shapes
            if row["Parent_Family"] == family and row["Surface_Match_Shape"]
        ) == Counter(expected)
    matched_surfaces = [row for row in shapes if row["Matched_Surface_ID"]]
    # Historical and directly attested parents have an observed surface comparandum.
    # Two modern Hindi/Marathi proxy parents are intentionally not treated as independent
    # surface evidence because their source forms were provisionally added for this analysis.
    assert len(matched_surfaces) == len(shapes) - 2 == 569
    assert {
        (row["Child_Form"], row["Parent_Language_ID"])
        for row in shapes if not row["Matched_Surface_ID"]
    } == {("sagai", "H"), ("vorāri pakkā", "M")}
    beat = next(row for row in shapes if row["Lexeme_ID"] == "nilex-d2f8905043be36")
    assert (beat["Parent_ID"], beat["Parent_Form"], beat["Parent_Gloss"]) == (
        "d2063", "*koṭṭ-", "; to beat",
    )
    thatch = next(row for row in shapes if row["Lexeme_ID"] == "nilex-c5788d18f49d96")
    assert (thatch["Parent_ID"], thatch["Parent_Form"], thatch["Parent_Gloss"]) == (
        "f_fnkvzvcjfvdpm", "*bel", "spread (vt)",
    )
    knee = next(row for row in shapes if row["Lexeme_ID"] == "nilex-dff855af7f4012")
    assert knee["Parent_ID"] == "d2983"
    assert (knee["Matched_Surface_Form"], knee["Matched_Surface_Gloss"]) == (
        "toŋge", "knee",
    )
    corrected = {
        row["Lexeme_ID"]: row["Parent_ID"] for row in shapes
        if row["Lexeme_ID"] in {
            "nilex-009647aa35896e", "nilex-2bad99199c9a89", "nilex-ba5caf8b96f48c",
        }
    }
    assert corrected == {
        "nilex-009647aa35896e": "1302",
        "nilex-2bad99199c9a89": "9330",
        "nilex-ba5caf8b96f48c": "7200",
    }
    broad = next(row for row in shapes if row["Lexeme_ID"] == "nilex-2bad99199c9a89")
    assert (broad["Matched_Surface_Form"], broad["Matched_Surface_Gloss"]) == (
        "baka", "big",
    )
    assert all(
        row["Matched_Surface_Form"] and row["Matched_Surface_Language_ID"]
        and row["Surface_Match_Shape"] in {"exact", "near", "moderate", "distant"}
        and 0 <= float(row["Surface_Form_Similarity"]) <= 1
        and 0 <= float(row["Surface_Gloss_Similarity"]) <= 1
        for row in matched_surfaces
    )
    assert summary["resolved_parent_route_family_involvement"] == {
        "Dravidian": {"Korku": 8, "Munda": 4, "Indo-Aryan": 14, "Dravidian": 90},
        "English": {"Korku": 1, "Munda": 0, "Indo-Aryan": 4, "Dravidian": 0},
        "Indo-Aryan": {"Korku": 112, "Munda": 3, "Indo-Aryan": 421, "Dravidian": 5},
        "Munda": {"Korku": 24, "Munda": 23, "Indo-Aryan": 6, "Dravidian": 1},
        "Other": {"Korku": 0, "Munda": 0, "Indo-Aryan": 13, "Dravidian": 0},
    }
    assert all(
        not row["Parent_ID"].startswith("nihprov-") and row["Parent_Language_ID"] != "Ni"
        and row["Interpretive_Caution"] and row["Match_Shape"] in {
            "exact", "near", "moderate", "distant",
        }
        for row in shapes
    )
    munda = [row for row in shapes if row["Parent_Family"] == "Munda"]
    assert len({row["Parent_ID"] for row in munda}) == summary["munda_resolved_parent_roots"] == 28
    assert Counter(row["Review_Assessment"] for row in munda) == Counter(
        summary["munda_correspondence_review"]
    ) == Counter({
        "possible-correspondence": 15, "near-contact-compatible": 11,
        "weak-comparison": 7,
    })
    assert Counter(row["Correspondence_Series"] for row in munda) == Counter(
        summary["munda_correspondence_series"]
    ) == Counter({"identity-or-near": 11, "other": 10, "none": 7, "c~s": 5})
    assert all(row["Review_Rationale"] and row["Review_Confidence"] for row in munda)


def test_dravidian_correspondence_audit_is_root_level_and_non_genealogical():
    rows = dicts(ANALYSIS / "nihali-dravidian-correspondence-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(rows) == summary["dravidian_correspondence_roots"] == 66
    assert len({row["Parent_ID"] for row in rows}) == 66
    assert sum(int(row["Link_Count"]) for row in rows) == 98
    assert Counter(row["Assessment"] for row in rows) == Counter(
        summary["dravidian_correspondence_assessments"]
    ) == Counter({
        "weak-form-link": 27, "near-contact-compatible": 22,
        "possible-comparison": 12, "weak-semantic-link": 5,
    })
    assert Counter(row["Series_Type"] for row in rows) == Counter(
        summary["dravidian_correspondence_series_types"]
    ) == Counter({
        "identity-initial": 45, "singleton-nonidentity": 14,
        "repeated-nonidentity": 7,
    })
    repeated = Counter(
        row["Initial_Correspondence"] for row in rows
        if row["Series_Type"] == "repeated-nonidentity"
    )
    assert repeated == Counter({"b~v": 3, "g~k": 2, "h~k": 2})
    assert all(
        row["Best_Surface_ID"] and row["Best_Surface_Form"]
        and 0 <= float(row["Best_Surface_Form_Similarity"]) <= 1
        and 0 <= float(row["Best_Surface_Gloss_Similarity"]) <= 1
        and "not a sound law or evidence of genetic inheritance" in row["Interpretation"]
        for row in rows
    )


def test_external_contact_clusters_have_nonexclusive_evidence_tiers():
    tiers = dicts(ANALYSIS / "nihali-contact-evidence-tier-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(tiers) == summary["contact_evidence_tier_rows"] == 2143
    assert Counter(row["Evidence_Tier"] for row in tiers) == Counter(
        summary["contact_evidence_tiers"]
    ) == Counter({
        "explicit-unqualified-proxy": 1245, "resolved-parent": 557,
        "label-only-unqualified-proxy": 65,
        "manually-qualified-proxy": 121, "manually-corroborated-proxy": 70,
        "manually-weak-or-unresolved": 62,
        "internal-variant-explicit-source": 14,
        "internal-variant-label-only-source": 2,
        "internal-variant-questioned-source": 7,
    })
    assert "questioned-unreviewed-proxy" not in summary["contact_evidence_tiers"]
    assert all(row["Families"] and row["Tier_Basis"] for row in tiers)

    brackets = dicts(ANALYSIS / "nihali-family-evidence-bracket-audit.csv")
    assert {row["Family"] for row in brackets} == {
        "Indo-Aryan", "Korku", "Munda", "Dravidian", "English",
    }
    assert {
        row["Family"]: (
            int(row["High_Specificity_Floor"]), int(row["Supported_Envelope"]),
            int(row["All_Labelled_Clusters"]),
        )
        for row in brackets
    } == {
        "Indo-Aryan": (452, 1222, 1306), "Korku": (30, 903, 982),
        "Munda": (8, 121, 153), "Dravidian": (96, 126, 143),
        "English": (5, 16, 18),
    }
    assert all("not a confidence interval" in row["Definition"] for row in brackets)
    family_tiers = dicts(ANALYSIS / "nihali-family-contact-evidence-audit.csv")
    assert len(family_tiers) == summary["family_contact_evidence_rows"] == 2602
    assert len({(row["Lexeme_ID"], row["Family"]) for row in family_tiers}) == 2602
    assert Counter(row["Family"] for row in family_tiers) == Counter({
        "Indo-Aryan": 1306, "Korku": 982, "Munda": 153,
        "Dravidian": 143, "English": 18,
    })
    for family, expected in summary["contact_family_by_tier"].items():
        assert Counter(
            row["Evidence_Tier"] for row in family_tiers if row["Family"] == family
        ) == Counter(expected)
    assert all("not genetic inheritance" in row["Interpretation"] for row in family_tiers)


def test_source_label_variation_separates_nested_routes_from_disjoint_claims():
    variation = dicts(ANALYSIS / "nihali-source-label-variation-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    counts = Counter(row["Relationship"] for row in variation)
    assert len(variation) == summary["source_variation_rows"] == 138
    assert counts == Counter({"nested-route/ultimate": 93, "disjoint": 45})
    assert counts == Counter(summary["source_variation_relationships"])
    assert all(row["Source_Evidence"] and row["Own_Source_Attributions"] for row in variation)
    assert all(
        row["Review_Priority"] == ("high" if row["Relationship"] == "disjoint" else "medium")
        for row in variation
    )
    disjoint = [row for row in variation if row["Relationship"] == "disjoint"]
    assessments = Counter(row["Review_Assessment"] for row in disjoint)
    assert assessments == Counter(summary["disjoint_source_review"])
    assert assessments == Counter({
        "compatible-contact-chain": 26, "favor-indo-aryan": 8,
        "unresolved": 6, "favor-korku": 4, "favor-munda": 1,
    })
    assert all(row["Review_Rationale"] and row["Review_Confidence"] for row in disjoint)


def test_unresolved_source_proxies_are_graded_by_reproducibility_and_doubt():
    quality = dicts(ANALYSIS / "nihali-source-proxy-quality-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(quality) == summary["source_proxy_quality_rows"]
    assert Counter(row["Evidence_Quality"] for row in quality) == Counter(
        summary["source_proxy_quality"]
    )
    assert Counter(row["Uncertainty"] for row in quality) == Counter(
        summary["source_proxy_uncertainty"]
    )
    assert Counter(row["Comparison_Shape"] for row in quality) == Counter(
        summary["source_proxy_comparison_shapes"]
    ) == Counter({
        "exact": 389, "near": 385, "moderate": 386,
            "distant": 322, "unscored": 147,
    })
    assert {row["Evidence_Quality"] for row in quality} == {
        "catalog-indexed", "explicit-comparanda", "donor-label-only",
    }
    assert {row["Uncertainty"] for row in quality} == {"questioned", "unqualified"}
    assert {row["Review_Priority"] for row in quality} == {
        "critical", "high", "medium", "low",
    }
    assert all(
        row["Lexeme_ID"].startswith("nilex-") and row["Stratum"]
        and row["Source_Evidence"] and int(row["Proxy_Record_Count"]) >= 1
        and row["Comparison_Shape"] in {"exact", "near", "moderate", "distant", "unscored"}
        for row in quality
    )
    critical = [row for row in quality if row["Review_Priority"] == "critical"]
    assert len(critical) == 20
    assert Counter(row["Review_Assessment"] for row in critical) == Counter(
        summary["core_source_proxy_review"]
    ) == Counter({
        "corroborated-contact": 8, "route-ambiguous-contact": 5,
        "unresolved": 4, "plausible-contact": 3,
    })
    assert all(row["Review_Rationale"] and row["Review_Confidence"] for row in critical)
    high = [row for row in quality if row["Review_Priority"] == "high"]
    assert len(high) == 44
    assert Counter(row["Review_Assessment"] for row in high) == Counter(
        summary["diagnostic_source_proxy_review"]
    ) == Counter({
        "plausible-contact": 15, "route-ambiguous-contact": 11,
        "corroborated-contact": 10, "weak-comparison": 5, "unresolved": 3,
    })
    assert all(row["Review_Rationale"] and row["Review_Confidence"] for row in high)
    questioned_review = dicts(ANALYSIS / "nihali-questioned-source-proxy-review.csv")
    assert len(questioned_review) == 200
    assert Counter(row["Review_Assessment"] for row in questioned_review) == Counter(
        summary["questioned_source_proxy_review"]
    ) == Counter({
        "plausible-contact": 57, "corroborated-contact": 54,
        "route-ambiguous-contact": 38, "weak-comparison": 30, "unresolved": 21,
    })
    assert all(
        row["Uncertainty"] == "questioned"
        and row["Review_Assessment"] and row["Review_Rationale"] and row["Review_Confidence"]
        for row in questioned_review
    )
    assert all(
        row["Review_Assessment"] and row["Review_Rationale"]
        for row in quality if row["Uncertainty"] == "questioned"
    )
    mixed_proxy = next(
        row for row in dicts(ANALYSIS / "nihali-provisional-etymology-audit.csv")
        if row["Lexeme_ID"] == "nilex-628503046f43ca"
    )
    assert mixed_proxy["Parent_Form"] == "Dravidian?/Indo-Aryan?; cf. Korku carmuru"
    assert mixed_proxy["Method"] == "source-proxy"
    assert mixed_proxy["Manual_Decision"] == "reject"


def test_korku_route_recovery_excludes_circular_proxies_and_separates_upstream_origin():
    rows = dicts(ANALYSIS / "nihali-korku-route-audit.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(rows) == summary["korku_route_rows"] == 854
    assert len({row["Lexeme_ID"] for row in rows}) == 854
    assert Counter(row["Route_Assessment"] for row in rows) == Counter(
        summary["korku_route_assessments"]
    ) == Counter({
        "form-only-match": 405, "weak-or-unmatched": 297,
        "strong-route-match": 86, "no-recoverable-comparandum": 38,
        "possible-route-match": 28,
    })
    strong = [row for row in rows if row["Route_Assessment"] == "strong-route-match"]
    assert Counter(row["Ultimate_Parent_Family"] or "unresolved-korku" for row in strong) == (
        Counter(summary["korku_strong_route_ultimate_families"])
    ) == Counter({"unresolved-korku": 78, "Munda": 8})
    assert all("nihali-provisional2026" not in row["Matched_Korku_Source"] for row in rows)
    assert all(
        "not evidence that the form originated in Korku" in row["Interpretation"]
        for row in rows
    )
    assert all(
        row["Matched_Korku_ID"] and 0 <= float(row["Form_Similarity"]) <= 1
        and 0 <= float(row["Gloss_Similarity"]) <= 1
        for row in rows if row["Route_Assessment"] != "no-recoverable-comparandum"
    )
    core = [row for row in rows if row["Core_Vocabulary"] == "yes"]
    assert len(core) == 54
    assert Counter(row["Route_Assessment"] for row in core) == Counter(
        summary["core_korku_route_assessments"]
    ) == Counter({
        "form-only-match": 18, "strong-route-match": 14,
        "weak-or-unmatched": 9, "no-recoverable-comparandum": 7,
        "possible-route-match": 6,
    })


def test_indo_aryan_route_profile_separates_named_comparanda_from_borrowing_dates():
    rows = dicts(ANALYSIS / "nihali-indo-aryan-route-profile.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(rows) == summary["indo_aryan_route_rows"] == 1306
    assert len({row["Lexeme_ID"] for row in rows}) == 1306
    assert Counter(row["Period_Evidence_Class"] for row in rows) == Counter(
        summary["indo_aryan_period_evidence_classes"]
    ) == Counter({
        "modern-ia-explicit": 1032, "generic-ia-explicit": 127,
        "resolved-no-specific-source-language": 76, "sanskrit-explicit": 33,
        "historical-and-modern-explicit": 24, "ia-label-no-specific-language": 14,
    })
    assert sum(row["Korku_Route_Mentioned"] == "yes" for row in rows) == (
        summary["indo_aryan_korku_route_clusters"]
    ) == 330
    assert all(
        "does not date borrowing" in row["Interpretation"]
        and "does not automatically establish direction" in row["Interpretation"]
        for row in rows
    )
    assert all(
        row["IA_Source_Language_IDs"]
        for row in rows if row["Period_Evidence_Class"] in {
            "modern-ia-explicit", "generic-ia-explicit", "sanskrit-explicit",
            "historical-and-modern-explicit",
        }
    )


def test_origin_evidence_matrix_makes_the_diagnosis_and_limits_explicit():
    rows = dicts(ANALYSIS / "nihali-origin-evidence-matrix.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assert len(rows) == summary["origin_evidence_matrix_rows"] == 17
    assert [row["Evidence_ID"] for row in rows] == [f"E{index:02d}" for index in range(1, 18)]
    assert all(
        row["Domain"] and row["Finding"] and row["Supports_Hypothesis"]
        and row["Challenges_Hypothesis"] and row["Evidential_Weight"]
        and row["Limitation"] and row["Audit_Or_Source"]
        for row in rows
    )
    synthesis = rows[-1]
    assert synthesis["Supports_Hypothesis"] == "independent lineage with layered relexification"
    assert synthesis["Evidential_Weight"] == "moderate overall, diagnosis by exclusion"
    assert "unknown deep relationship remains possible" in synthesis["Limitation"].lower()
    munda = next(row for row in rows if row["Evidence_ID"] == "E04")
    assert "floor of 8 among 153" in munda["Finding"] and "only 5 of 51" in munda["Finding"]
    variants = next(row for row in rows if row["Evidence_ID"] == "E15")
    assert "205 residual/contact cluster pairs" in variants["Finding"]
    assert "172 are plausible variants" in variants["Finding"]
    components = next(row for row in rows if row["Evidence_ID"] == "E16")
    assert "23 residual expressions" in components["Finding"]
    assert "18 transparent" in components["Finding"]
    assert "entire residue count as inherited root evidence" in components["Challenges_Hypothesis"]


def test_residue_score_sensitivity_does_not_bypass_component_gates():
    rows = dicts(ANALYSIS / "nihali-residue-threshold-sensitivity.csv")
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    expected = {
        "very loose": (479, 256, 190, 32, 449),
        "loose": (334, 178, 133, 22, 304),
        "moderate": (171, 86, 68, 16, 141),
        "high score, low margin": (70, 38, 22, 9, 40),
        "production-like score/margin only": (65, 36, 22, 6, 39),
        "very high score": (16, 9, 5, 1, 0),
    }
    by_label = {row["Threshold_Label"]: row for row in rows}
    assert set(by_label) == set(expected) == set(summary["residue_threshold_sensitivity"])
    for label, values in expected.items():
        row = by_label[label]
        assert tuple(int(row[field]) for field in (
            "Flagged_Residue_Clusters", "Indo_Aryan_Candidates",
            "Dravidian_Candidates", "Munda_Candidates", "Unreviewed",
        )) == values
        assert "not assignments" in row["Interpretation"]
        assert "unequal donor-database coverage" in row["Interpretation"]
    assert int(by_label["very high score"]["Manual_Rejected"]) == 8
    assert int(by_label["very high score"]["Manual_Deferred"]) == 8


def test_generated_overlay_is_complete_and_explicitly_provisional():
    summary = json.loads((ANALYSIS / "nihali-provisional-etymology-summary.json").read_text())
    assignments = [
        row for row in read_assignments()
        if row["Notes"].startswith(MARKER)
    ]
    params = list(csv.reader(
        (ROOT / "data/other/params/20260901-nihali-provisional.csv").open(
            encoding="utf-8", newline=""
        )
    ))
    assert summary["records"] == 4299
    assert summary["existing_rank1"] == 217
    assert summary["lexeme_clusters"] == summary["cluster_audit_rows"]
    assert "computational-resolved" not in summary["methods"]
    assert len(assignments) == summary["generated_assignments"] == 4082
    assert len(params) == summary["proxy_entries"]
    assert all(row[2] in {"borrowed", "reflex"} and row[3] == "1" for row in [
        [item["Form_ID"], item["Etymon_ID"], item["Kind"], item["Rank"]]
        for item in assignments
    ])


def test_built_graph_has_a_rank_one_hypothesis_for_every_target():
    forms = dicts(ROOT / "cldf/forms.csv")
    target_ids = {
        row["ID"] for row in forms
        if is_study_attestation(row)
    }
    edges = dicts(ROOT / "cldf/edges.csv")
    rank1 = Counter(
        row["Child_ID"] for row in edges
        if row["Rank"] == "1" and row["Kind"] in {"reflex", "borrowed", "variant"}
        and row["Child_ID"] in target_ids
    )
    assert set(rank1) == target_ids
    assert set(rank1.values()) == {1}
