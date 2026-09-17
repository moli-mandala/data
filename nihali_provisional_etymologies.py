#!/usr/bin/env python3
"""Build a complete, explicitly provisional etymology overlay for Nihali.

The four reviewed Nihali dictionaries and the Jamod survey list contain many repeated
attestations and many prose comparisons which could not be reduced to a CDIAL/DEDR number by
their importers.  This script treats the *attested record* as the unit of coverage while sharing
one hypothesis node across sufficiently similar records.

Evidence is applied in this order:

1. retain an existing accepted Jambu edge;
2. resolve a printed/editorial donor attribution against a compatible Jambu etymon;
3. retain an unresolved printed/editorial attribution as a clearly marked donor proxy;
4. seek a conservative form-and-gloss match in the contact families represented in Jambu;
5. create a source-local Nihali-residue lexical grouping when no external candidate clears the
   threshold.

The last two layers are editorial hypotheses, not claims made by the lexical source.  Every row
and every proxy says so in its evidence/note.  Run from the data repository root:

    uv run python nihali_provisional_etymologies.py --install

Without ``--install`` the artifacts are written below ``tmp/nihali-provisional``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


import etymology_assignments as overlay

ROOT = Path(__file__).resolve().parent
FORMS = ROOT / "cldf/forms.csv"
EDGES = ROOT / "cldf/edges.csv"
LANGUAGES = ROOT / "cldf/languages.csv"
CONCEPTS = ROOT / "cldf/concepts.csv"
FORM_CONCEPTS = ROOT / "cldf/form_concepts.csv"
# Curated etymology rows live in per-source sidecars; see etymology_assignments.py.
ETYMOLOGIES = ROOT / "data/etymologies.csv"
PARAMS_NAME = "20260901-nihali-provisional.csv"
AUDIT_NAME = "nihali-provisional-etymology-audit.csv"
CLUSTER_AUDIT_NAME = "nihali-provisional-lexeme-audit.csv"
CORE_AUDIT_NAME = "nihali-core-vocabulary-audit.csv"
GLOBAL_VARIANT_SENSITIVITY_AUDIT_NAME = "nihali-global-variant-sensitivity-audit.csv"
CORE_CONCEPT_PROFILE_AUDIT_NAME = "nihali-core-concept-origin-profile.csv"
CORE_RESIDUE_ROOT_AUDIT_NAME = "nihali-core-residue-root-audit.csv"
CORE_RESIDUE_ROOT_INVENTORY_NAME = "nihali-core-residue-root-inventory.csv"
REPLICATED_RESIDUE_AUDIT_NAME = "nihali-replicated-residue-audit.csv"
RESOLVED_CONTACT_SHAPE_AUDIT_NAME = "nihali-resolved-contact-shape-audit.csv"
DRAVIDIAN_CORRESPONDENCE_AUDIT_NAME = "nihali-dravidian-correspondence-audit.csv"
CONTACT_EVIDENCE_TIER_AUDIT_NAME = "nihali-contact-evidence-tier-audit.csv"
FAMILY_EVIDENCE_BRACKET_AUDIT_NAME = "nihali-family-evidence-bracket-audit.csv"
FAMILY_CONTACT_EVIDENCE_AUDIT_NAME = "nihali-family-contact-evidence-audit.csv"
SOURCE_VARIATION_AUDIT_NAME = "nihali-source-label-variation-audit.csv"
SOURCE_PROXY_QUALITY_AUDIT_NAME = "nihali-source-proxy-quality-audit.csv"
KORKU_ROUTE_AUDIT_NAME = "nihali-korku-route-audit.csv"
INDO_ARYAN_ROUTE_AUDIT_NAME = "nihali-indo-aryan-route-profile.csv"
QUESTIONED_SOURCE_PROXY_REVIEW_NAME = "nihali-questioned-source-proxy-review.csv"
LAYER_REPLICATION_AUDIT_NAME = "nihali-contact-layer-replication-audit.csv"
SOURCE_PROFILE_AUDIT_NAME = "nihali-source-normalized-profile.csv"
CROSS_SOURCE_AGREEMENT_AUDIT_NAME = "nihali-cross-source-attribution-agreement.csv"
FAMILY_ATTRIBUTION_REPLICATION_AUDIT_NAME = "nihali-family-attribution-replication.csv"
LAYER_CATEGORY_AUDIT_NAME = "nihali-layer-category-profile.csv"
LAYER_FORM_SHAPE_AUDIT_NAME = "nihali-layer-form-shape-profile.csv"
CLOSED_CLASS_AUDIT_NAME = "nihali-closed-class-diagnostic-audit.csv"
SUMMARY_NAME = "nihali-provisional-etymology-summary.json"
REPORT_NAME = "REPORT.md"
ORIGIN_EVIDENCE_MATRIX_NAME = "nihali-origin-evidence-matrix.csv"
RESIDUE_THRESHOLD_SENSITIVITY_NAME = "nihali-residue-threshold-sensitivity.csv"
MANUAL_REVIEW = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-computational-candidate-review.csv"
)
LOW_MARGIN_REVIEW = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-low-margin-candidate-review.csv"
)
TRANSPARENT_LOAN_REVIEW = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-transparent-loan-review.csv"
)
SOURCE_PARENT_REVIEW = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-source-parent-review.csv"
)
DISJOINT_SOURCE_REVIEW = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-disjoint-source-review.csv"
)
CORE_EXCLUSIONS = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-core-vocabulary-exclusions.csv"
)
CORE_VARIANT_REVIEW = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-core-variant-sensitivity-review.csv"
)
GLOBAL_VARIANT_REVIEW = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-global-variant-sensitivity-review.csv"
)
RESIDUE_CONTACT_COMPONENT_REVIEW = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-residue-contact-component-review.csv"
)
CORE_SOURCE_PROXY_REVIEW = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-core-source-proxy-review.csv"
)
DIAGNOSTIC_SOURCE_PROXY_REVIEW = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-diagnostic-source-proxy-review.csv"
)
CORE_RESIDUE_ROOT_REVIEW = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-core-residue-root-review.csv"
)
MUNDA_CORRESPONDENCE_REVIEW = (
    ROOT / "data/other/analysis/nihali-provisional/nihali-munda-correspondence-review.csv"
)

# A source-parent adjudication can identify a specific descendant more reliably than generic
# English-gloss similarity (for example, broad ~ big is semantically close but string-distant).
# Keep such exceptions explicit and regression-tested instead of expanding the matching lexicon
# in a way that would silently affect candidate generation throughout the database.
PREFERRED_CONTACT_SURFACE_BY_LEXEME = {
    "nilex-2bad99199c9a89": "f_zzn4opitqpxni",  # Sinhala baka 'big', CDIAL 9330
}

# Some source notes mix a bare family label with a concrete comparison.  The generic parser
# must retain that full uncertainty, but the first token alone can make the app display a
# self-contradictory label (for example, "Dravidian IA?").  Override only the human-facing
# proxy form here; the source attribution, stratum, language bucket, and review decision remain
# unchanged and are all preserved in the audit.
SOURCE_PROXY_FORM_DISPLAY_BY_LEXEME = {
    "nilex-628503046f43ca": "Dravidian?/Indo-Aryan?; cf. Korku carmuru",
}

# Manual disposition of every explicitly hedged donor proxy outside the separately reviewed core
# and Munda/Dravidian diagnostic sets.  These IDs are intentionally exhaustive: the builder fails
# if a newly questioned case appears or an old one disappears without an updated decision.  The
# categories grade the printed comparison; they do not silently change the source-attributed graph.
QUESTIONED_PROXY_REVIEW_IDS = {
    "corroborated-contact": {
        "nilex-05edcabd2cf39a", "nilex-19306a2ff70607", "nilex-1ea4cd4b7eee62",
        "nilex-20d55be6fc9487", "nilex-2429a6ff669fcf", "nilex-28445f27a700e1",
        "nilex-2ed4893a488e33", "nilex-3169692f2ce26a", "nilex-357d88f9a0a049",
        "nilex-4265438cf85372", "nilex-4a8d4b514304c1", "nilex-4ea5d438faa674",
        "nilex-557134836ba104", "nilex-56127f66d8869b", "nilex-59a2eb554e902f",
        "nilex-5eb7568e798df5", "nilex-63a4042242cab5", "nilex-646136fc589668",
        "nilex-65b861377ada97", "nilex-66dee0dcb1bc78", "nilex-6da78e546f5947",
        "nilex-73be228e13fe3e", "nilex-7c545f42bff6b5", "nilex-7e2cf9d496db8f",
        "nilex-823cb9b5f7019a", "nilex-8300f0c32eeabe", "nilex-84ebb74158dbd8",
        "nilex-925b44288f3670", "nilex-936caa98d9de63", "nilex-9504f311e5a091",
        "nilex-9ad864873b581a", "nilex-9b3fd5957f2181", "nilex-a20551f032111a",
        "nilex-a4607217d621bb", "nilex-a903c221ec091e", "nilex-a9b4bd7eff1fb9",
        "nilex-aa4d6374c6547c", "nilex-ab478ce18bf1a0", "nilex-ad78160cd0f41d",
        "nilex-b3c3e0059ea1cf", "nilex-b4f00540d43cc6", "nilex-b7d39b219682e5",
        "nilex-be5d4c0fbc974d",
        "nilex-b851a6df769659", "nilex-b95b3813c1be0a", "nilex-bd545813dff510",
        "nilex-bd68d0b9df9e18", "nilex-bfb7c606270bb8", "nilex-c470acbbbc2f06",
        "nilex-c84b25f877173c", "nilex-d6d52070b92576", "nilex-da29693728bce8",
        "nilex-ecd750db0dc02e", "nilex-fbc19907f52c43",
    },
    "route-ambiguous-contact": {
        "nilex-0d7f4c6281f93b", "nilex-0e83b83f5aa85d", "nilex-0fbe0bdb01c148",
        "nilex-1e56276b45f06a", "nilex-25a001680e9e47", "nilex-3b4860b7897210",
        "nilex-4bbc70c040a989", "nilex-5020ab7addaa02", "nilex-55503e025438f0",
        "nilex-5dbfa544a4e16f", "nilex-5dc016cba99266", "nilex-6fe83c2c3689a3",
        "nilex-78adb33cbd0ec8", "nilex-78c36b11201e2b", "nilex-8c40fffb6e14b6",
        "nilex-8e790cd672f112", "nilex-92f7227126d29e", "nilex-9ad7afe16ae760",
        "nilex-9f7e5e6cfca858", "nilex-a19208acb7c84b", "nilex-a8743a6e275eea",
        "nilex-a9d01ce1e1aca1", "nilex-ab6bee32739c1c", "nilex-ad5e3d57caea0c",
        "nilex-ad65629978f673", "nilex-c067f99f30db2a", "nilex-ca119032b82650",
        "nilex-cb19a4337249e4", "nilex-cd1fa688310317", "nilex-d4a018e38d9b05",
        "nilex-d862fa2b18f5df", "nilex-dd02a6da4c382c", "nilex-e37cf6c7802eac",
        "nilex-e55a9e7bf687ef", "nilex-ee66aec54e65e9", "nilex-f48f12b02ed0b0",
        "nilex-f7d68d79b8f74a", "nilex-fcf07250b0a41b",
    },
    "plausible-contact": {
        "nilex-0413d5e49d4ac5", "nilex-092ab3d69266da", "nilex-0e1d2fcf8c8a8a",
        "nilex-0f6831752486e6", "nilex-12fafc183bad17", "nilex-14361cb5843a7d",
        "nilex-158b5cc2d032b3", "nilex-15eeaaf3bf4193", "nilex-1e23e180eeae78",
        "nilex-1e4bf10c49a525", "nilex-2067b1dd67a904", "nilex-25329961470a7a",
        "nilex-275eaabae8ee07", "nilex-2e3abdb086fcb9", "nilex-33a4160bec913e",
        "nilex-4406d51d5e9ec4", "nilex-49db0650889fb4", "nilex-4be90fee96f069",
        "nilex-511a94f51f6fa3", "nilex-541b9ef4b0744e", "nilex-5891ac528f5a1d",
        "nilex-5dc02f47dff46f", "nilex-5fe96cb7f75f3f", "nilex-62340f267b1016",
        "nilex-62b67dab25028a", "nilex-637af0423ae6ae", "nilex-6ae3653fca3fe4",
        "nilex-6b05f32f2f9323", "nilex-732a55151bac6d", "nilex-7b4f36f883b9a2",
        "nilex-86834a141a001c", "nilex-87e8446b1df391", "nilex-8e94ad21b41075",
        "nilex-91974134a707be", "nilex-a0ae14a4d383b3",
        "nilex-a1cf794675699b", "nilex-a674fbc7f36992", "nilex-a78a23541dc934",
        "nilex-a83545ebbd1968", "nilex-ab11346e8b2a03", "nilex-adb36ebfdba86d",
        "nilex-af7709901d549d", "nilex-b6f4b739fade5a", "nilex-b77966ab464a76",
        "nilex-b90b029df921e8", "nilex-b913ff310a2d93", "nilex-baaf2d5cc5921f",
        "nilex-bf2e16898a389c", "nilex-c29f4645211ef4", "nilex-c2ea7bdcaaa850",
        "nilex-d485b9df6cfb36", "nilex-db2776f94b9e11", "nilex-df606458e57dc4",
        "nilex-e811e1b64d4161", "nilex-e8ebf454cabfac", "nilex-f4ed9927182c58",
        "nilex-ff30d76b6d84f5",
    },
    "weak-comparison": {
        "nilex-0097531527ce71", "nilex-1338d1373ad29c", "nilex-1715405f941050",
        "nilex-17525083800b61", "nilex-179329599c80a4", "nilex-2b86bdb2c5fd47",
        "nilex-2f0480219cb38b", "nilex-3eef26c0a328df", "nilex-48435e67770dea",
        "nilex-4845fec85545a4", "nilex-4ce7cfe1e7dc42", "nilex-62142a5e8193f4",
        "nilex-632d6a25419674", "nilex-7f0be7f38f8011", "nilex-8f5016fa653758",
        "nilex-9ef6680b039b08", "nilex-b41632878a9953", "nilex-b5a4b5077edca2",
        "nilex-b8a4bba92e0982", "nilex-b8fcb5adea29d0", "nilex-c2ef6a415e2396",
        "nilex-c58fc2b4b86164", "nilex-c8aa66e38ef06e", "nilex-d1df0633f303c7",
        "nilex-d55c2010d447f4", "nilex-d67b12f8cbf64b", "nilex-e95bc94f304215",
        "nilex-fcf42d3281b9a9", "nilex-c9926cb626bfd8", "nilex-05b1049435e1c4",
    },
    "unresolved": {
        "nilex-0ecaaea5434f60", "nilex-1c39a4410ae7af", "nilex-3a0bb781f4eeee",
        "nilex-3d244c5273535a", "nilex-3e335aae60d3b6", "nilex-48da016b3f4042",
        "nilex-4c355d8eb5ce62", "nilex-5a4cb8d8068c04", "nilex-7643a0bd4bc9c9",
        "nilex-770fbcaaa18f28", "nilex-806e0ecdfc0fcf", "nilex-8579c4c1bb5fd6",
        "nilex-8c4c14e2f0d955", "nilex-8ca1acb5a46961", "nilex-9941652120b23a",
        "nilex-8e68e203a2bea0",
        "nilex-a8c8d7d2a21bb0", "nilex-ae47850909440e", "nilex-d4e08a0cce24f5",
        "nilex-dff61988ca4015", "nilex-e26aa229ae592e",
    },
}
REFERENCE = "nihali-provisional2026"
ASSIGNMENT_MARKER = "Nihali provisional 2026"

AUDIT_FIELDS = [
    "Form_ID", "Lexeme_ID", "Lexeme_Size", "Form", "Gloss", "Lexical_Source",
    "Form_Source", "Original_Etymology", "Method", "Stratum", "Confidence", "Score",
    "Margin", "Parent_ID", "Parent_Form",
    "Parent_Language_ID", "Parent_Language", "Parent_Clade", "Kind", "Source_Attribution",
    "Own_Source_Attribution", "Attribution_Basis", "Manual_Trigger", "Manual_Decision",
    "Manual_Rationale", "Evidence", "Alternatives",
]
CLUSTER_AUDIT_FIELDS = [
    "Lexeme_ID", "Representative_Form", "Representative_Gloss", "Record_Count",
    "Source_Count", "Lexical_Sources", "Forms", "Glosses", "Stratum",
    "Own_Source_Attributions", "Methods", "Parent_IDs", "Confidence", "Manual_Decisions",
    "Manual_Triggers", "Manual_Rationales", "Review_Flags",
]
CORE_AUDIT_FIELDS = [
    "Lexeme_ID", "Concepts", "Representative_Form", "Representative_Gloss",
    "Record_Count", "Stratum", "Methods", "Confidence", "Lexical_Sources",
    "Sensitivity_Stratum", "Sensitivity_Reference_Lexeme_ID", "Sensitivity_Form_Similarity",
    "Sensitivity_Confidence", "Sensitivity_Rationale",
]
GLOBAL_VARIANT_SENSITIVITY_AUDIT_FIELDS = [
    "Lexeme_ID", "Representative_Form", "Representative_Gloss", "Lexical_Sources",
    "Reference_Lexeme_ID", "Reference_Form", "Reference_Gloss", "Reference_Sources",
    "Reference_Stratum", "Whole_Form_Similarity", "Review_Set", "Assessment",
    "Confidence", "Rationale", "Interpretation",
]
CORE_CONCEPT_PROFILE_AUDIT_FIELDS = [
    "Concept", "Lexeme_Cluster_Count", "Lexeme_IDs", "Representative_Forms",
    "Strict_Strata", "Effective_Strata", "Residue_Present",
    "Residual_Root_Hypotheses", "Best_Residual_Replication",
    "Any_Multi_Source_Residual_Root", "Early_Residual_Root",
    "Contact_Families", "Profile_Class",
    "Interpretation",
]
CORE_RESIDUE_ROOT_AUDIT_FIELDS = [
    "Concept", "Cluster_Count", "Root_Group_Count", "Groups", "Confidence", "Rationale",
]
CORE_RESIDUE_ROOT_INVENTORY_FIELDS = [
    "Root_ID", "Concept", "Root_Number", "Cluster_Count", "Cluster_IDs",
    "Representative_Form", "Forms", "Glosses", "Record_Count", "Source_Count",
    "Lexical_Sources", "Early_Source_Attested", "Replication_Grade",
    "Grouping_Confidence", "Grouping_Rationale", "Methods", "Closest_Alternatives",
    "Interpretation",
]
REPLICATED_RESIDUE_AUDIT_FIELDS = [
    "Lexeme_ID", "Representative_Form", "Representative_Gloss", "Record_Count",
    "Source_Count", "Lexical_Sources", "Forms", "Glosses", "Minimum_Form_Similarity",
    "Minimum_Gloss_Similarity", "Core_Concepts", "Core_Effective_Residue",
    "Core_Sensitivity_Stratum", "Methods", "Manual_Decisions", "Closest_Alternatives",
    "Replication_Grade", "Interpretation",
]
RESOLVED_CONTACT_SHAPE_AUDIT_FIELDS = [
    "Lexeme_ID", "Child_Form", "Child_Gloss", "Immediate_Parent_ID",
    "Immediate_Parent_Form", "Resolution_Path", "Parent_ID", "Parent_Form",
    "Parent_Gloss", "Parent_Language_ID", "Parent_Language", "Parent_Family", "Method", "Confidence",
    "Source_Stratum", "Form_Similarity", "Match_Shape", "Initial_Correspondence",
    "Final_Correspondence", "Matched_Surface_ID", "Matched_Surface_Form",
    "Matched_Surface_Gloss", "Matched_Surface_Language_ID", "Matched_Surface_Language",
    "Matched_Surface_Family", "Surface_Form_Similarity", "Surface_Gloss_Similarity",
    "Surface_Match_Shape", "Parent_Link_Count", "Review_Assessment",
    "Correspondence_Series", "Review_Confidence", "Review_Rationale",
    "Interpretive_Caution",
]
DRAVIDIAN_CORRESPONDENCE_AUDIT_FIELDS = [
    "Parent_ID", "Parent_Form", "Parent_Gloss", "Link_Count", "Lexeme_Count",
    "Lexeme_IDs", "Child_Forms", "Child_Glosses", "Source_Strata",
    "Best_Surface_ID", "Best_Surface_Form", "Best_Surface_Gloss",
    "Best_Surface_Language", "Best_Surface_Family", "Best_Surface_Form_Similarity",
    "Best_Surface_Gloss_Similarity", "Assessment", "Initial_Correspondence",
    "Initial_Series_Root_Count", "Series_Type", "Interpretation",
]
CONTACT_EVIDENCE_TIER_AUDIT_FIELDS = [
    "Lexeme_ID", "Representative_Form", "Representative_Gloss", "Record_Count",
    "Source_Count", "Lexical_Sources", "Stratum", "Families", "Evidence_Tier",
    "Tier_Basis", "Resolved_Parent_Families", "Proxy_Evidence_Quality",
    "Proxy_Uncertainty", "Manual_Assessment", "Manual_Confidence",
]
FAMILY_EVIDENCE_BRACKET_AUDIT_FIELDS = [
    "Family", "All_Labelled_Clusters", "High_Specificity_Floor",
    "Supported_Envelope", "Weak_Or_Unresolved_Excluded", "Floor_Share_Of_All_Lexemes",
    "Envelope_Share_Of_All_Lexemes", "Definition",
]
FAMILY_CONTACT_EVIDENCE_AUDIT_FIELDS = [
    "Lexeme_ID", "Family", "Representative_Form", "Representative_Gloss", "Stratum",
    "Evidence_Tier", "Tier_Basis", "Resolved_Parent_IDs", "Resolved_Parent_Families",
    "Matched_Surface_Languages", "Manual_Assessments", "Interpretation",
]
SOURCE_VARIATION_AUDIT_FIELDS = [
    "Lexeme_ID", "Representative_Form", "Representative_Gloss", "Record_Count",
    "Relationship", "Own_Source_Attributions", "Lexical_Sources", "Source_Evidence",
    "Current_Stratum", "Current_Parent_IDs", "Review_Priority", "Review_Assessment",
    "Preferred_Immediate_Donor", "Preferred_Ultimate_Source", "Review_Confidence",
    "Review_Rationale",
]
SOURCE_PROXY_QUALITY_AUDIT_FIELDS = [
    "Lexeme_ID", "Representative_Form", "Representative_Gloss", "Record_Count",
    "Proxy_Record_Count", "Direct_Note_Record_Count", "Propagated_Record_Count",
    "Source_Count", "Lexical_Sources", "Stratum", "Evidence_Quality",
    "Uncertainty", "Directionality", "Compared_Forms", "Catalog_References",
    "Best_Compared_Form", "Best_Form_Similarity", "Comparison_Shape",
    "Core_Vocabulary", "Review_Priority", "Review_Assessment",
    "Preferred_Immediate_Donor", "Preferred_Ultimate_Source", "Review_Confidence",
    "Review_Rationale", "Source_Evidence",
]
KORKU_ROUTE_AUDIT_FIELDS = [
    "Lexeme_ID", "Representative_Form", "Representative_Gloss", "Stratum",
    "Core_Vocabulary", "Compared_Forms", "Evidence_Quality", "Uncertainty", "Matched_Korku_ID",
    "Matched_Korku_Form", "Matched_Korku_Gloss", "Matched_Korku_Source",
    "Form_Similarity", "Gloss_Similarity", "Route_Assessment", "Korku_Edge_Kind",
    "Ultimate_Parent_ID", "Ultimate_Parent_Form", "Ultimate_Parent_Gloss",
    "Ultimate_Parent_Language", "Ultimate_Parent_Family", "Interpretation",
]
INDO_ARYAN_ROUTE_AUDIT_FIELDS = [
    "Lexeme_ID", "Representative_Form", "Representative_Gloss", "Stratum",
    "IA_Source_Language_IDs", "IA_Source_Languages", "Korku_Route_Mentioned",
    "English_Mentioned", "Period_Evidence_Class", "Resolved_Parent_IDs",
    "Resolved_Parent_Languages", "Direct_Source_Note", "Interpretation",
]
LAYER_REPLICATION_AUDIT_FIELDS = [
    "Layer", "Total_Clusters", "Multi_Source_Clusters", "Multi_Source_Share",
    "Konow_Or_Bhattacharya_Attested", "Nagaraja_Attested", "All_Five_Sources",
    "Interpretation",
]
SOURCE_PROFILE_AUDIT_FIELDS = [
    "Lexical_Source", "Record_Count", "Lexeme_Clusters", "Direct_Note_Clusters",
    "Propagated_External_Clusters", "External_Clusters", "External_Share",
    "Residue_Clusters", "Residue_Share", "Korku_Clusters", "Korku_Share",
    "Munda_Clusters", "Munda_Share", "Indo_Aryan_Clusters", "Indo_Aryan_Share",
    "Dravidian_Clusters", "Dravidian_Share", "Interpretation",
]
CROSS_SOURCE_AGREEMENT_AUDIT_FIELDS = [
    "Lexeme_ID", "Representative_Form", "Representative_Gloss", "Record_Count",
    "Source_Count", "Lexical_Sources", "Labelled_Source_Count", "Labels_By_Source",
    "Agreement_Class", "Current_Stratum", "Source_Evidence", "Interpretation",
]
FAMILY_ATTRIBUTION_REPLICATION_AUDIT_FIELDS = [
    "Family", "All_Labelled_Clusters", "Multi_Source_Clusters",
    "No_Direct_Family_Label", "One_Direct_Labelled_Source",
    "Two_Plus_Direct_Labelled_Sources", "Two_Plus_Share_Of_Multi_Source",
    "Interpretation",
]
LAYER_CATEGORY_AUDIT_FIELDS = [
    "Layer", "Total_Clusters", "Concept_Linked_Clusters", "Noun_Clusters", "Verb_Clusters",
    "Adjective_Clusters", "Numeral_Clusters", "Other_Clusters", "Interpretation",
]
LAYER_FORM_SHAPE_AUDIT_FIELDS = [
    "Layer", "Total_Clusters", "Mean_Folded_Length", "Median_Folded_Length",
    "Final_Vowel_Count", "Final_Vowel_Share", "Multiword_Or_Compound_Count",
    "Multiword_Or_Compound_Share", "Retroflex_Count", "Retroflex_Share",
    "Aspiration_Count", "Aspiration_Share", "Nasalization_Count",
    "Nasalization_Share", "Interpretation",
]
CLOSED_CLASS_AUDIT_FIELDS = [
    "Domain", "Concept", "Lexeme_ID", "Representative_Form", "Forms", "Glosses",
    "Record_Count", "Source_Count", "Lexical_Sources", "Strict_Stratum",
    "Effective_Stratum", "Core_Sensitivity_Applied", "Methods", "Confidence",
    "Interpretation",
]
ORIGIN_EVIDENCE_MATRIX_FIELDS = [
    "Evidence_ID", "Domain", "Finding", "Supports_Hypothesis",
    "Challenges_Hypothesis", "Evidential_Weight", "Limitation", "Audit_Or_Source",
]
RESIDUE_THRESHOLD_SENSITIVITY_FIELDS = [
    "Threshold_Label", "Minimum_Composite_Score", "Minimum_Margin",
    "Flagged_Residue_Clusters", "Indo_Aryan_Candidates", "Dravidian_Candidates",
    "Munda_Candidates", "Other_Candidates", "English_Candidates",
    "Manual_Rejected", "Manual_Deferred", "Unreviewed", "Interpretation",
]

# Exact database concept names corresponding to a conservative 93-item subset of the classic
# Swadesh-100 domains. Seven unavailable names (GREASE, HORN, FLY, LIE, RAIN, EARTH, SMOKE) are
# omitted rather than approximated with broader or narrower database concepts.
CORE_CONCEPTS = frozenset(
    "I YOU WE THIS THAT WHO WHAT NOT ALL MANY ONE TWO BIG LONG SMALL WOMAN MAN PERSON "
    "FISH BIRD DOG LOUSE TREE SEED LEAF ROOT BARK SKIN FLESH BLOOD BONE EGG TAIL "
    "FEATHER HAIR HEAD EAR EYE NOSE MOUTH TOOTH TONGUE CLAW FOOT KNEE HAND BELLY NECK "
    "BREAST HEART LIVER DRINK EAT BITE SEE HEAR KNOW SLEEP DIE KILL SWIM WALK COME SIT "
    "STAND GIVE SAY SUN MOON STAR WATER STONE SAND CLOUD FIRE ASH BURN PATH MOUNTAIN RED "
    "GREEN YELLOW WHITE BLACK NIGHT HOT COLD FULL NEW GOOD ROUND DRY NAME".split()
)

STRATUM_ORDER = (
    "Korku", "Munda", "Indo-Aryan", "Dravidian", "English", "Other",
)

IA_CLADES = {
    "OIA", "Early NIA", "MIA", "Dardic", "Nuristani", "Shinaic", "Kohistani",
    "Lahndic", "Sindhic", "W. Pahari", "E. Pahari", "W. Hindi", "E. Hindi", "Bihari",
    "Eastern", "Marathi-Konkani", "Gujaratic", "Rajasthanic", "Bhil", "Romani",
}
DRAVIDIAN_CLADES = {
    "Old Dravidian", "S. Dravidian I", "S. Dravidian II", "C. Dravidian", "N. Dravidian",
}
STOPWORDS = {
    "a", "an", "and", "be", "for", "in", "kind", "of", "one", "someone", "something",
    "the", "to", "type", "with", "etc", "id", "old", "word",
}
SOURCE_LABELS = {
    "Korku": "Korku",
    "Hindi": "Indo-Aryan", "Marathi": "Indo-Aryan", "Sanskrit": "Indo-Aryan",
    "Bengali": "Indo-Aryan", "Beng": "Indo-Aryan", "IA": "Indo-Aryan",
    "Indo-Aryan": "Indo-Aryan", "Konkani": "Indo-Aryan", "Pj": "Indo-Aryan",
    "Dravidian": "Dravidian", "Drav": "Dravidian", "Dr": "Dravidian",
    "Proto-Dravidian": "Dravidian", "Proto-Dr": "Dravidian", "PDr": "Dravidian",
    "Gondi": "Dravidian", "Kolami": "Dravidian", "Naiki": "Dravidian",
    "Tamil": "Dravidian", "Telugu": "Dravidian", "Kurukh": "Dravidian",
    "Munda": "Munda", "Mundari": "Munda", "Santali": "Munda", "Kharia": "Munda",
    "Par": "Munda", "Gadaba": "Munda", "Sora": "Munda", "Gutob": "Munda",
    "English": "English",
    "Burushaski": "Other",
}
SPECIFIC_LANGUAGE = {
    "Korku": "ko", "Hindi": "H", "Marathi": "M", "Sanskrit": "Sk",
    "Bengali": "B", "Beng": "B", "Konkani": "Ko", "English": "Eng",
    "Mundari": "mu", "Santali": "sa", "Gondi": "Gondi", "Kolami": "Kolami",
    "Naiki": "Naiki", "Tamil": "Tamil", "Telugu": "Telugu",
    "Dravidian": "Drav", "Drav": "Drav", "Dr": "Drav", "PDr": "PDr",
    "Proto-Dravidian": "PDr", "Proto-Dr": "PDr", "Munda": "PMu",
    "Indo-Aryan": "Indo-Aryan", "IA": "Indo-Aryan",
    "Burushaski": "Bur",
}

# Mundlay's printed abbreviation key (1996: 16) defines K = Korku, H/Hd/HM =
# Hindi or Hindi-Marathi, M/Md = Marathi, Sk = Sanskrit, NM/SM = North/South
# Munda, and GR = Gutob-Remo.  Additional Ga/Gu/SG/GRG comparisons in her
# discussion are South Munda language abbreviations.  These patterns remain
# case-sensitive so ordinary prose and grammatical abbreviations do not become
# donor labels.
SOURCE_ABBREVIATIONS = (
    (r"(?<![A-Za-z])K\.", "K.", "Korku", "ko"),
    (r"(?<![A-Za-z])(?:H|Hd|HM|Hi)\.", "H.", "Indo-Aryan", "H"),
    (r"(?<![A-Za-z])HM(?![A-Za-z])", "HM", "Indo-Aryan", "Indo-Aryan"),
    (r"(?<![A-Za-z])(?:M|Md)\.", "M.", "Indo-Aryan", "M"),
    (r"(?<![A-Za-z])Sk\.", "Sk.", "Indo-Aryan", "Sk"),
    (r"(?<![A-Za-z])(?:NM|SM|PM)(?:\.|(?![A-Za-z]))", "SM", "Munda", "PMu"),
    (r"(?<![A-Za-z])(?:GRG|GR|SG)(?:\.|(?![A-Za-z]))", "GR", "Munda", "PMu"),
    (r"(?<![A-Za-z])Ga\.", "Ga.", "Munda", "PMu"),
    (r"(?<![A-Za-z])G[Uu](?:\.|(?![A-Za-z]))", "Gu", "Munda", "gu"),
)


def read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_dicts(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_candidate_reviews(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows = read_dicts(path)
    reviews: dict[str, dict[str, str]] = {}
    for review in rows:
        lexeme_id = review["Lexeme_ID"]
        if lexeme_id in reviews:
            raise RuntimeError(f"duplicate manual review for {lexeme_id} in {path.name}")
        if review["Decision"] not in {"accept", "reject", "defer"}:
            raise RuntimeError(f"invalid manual decision for {lexeme_id}: {review['Decision']}")
        if review["Decision"] == "accept":
            if not review["Parent_ID"]:
                raise RuntimeError(f"accepted manual review lacks parent for {lexeme_id}")
            if not review["Stratum"] or review["Confidence"] not in {"high", "medium", "low"}:
                raise RuntimeError(f"incomplete accepted manual review for {lexeme_id}")
        if not review["Rationale"]:
            raise RuntimeError(f"manual review lacks rationale for {lexeme_id}")
        reviews[lexeme_id] = review
    return rows, reviews


def fold(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.translate(str.maketrans({
        "ʈ": "t", "ṭ": "t", "ɖ": "d", "ḍ": "d", "ɽ": "r", "ṛ": "r",
        "ɳ": "n", "ṇ": "n", "ŋ": "n", "ñ": "n", "ʃ": "s", "ś": "s",
        "ṣ": "s", "č": "c", "ʔ": "", "ː": "", "w": "v", "ᵑ": "n",
        "ə": "a", "ɛ": "e", "ɔ": "o", "ɸ": "ph", "β": "b",
    }))
    return re.sub(r"[^a-z0-9]+", "", value)


def form_variants(value: str) -> set[str]:
    pieces = re.split(r"[\s/;,~]+", value)
    result = {fold(value)} | {fold(piece) for piece in pieces if piece}
    expanded = set(result)
    for item in result:
        for ending in ("kambe", "jere", "be", "bo", "ka", "ko"):
            if item.endswith(ending) and len(item) > len(ending) + 2:
                expanded.add(item[:-len(ending)])
    return {item for item in expanded if len(item) >= 2}


def gloss_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"\([^)]*\)", " ", value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def gloss_tokens(value: str) -> set[str]:
    tokens = set()
    for token in gloss_text(value).split():
        if token in STOPWORDS or len(token) < 2:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.add(token)
    return tokens


def gloss_similarity(left: str, right: str) -> float:
    a, b = gloss_tokens(left), gloss_tokens(right)
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    jaccard = overlap / len(a | b)
    containment = overlap / min(len(a), len(b))
    string = SequenceMatcher(None, gloss_text(left), gloss_text(right)).ratio()
    return max(jaccard, 0.75 * containment + 0.25 * string)


def form_similarity(left: set[str], right: set[str]) -> float:
    return max((SequenceMatcher(None, a, b).ratio() for a in left for b in right), default=0.0)


def language_family(language_id: str, languages: dict[str, dict[str, str]]) -> str:
    if language_id == "Ni":
        return "Nihali residue"
    if language_id == "ko":
        return "Korku"
    if language_id in {"Indo-Aryan", "Sk", "OIA", "MIA"}:
        return "Indo-Aryan"
    if language_id in {"Drav", "PDr", "PSTDr", "PSD1", "PSD2", "PCDr", "PKMDr", "PNDr"}:
        return "Dravidian"
    if language_id in {"PMu", "PKher"}:
        return "Munda"
    if language_id == "Eng":
        return "English"
    clade = languages.get(language_id, {}).get("Clade", "")
    if clade == "Munda":
        return "Munda"
    if clade in DRAVIDIAN_CLADES:
        return "Dravidian"
    if clade in IA_CLADES:
        return "Indo-Aryan"
    return "Other"


def lexical_source(source: str) -> str:
    return source.split("[", 1)[0]


def ordered_strata(values: set[str]) -> list[str]:
    return [value for value in STRATUM_ORDER if value in values]


def source_attribution(etymology: str) -> tuple[list[str], list[str], list[str]]:
    """Return broad strata, mentioned language IDs, and plausible compared forms."""
    if not etymology:
        return [], [], []
    hits: list[tuple[int, str, str, str]] = []
    for label, family in SOURCE_LABELS.items():
        for match in re.finditer(rf"(?<![A-Za-z]){re.escape(label)}(?![A-Za-z])", etymology, re.I):
            hits.append((match.start(), match.group(), family, SPECIFIC_LANGUAGE.get(label, "")))
    for pattern, label, family, language_id in SOURCE_ABBREVIATIONS:
        for match in re.finditer(pattern, etymology):
            hits.append((match.start(), match.group(), family, language_id))
    hits.sort()
    found_strata = set(family for _, _, family, _ in hits)
    strata = ordered_strata(found_strata)
    language_ids = list(dict.fromkeys(
        language_id for _, _, _, language_id in hits if language_id
    ))
    compared: list[str] = []
    for position, label, _family, _language_id in hits:
        start = position + len(label)
        tail = etymology[start:start + 90]
        tail = re.sub(r"^[\s:<>?./-]+", "", tail)
        tail = re.split(
            r"[;\n]|\s+(?:cf\.|etc\.|from|probably|possibly|perhaps|may\b)",
            tail, maxsplit=1, flags=re.I,
        )[0]
        # In entries such as Hindi aghānā 'to be full', everything before the definition quote
        # is the compared form.  Keep at most three tokens to avoid swallowing prose.
        tail = re.split(r"['\"“”]", tail, maxsplit=1)[0].strip(" ,.:()")
        words = tail.split()
        if words:
            candidate = " ".join(words[:3])
            if (fold(candidate) and not re.fullmatch(
                r"(?i)(word|loan|id|same|likely|uncertain|source|borrowing)", candidate
            )):
                compared.append(candidate)
    return strata, language_ids, list(dict.fromkeys(compared))


def source_proxy_display_form(
    lexeme_id: str, compared_forms: list[str], fallback_form: str,
) -> str:
    """Return a readable proxy label without rewriting the preserved source evidence.

    ``source_attribution`` deliberately keeps short pieces of the original note for matching and
    auditing. A parenthetical qualifier can therefore be truncated at the edge of its retrieval
    window (for example ``dialectal) cili`` or ``*jhapp- (CDAL 5337``). Those raw pieces remain in
    the audit, but they should not become the form displayed as a generated donor node in the app.
    This cleanup is presentation-only and cannot affect candidate retrieval or graph decisions.
    """
    override = SOURCE_PROXY_FORM_DISPLAY_BY_LEXEME.get(lexeme_id)
    if override:
        return override
    if not compared_forms:
        return "?" + fallback_form

    display = compared_forms[0].strip()
    # The parser can strip only the opening parenthesis from a leading metadata qualifier.
    display = re.sub(
        r"^(?:dialectal|colloquial|rare|archaic|obsolete)\)\s*",
        "",
        display,
        flags=re.I,
    )
    if display.count("(") > display.count(")"):
        # A spaced parenthesis begins prose/catalog metadata, not the lexical form itself.
        if " (" in display:
            display = display.split(" (", 1)[0]
        else:
            # Compact optional segments such as isa(ʔ) and ghaṭa(w) are linguistic notation.
            display += ")"
    elif display.count(")") > display.count("("):
        compact_optional = re.match(r"^([^\W\d_])\)(\S.*)$", display, flags=re.UNICODE)
        if compact_optional:
            display = f"({compact_optional.group(1)}){compact_optional.group(2)}"
        else:
            display = display.split(")", 1)[1].lstrip()
    display = display.rstrip(" ,;:/")
    return display or "?" + fallback_form


def build_lexeme_clusters(
    targets: list[dict[str, str]],
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Conservatively cluster repeated dictionary attestations of the same lexical item.

    Exact normalized forms require compatible meanings.  Near matches require overlapping gloss
    evidence, distinct lexical sources, and at least 0.88 form similarity.  This is deliberately
    stricter than ordinary cognate detection: the purpose is only to recognize repeated Nihali
    dictionary records, not to discover relations among different Nihali roots.  It does not use
    generated concept links, so rebuilding this overlay cannot change its own clustering inputs.
    """
    parent = {row["ID"]: row["ID"] for row in targets}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_exact: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_gloss_token: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in targets:
        normalized = fold(row["Form"] or row["Original"])
        if normalized:
            by_exact[normalized].append(row)
        for token in gloss_tokens(row["Gloss"]):
            by_gloss_token[token].append(row)

    for group in by_exact.values():
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                if gloss_similarity(left["Gloss"], right["Gloss"]) >= 0.35:
                    union(left["ID"], right["ID"])

    seen_pairs: set[tuple[str, str]] = set()
    for group in by_gloss_token.values():
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                if lexical_source(left["Source"]) == lexical_source(right["Source"]):
                    continue
                pair = tuple(sorted((left["ID"], right["ID"])))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if (
                    gloss_similarity(left["Gloss"], right["Gloss"]) >= 0.55
                    and form_similarity(
                        {fold(left["Form"] or left["Original"])},
                        {fold(right["Form"] or right["Original"])},
                    ) >= 0.88
                ):
                    union(left["ID"], right["ID"])

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in targets:
        grouped[find(row["ID"])].append(row)

    clusters: dict[str, dict[str, object]] = {}
    cluster_for_form: dict[str, str] = {}
    for group in grouped.values():
        member_ids = sorted(row["ID"] for row in group)
        lexeme_id = "nilex-" + hashlib.sha1("|".join(member_ids).encode("utf-8")).hexdigest()[:14]
        strata_set: set[str] = set()
        language_ids: list[str] = []
        compared_forms: list[str] = []
        own_labels: set[str] = set()
        query_forms: set[str] = set()
        for row in group:
            strata, mentioned_ids, compared = source_attribution(row.get("Etymology", ""))
            strata_set.update(strata)
            if strata:
                own_labels.add("+".join(strata))
            language_ids.extend(mentioned_ids)
            compared_forms.extend(compared)
            query_forms.update(form_variants(row["Form"] or row["Original"]))
            for item in compared:
                query_forms.update(form_variants(item))
        info: dict[str, object] = {
            "id": lexeme_id,
            "rows": group,
            "size": len(group),
            "strata": ordered_strata(strata_set),
            "language_ids": list(dict.fromkeys(language_ids)),
            "compared_forms": list(dict.fromkeys(compared_forms)),
            "query_forms": query_forms,
            "source_labels": sorted(own_labels),
            "lexical_sources": sorted({lexical_source(row["Source"]) for row in group}),
        }
        clusters[lexeme_id] = info
        for form_id in member_ids:
            cluster_for_form[form_id] = lexeme_id
    return clusters, cluster_for_form


@dataclass(frozen=True)
class Candidate:
    parent_id: str
    surface_id: str
    form: str
    gloss: str
    language_id: str
    family: str
    form_set: frozenset[str]


def effective_parent(form_id: str, rank1: dict[str, dict[str, str]]) -> str:
    seen = set()
    current = form_id
    while current in rank1 and current not in seen:
        seen.add(current)
        current = rank1[current]["Parent_ID"]
    return current


def build_candidates(
    forms: list[dict[str, str]], rank1: dict[str, dict[str, str]],
    languages: dict[str, dict[str, str]],
) -> tuple[list[Candidate], dict[str, set[int]], dict[str, set[int]]]:
    by_id = {row["ID"]: row for row in forms}
    candidates: list[Candidate] = []
    seen = set()
    by_token: dict[str, set[int]] = defaultdict(set)
    by_gloss: dict[str, set[int]] = defaultdict(set)
    for row in forms:
        if (
            row["Language_ID"] == "Ni" or row["ID"].startswith("nihprov-")
            or row.get("Source") == REFERENCE
        ):
            continue
        family = language_family(row["Language_ID"], languages)
        if family not in {"Korku", "Munda", "Indo-Aryan", "Dravidian", "English"}:
            continue
        parent_id = effective_parent(row["ID"], rank1)
        parent = by_id.get(parent_id)
        # Only an entry can be the target of an assignment.  An unlinked donor attestation may be
        # excellent surface evidence, but Jambu deliberately prevents unattached attestations from
        # becoming ancestors.  Such a case must remain a donor proxy until that donor is itself
        # etymologised.
        if not parent or parent["Language_ID"] == "Ni" or parent.get("Status") == "unlinked":
            continue
        fset = frozenset(form_variants(row["Form"] or row["Original"]))
        if not fset:
            continue
        gloss = row["Gloss"] or parent["Gloss"]
        key = (parent_id, tuple(sorted(fset)), gloss_text(gloss), row["Language_ID"])
        if key in seen:
            continue
        seen.add(key)
        index = len(candidates)
        candidates.append(Candidate(
            parent_id, row["ID"], row["Form"], gloss, row["Language_ID"], family, fset
        ))
        for token in gloss_tokens(gloss):
            by_token[token].add(index)
        if gloss_text(gloss):
            by_gloss[gloss_text(gloss)].add(index)
    return candidates, by_token, by_gloss


def candidate_pool(
    gloss: str, by_token: dict[str, set[int]], by_gloss: dict[str, set[int]]
) -> set[int]:
    exact = by_gloss.get(gloss_text(gloss), set())
    if exact:
        pool = set(exact)
    else:
        token_sets = [by_token[token] for token in gloss_tokens(gloss) if token in by_token]
        pool = set().union(*token_sets) if token_sets else set()
    # Very generic meanings can produce tens of thousands of attestations.  Stable truncation is
    # acceptable here because candidates are later deduplicated by parent and sorted by score.
    return set(sorted(pool)[:20000])


def rank_candidates(
    row: dict[str, str], query_forms: set[str], allowed_strata: set[str], allowed_ids: set[str],
    candidates: list[Candidate], pool: set[int],
) -> list[tuple[float, float, float, Candidate]]:
    best_by_parent: dict[str, tuple[float, float, float, Candidate]] = {}
    for index in pool:
        candidate = candidates[index]
        if allowed_strata and candidate.family not in allowed_strata:
            continue
        # A source that actually names Marathi, Hindi, Korku, etc. licenses comparison with that
        # surface donor, whose accepted edge may then lead to a deeper entry.  It does not license
        # choosing an arbitrary same-family lect merely because its form happens to be similar.
        # Generic family IDs remain broad by design.
        generic_ids = {"Indo-Aryan", "Drav", "PDr", "PMu"}
        generic_families = {
            "Indo-Aryan": "Indo-Aryan", "Drav": "Dravidian",
            "PDr": "Dravidian", "PMu": "Munda",
        }
        specific_ids = allowed_ids - generic_ids
        broad_families = {generic_families[item] for item in allowed_ids & generic_ids}
        if (
            specific_ids and candidate.language_id not in specific_ids
            and candidate.family not in broad_families
        ):
            continue
        fsim = form_similarity(query_forms, set(candidate.form_set))
        if fsim < 0.46:
            continue
        gsim = gloss_similarity(row["Gloss"], candidate.gloss)
        language_bonus = 0.04 if allowed_ids and candidate.language_id in allowed_ids else 0.0
        score = min(1.0, 0.64 * fsim + 0.36 * gsim + language_bonus)
        value = (score, fsim, gsim, candidate)
        if candidate.parent_id not in best_by_parent or value[:3] > best_by_parent[candidate.parent_id][:3]:
            best_by_parent[candidate.parent_id] = value
    return sorted(best_by_parent.values(), key=lambda item: (item[0], item[1], item[2]), reverse=True)


def proxy_id(key: str) -> str:
    return "nihprov-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:14]


def concise_alternatives(ranked: list[tuple[float, float, float, Candidate]]) -> str:
    return "; ".join(
        f"{item[3].parent_id}:{item[3].form}<{item[3].language_id}>={item[0]:.3f}"
        for item in ranked[:3]
    )


def build_cluster_audit(audit: list[dict[str, str]]) -> list[dict[str, str]]:
    by_lexeme: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        by_lexeme[row["Lexeme_ID"]].append(row)
    confidence_rank = {"high": 3, "medium": 2, "low": 1, "unresolved": 0}
    result = []
    for lexeme_id, group in sorted(by_lexeme.items()):
        representative = sorted(group, key=lambda row: (len(row["Form"]), row["Form_ID"]))[0]
        labels = {row["Stratum"] for row in group}
        external_parts = {
            part for label in labels if label not in {"Nihali residue", "Other", ""}
            for part in label.split("+")
        }
        if external_parts:
            stratum = "+".join(ordered_strata(external_parts))
        elif "Other" in labels:
            stratum = "Other"
        else:
            stratum = "Nihali residue"
        own_labels = sorted({
            row["Own_Source_Attribution"] for row in group if row["Own_Source_Attribution"]
        })
        methods = sorted({row["Method"] for row in group})
        parents = sorted({row["Parent_ID"] for row in group})
        manual_decisions = sorted({
            row["Manual_Decision"] for row in group if row["Manual_Decision"]
        })
        manual_triggers = sorted({
            row["Manual_Trigger"] for row in group if row["Manual_Trigger"]
        })
        manual_rationales = sorted({
            row["Manual_Rationale"] for row in group if row["Manual_Rationale"]
        })
        flags = []
        if len(group) == 1:
            flags.append("single-attestation")
        if len(own_labels) > 1:
            flags.append("competing-source-labels")
        if len(parents) > 1:
            flags.append("multiple-parent-hypotheses")
        if "manual-deferred" in methods:
            flags.append("manual-review-deferred")
        if "manual-rejected" in methods:
            flags.append("machine-suggestion-rejected")
        if "manual-resolved" in methods:
            flags.append("manual-candidate-accepted")
        if stratum == "Nihali residue":
            flags.append("residue-not-asserted-inherited")
        result.append({
            "Lexeme_ID": lexeme_id,
            "Representative_Form": representative["Form"],
            "Representative_Gloss": representative["Gloss"],
            "Record_Count": str(len(group)),
            "Source_Count": str(len({row["Lexical_Source"] for row in group})),
            "Lexical_Sources": "; ".join(sorted({row["Lexical_Source"] for row in group})),
            "Forms": "; ".join(sorted({row["Form"] for row in group})),
            "Glosses": "; ".join(sorted({row["Gloss"] for row in group})),
            "Stratum": stratum,
            "Own_Source_Attributions": "; ".join(own_labels),
            "Methods": "; ".join(methods),
            "Parent_IDs": "; ".join(parents),
            "Confidence": max(
                (row["Confidence"] for row in group), key=lambda item: confidence_rank[item]
            ),
            "Manual_Decisions": "; ".join(manual_decisions),
            "Manual_Triggers": "; ".join(manual_triggers),
            "Manual_Rationales": "; ".join(manual_rationales),
            "Review_Flags": "; ".join(flags),
        })
    return result


def build_global_variant_sensitivity_audit(
    cluster_audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Audit conservative under-clustering without changing graph assignments.

    The mechanical screen is intentionally narrow: exact normalized gloss and at least .72
    whole-form similarity between a residual cluster and an externally attributed cluster. Every
    non-core hit must occur in the full-lexicon review register. That register may also retain
    manually identified close semantic variants just outside the exact-gloss screen. The already
    reviewed core list is then added, including its lower-similarity morphological judgments, to
    expose a broad and a stricter sensitivity bound.
    """
    by_id = {row["Lexeme_ID"]: row for row in cluster_audit}
    core_rows = read_dicts(CORE_VARIANT_REVIEW)
    global_rows = read_dicts(GLOBAL_VARIANT_REVIEW)
    core_reviews = {row["Lexeme_ID"]: row for row in core_rows}
    global_reviews = {row["Lexeme_ID"]: row for row in global_rows}
    if len(core_reviews) != len(core_rows) or len(global_reviews) != len(global_rows):
        raise RuntimeError("duplicate lexical-variant sensitivity review")
    if set(core_reviews) & set(global_reviews):
        raise RuntimeError("core and full-lexicon variant review registers overlap")

    external_by_gloss: dict[str, list[dict[str, str]]] = defaultdict(list)
    residual = []
    for row in cluster_audit:
        if row["Stratum"] == "Nihali residue":
            residual.append(row)
        elif row["Stratum"] not in {"Other", ""}:
            external_by_gloss[gloss_text(row["Representative_Gloss"])].append(row)
    mechanical_targets = set()
    for row in residual:
        target_form = fold(row["Representative_Form"])
        for reference in external_by_gloss.get(gloss_text(row["Representative_Gloss"]), []):
            similarity = SequenceMatcher(
                None, target_form, fold(reference["Representative_Form"])
            ).ratio()
            if similarity >= 0.72:
                mechanical_targets.add(row["Lexeme_ID"])
                break
    expected_global = mechanical_targets - set(core_reviews)
    if not expected_global <= set(global_reviews):
        raise RuntimeError(
            "full-lexicon variant review coverage mismatch; missing="
            f"{sorted(expected_global - set(global_reviews))}"
        )

    result = []
    for review_set, reviews in (
        ("core-basic-vocabulary", core_reviews),
        ("global-full-lexicon", global_reviews),
    ):
        for lexeme_id, review in sorted(reviews.items()):
            reference_id = review["Reference_Lexeme_ID"]
            target = by_id.get(lexeme_id)
            reference = by_id.get(reference_id)
            if not target or not reference:
                raise RuntimeError(
                    f"variant review references missing cluster: {lexeme_id} -> {reference_id}"
                )
            if target["Stratum"] != "Nihali residue" or reference["Stratum"] in {
                "Nihali residue", "Other", "",
            }:
                raise RuntimeError(
                    f"variant review has invalid strata: {lexeme_id} -> {reference_id}"
                )
            if review_set == "core-basic-vocabulary":
                if review["Confidence"] not in {"high", "medium", "low"}:
                    raise RuntimeError(f"invalid core variant confidence: {lexeme_id}")
                assessment = (
                    "variant" if review["Confidence"] in {"high", "medium"} else "qualified"
                )
            else:
                assessment = review["Assessment"]
                if assessment not in {"variant", "qualified", "reject"}:
                    raise RuntimeError(f"invalid global variant assessment: {lexeme_id}")
                if review["Confidence"] not in {"high", "medium", "low"}:
                    raise RuntimeError(f"invalid global variant confidence: {lexeme_id}")
            if not review["Rationale"]:
                raise RuntimeError(f"variant review lacks rationale: {lexeme_id}")
            similarity = SequenceMatcher(
                None,
                fold(target["Representative_Form"]),
                fold(reference["Representative_Form"]),
            ).ratio()
            result.append({
                "Lexeme_ID": lexeme_id,
                "Representative_Form": target["Representative_Form"],
                "Representative_Gloss": target["Representative_Gloss"],
                "Lexical_Sources": target["Lexical_Sources"],
                "Reference_Lexeme_ID": reference_id,
                "Reference_Form": reference["Representative_Form"],
                "Reference_Gloss": reference["Representative_Gloss"],
                "Reference_Sources": reference["Lexical_Sources"],
                "Reference_Stratum": reference["Stratum"],
                "Whole_Form_Similarity": f"{similarity:.3f}",
                "Review_Set": review_set,
                "Assessment": assessment,
                "Confidence": review["Confidence"],
                "Rationale": review["Rationale"],
                "Interpretation": (
                    "Sensitivity only: a reviewed lexical-variant relation can reveal strict "
                    "under-clustering, but does not alter the installed rank-1 hypothesis or prove "
                    "the reference cluster's donor attribution."
                ),
            })
    return sorted(result, key=lambda row: (row["Lexeme_ID"], row["Reference_Lexeme_ID"]))


def load_residue_contact_component_review(
    cluster_audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Validate transparent contact material retained inside otherwise residual expressions."""
    rows = read_dicts(RESIDUE_CONTACT_COMPONENT_REVIEW)
    by_id = {row["Lexeme_ID"]: row for row in cluster_audit}
    if len(rows) != len({row["Lexeme_ID"] for row in rows}):
        raise RuntimeError("duplicate residue contact-component review")
    for row in rows:
        target = by_id.get(row["Lexeme_ID"])
        if not target or target["Stratum"] != "Nihali residue":
            raise RuntimeError(
                f"contact-component review target is not residual: {row['Lexeme_ID']}"
            )
        if row["Assessment"] not in {
            "transparent-component", "qualified-component", "source-gloss-conflict",
        }:
            raise RuntimeError(f"invalid contact-component assessment: {row['Lexeme_ID']}")
        if row["Confidence"] not in {"high", "medium", "low"}:
            raise RuntimeError(f"invalid contact-component confidence: {row['Lexeme_ID']}")
        if not all(row[field] for field in (
            "Form", "Gloss", "Contact_Component", "Layer", "Rationale",
        )):
            raise RuntimeError(f"incomplete contact-component review: {row['Lexeme_ID']}")
    return rows


def build_layer_replication_audit(
    cluster_audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Compare cross-source replication without treating documentation date as loan date."""
    layers = ("Nihali residue", "Korku", "Munda", "Indo-Aryan", "Dravidian")
    early_sources = {"konow1906", "bhattacharya1957"}
    result = []
    for layer in layers:
        rows = [
            row for row in cluster_audit
            if (
                row["Stratum"] == layer if layer == "Nihali residue"
                else layer in row["Stratum"].split("+")
            )
        ]
        multi_source = sum(int(row["Source_Count"]) >= 2 for row in rows)
        early = sum(
            bool(early_sources & set(row["Lexical_Sources"].split("; "))) for row in rows
        )
        nagaraja = sum("nagaraja2014" in row["Lexical_Sources"].split("; ") for row in rows)
        all_five = sum(int(row["Source_Count"]) == 5 for row in rows)
        result.append({
            "Layer": layer,
            "Total_Clusters": str(len(rows)),
            "Multi_Source_Clusters": str(multi_source),
            "Multi_Source_Share": f"{multi_source / len(rows):.3f}",
            "Konow_Or_Bhattacharya_Attested": str(early),
            "Nagaraja_Attested": str(nagaraja),
            "All_Five_Sources": str(all_five),
            "Interpretation": (
                "Replication establishes documentary stability, not the age or inherited status "
                "of the layer; source coverage and dialect sampling are unequal."
            ),
        })
    return result


def build_source_profile_audit(audit: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize family involvement within each lexical source's own cluster inventory."""
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        by_source[row["Lexical_Source"]].append(row)
    result = []
    for source, rows in sorted(by_source.items()):
        lexemes = {row["Lexeme_ID"] for row in rows}
        external = {
            row["Lexeme_ID"] for row in rows
            if row["Stratum"] not in {"Nihali residue", "Other", ""}
        }
        direct = {row["Lexeme_ID"] for row in rows if row["Own_Source_Attribution"]}
        family_sets = {
            family: {
                row["Lexeme_ID"] for row in rows if family in row["Stratum"].split("+")
            }
            for family in ("Korku", "Munda", "Indo-Aryan", "Dravidian")
        }
        residue = lexemes - external
        total = len(lexemes)
        result.append({
            "Lexical_Source": source,
            "Record_Count": str(len(rows)),
            "Lexeme_Clusters": str(total),
            "Direct_Note_Clusters": str(len(direct)),
            "Propagated_External_Clusters": str(len(external - direct)),
            "External_Clusters": str(len(external)),
            "External_Share": f"{len(external) / total:.3f}",
            "Residue_Clusters": str(len(residue)),
            "Residue_Share": f"{len(residue) / total:.3f}",
            "Korku_Clusters": str(len(family_sets["Korku"])),
            "Korku_Share": f"{len(family_sets['Korku']) / total:.3f}",
            "Munda_Clusters": str(len(family_sets["Munda"])),
            "Munda_Share": f"{len(family_sets['Munda']) / total:.3f}",
            "Indo_Aryan_Clusters": str(len(family_sets["Indo-Aryan"])),
            "Indo_Aryan_Share": f"{len(family_sets['Indo-Aryan']) / total:.3f}",
            "Dravidian_Clusters": str(len(family_sets["Dravidian"])),
            "Dravidian_Share": f"{len(family_sets['Dravidian']) / total:.3f}",
            "Interpretation": (
                "Shares are source-normalized cluster coverage, not chronological loan rates. "
                "Differences also reflect location, elicitation scope, and editorial practice."
            ),
        })
    return result


def build_cross_source_agreement_audit(
    audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Measure direct donor-label agreement only among replicated lexical clusters."""
    by_lexeme: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        by_lexeme[row["Lexeme_ID"]].append(row)
    result = []
    for lexeme_id, rows in sorted(by_lexeme.items()):
        lexical_sources = sorted({row["Lexical_Source"] for row in rows})
        if len(lexical_sources) < 2:
            continue
        labels_by_source: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row["Own_Source_Attribution"]:
                labels_by_source[row["Lexical_Source"]].update(
                    row["Own_Source_Attribution"].split("+")
                )
        label_sets = [frozenset(value) for value in labels_by_source.values() if value]
        if not label_sets:
            agreement = "no-direct-label"
        elif len(label_sets) == 1:
            agreement = "single-labelled-source"
        elif len(set(label_sets)) == 1:
            agreement = "multi-source-exact-agreement"
        elif all(
            left <= right or right <= left
            for index, left in enumerate(label_sets)
            for right in label_sets[index + 1:]
        ):
            agreement = "multi-source-nested-compatible"
        elif set.intersection(*(set(value) for value in label_sets)):
            agreement = "multi-source-overlap"
        else:
            agreement = "multi-source-disjoint"
        representative = min(rows, key=lambda row: (len(row["Form"]), row["Form_ID"]))
        source_evidence = sorted({
            f"{row['Lexical_Source']}: {row['Original_Etymology']}"
            for row in rows if row["Own_Source_Attribution"] and row["Original_Etymology"]
        })
        current_parts = {
            part for row in rows for part in row["Stratum"].split("+")
            if part not in {"Nihali residue", "Other", ""}
        }
        current_stratum = (
            "+".join(ordered_strata(current_parts)) if current_parts
            else ("Other" if any(row["Stratum"] == "Other" for row in rows)
                  else "Nihali residue")
        )
        result.append({
            "Lexeme_ID": lexeme_id,
            "Representative_Form": representative["Form"],
            "Representative_Gloss": representative["Gloss"],
            "Record_Count": str(len(rows)),
            "Source_Count": str(len(lexical_sources)),
            "Lexical_Sources": "; ".join(lexical_sources),
            "Labelled_Source_Count": str(len(label_sets)),
            "Labels_By_Source": " || ".join(
                f"{source}: {'+'.join(ordered_strata(labels))}"
                for source, labels in sorted(labels_by_source.items()) if labels
            ),
            "Agreement_Class": agreement,
            "Current_Stratum": current_stratum,
            "Source_Evidence": " || ".join(source_evidence),
            "Interpretation": (
                "Agreement measures direct family labels in independently stored source rows. "
                "Silence is not disagreement, and agreement is not necessarily independent "
                "because later dictionaries may repeat earlier analyses."
            ),
        })
    return result


def build_family_attribution_replication_audit(
    audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Separate replication of a word from replication of a family attribution."""
    by_lexeme: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        by_lexeme[row["Lexeme_ID"]].append(row)
    result = []
    for family in ("Korku", "Munda", "Indo-Aryan", "Dravidian"):
        labelled = [
            rows for rows in by_lexeme.values()
            if any(family in row["Stratum"].split("+") for row in rows)
        ]
        replicated = [
            rows for rows in labelled
            if len({row["Lexical_Source"] for row in rows}) >= 2
        ]
        direct_source_counts = []
        for rows in replicated:
            labels_by_source: dict[str, set[str]] = defaultdict(set)
            for row in rows:
                if row["Own_Source_Attribution"]:
                    labels_by_source[row["Lexical_Source"]].update(
                        row["Own_Source_Attribution"].split("+")
                    )
            direct_source_counts.append(sum(
                family in labels for labels in labels_by_source.values()
            ))
        two_plus = sum(count >= 2 for count in direct_source_counts)
        result.append({
            "Family": family,
            "All_Labelled_Clusters": str(len(labelled)),
            "Multi_Source_Clusters": str(len(replicated)),
            "No_Direct_Family_Label": str(sum(count == 0 for count in direct_source_counts)),
            "One_Direct_Labelled_Source": str(sum(count == 1 for count in direct_source_counts)),
            "Two_Plus_Direct_Labelled_Sources": str(two_plus),
            "Two_Plus_Share_Of_Multi_Source": f"{two_plus / len(replicated):.3f}",
            "Interpretation": (
                "Multi-source attestation replicates the Nihali lexeme; only the two-plus column "
                "replicates the family attribution in direct source notes. Even that agreement "
                "may be editorially dependent and is not proof of inheritance."
            ),
        })
    return result


def build_layer_category_audit(
    audit: list[dict[str, str]], cluster_audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Describe concept-category profiles without treating database categories as etymology."""
    form_to_lexeme = {row["Form_ID"]: row["Lexeme_ID"] for row in audit}
    concept_categories = {row["ID"]: row["Category"] for row in read_dicts(CONCEPTS)}
    categories_by_lexeme: dict[str, set[str]] = defaultdict(set)
    for link in read_dicts(FORM_CONCEPTS):
        lexeme_id = form_to_lexeme.get(link["Form_ID"])
        category = concept_categories.get(link["Concept_ID"])
        if lexeme_id and category:
            categories_by_lexeme[lexeme_id].add(category)
    layers = ("Nihali residue", "Korku", "Munda", "Indo-Aryan", "Dravidian")
    result = []
    for layer in layers:
        rows = [
            row for row in cluster_audit
            if (
                row["Stratum"] == layer if layer == "Nihali residue"
                else layer in row["Stratum"].split("+")
            )
        ]
        lexeme_ids = {row["Lexeme_ID"] for row in rows}
        counts = Counter(
            category for lexeme_id in lexeme_ids
            for category in categories_by_lexeme.get(lexeme_id, set())
        )
        result.append({
            "Layer": layer,
            "Total_Clusters": str(len(rows)),
            "Concept_Linked_Clusters": str(sum(
                bool(categories_by_lexeme.get(lexeme_id)) for lexeme_id in lexeme_ids
            )),
            "Noun_Clusters": str(counts["noun"]),
            "Verb_Clusters": str(counts["verb"]),
            "Adjective_Clusters": str(counts["adjective"]),
            "Numeral_Clusters": str(counts["numeral"]),
            "Other_Clusters": str(counts["other"]),
            "Interpretation": (
                "Concept categories are non-exclusive and incompletely linked; the profile "
                "describes lexical-documentation bias and cannot diagnose inheritance by itself."
            ),
        })
    return result


def build_layer_form_shape_audit(
    cluster_audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Compare surface word-shape diagnostics without treating them as family markers."""
    layers = ("Nihali residue", "Korku", "Munda", "Indo-Aryan", "Dravidian")
    result = []
    retroflexes = set("ʈṭɖḍɽṛɳṇ")
    for layer in layers:
        rows = [
            row for row in cluster_audit
            if (
                row["Stratum"] == layer if layer == "Nihali residue"
                else layer in row["Stratum"].split("+")
            )
        ]
        forms = [row["Representative_Form"] for row in rows]
        folded = [fold(form) for form in forms]
        final_vowel = sum(bool(form) and form[-1] in "aeiou" for form in folded)
        compound = sum(bool(re.search(r"[\s-]", form)) for form in forms)
        retroflex = sum(bool(set(form) & retroflexes) for form in forms)
        aspiration = sum(
            "ʰ" in form or bool(re.search(r"[kgcjtdpb]h", fold(form))) for form in forms
        )
        nasalization = sum(
            "\u0303" in unicodedata.normalize("NFD", form) or "ᵑ" in form for form in forms
        )
        total = len(rows)
        lengths = [len(form) for form in folded]
        result.append({
            "Layer": layer,
            "Total_Clusters": str(total),
            "Mean_Folded_Length": f"{statistics.mean(lengths):.3f}",
            "Median_Folded_Length": f"{statistics.median(lengths):.1f}",
            "Final_Vowel_Count": str(final_vowel),
            "Final_Vowel_Share": f"{final_vowel / total:.3f}",
            "Multiword_Or_Compound_Count": str(compound),
            "Multiword_Or_Compound_Share": f"{compound / total:.3f}",
            "Retroflex_Count": str(retroflex),
            "Retroflex_Share": f"{retroflex / total:.3f}",
            "Aspiration_Count": str(aspiration),
            "Aspiration_Share": f"{aspiration / total:.3f}",
            "Nasalization_Count": str(nasalization),
            "Nasalization_Share": f"{nasalization / total:.3f}",
            "Interpretation": (
                "Surface diagnostics reflect Nihali phonological adaptation, transcription, and "
                "morphological packaging. Similarity or difference between layers is not a "
                "genealogical test and cannot identify the residue's ancestry."
            ),
        })
    return result


def build_closed_class_audit(
    audit: list[dict[str, str]], cluster_audit: list[dict[str, str]],
    core_audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Inventory diagnostic closed-class and low-numeral concepts.

    The unit is a concept-to-lexeme link, not a reconstructed root.  This deliberately exposes
    synonymy and conservative under-clustering rather than collapsing it automatically.
    """
    domains = {
        "pronoun": {"I", "HE", "SHE", "WE", "YOU", "THEY"},
        "demonstrative": {"THIS", "THAT"},
        "interrogative": {"WHO", "WHAT", "WHERE", "WHEN", "HOW"},
        "polarity": {"NOT", "NO", "YES"},
        "low numeral": {
            "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
            "NINE", "TEN",
        },
    }
    domain_for_concept = {
        concept: domain for domain, concepts in domains.items() for concept in concepts
    }
    concept_names = {row["ID"]: row["Name"] for row in read_dicts(CONCEPTS)}
    lexeme_for_form = {row["Form_ID"]: row["Lexeme_ID"] for row in audit}
    cluster_by_id = {row["Lexeme_ID"]: row for row in cluster_audit}
    core_by_id = {row["Lexeme_ID"]: row for row in core_audit}
    pairs = set()
    for link in read_dicts(FORM_CONCEPTS):
        concept = concept_names.get(link["Concept_ID"])
        lexeme_id = lexeme_for_form.get(link["Form_ID"])
        if concept in domain_for_concept and lexeme_id:
            pairs.add((concept, lexeme_id))
    result = []
    for concept, lexeme_id in sorted(pairs):
        cluster = cluster_by_id[lexeme_id]
        core = core_by_id.get(lexeme_id)
        effective = (
            core["Sensitivity_Stratum"] if core and core["Sensitivity_Stratum"]
            else cluster["Stratum"]
        )
        result.append({
            "Domain": domain_for_concept[concept],
            "Concept": concept,
            "Lexeme_ID": lexeme_id,
            "Representative_Form": cluster["Representative_Form"],
            "Forms": cluster["Forms"],
            "Glosses": cluster["Glosses"],
            "Record_Count": cluster["Record_Count"],
            "Source_Count": cluster["Source_Count"],
            "Lexical_Sources": cluster["Lexical_Sources"],
            "Strict_Stratum": cluster["Stratum"],
            "Effective_Stratum": effective,
            "Core_Sensitivity_Applied": (
                "yes" if core and core["Sensitivity_Stratum"] else "no"
            ),
            "Methods": cluster["Methods"],
            "Confidence": cluster["Confidence"],
            "Interpretation": (
                "Concept-linked closed-class diagnostic. Multiple rows may be variants or "
                "synonyms; an unmatched form is not thereby evidence of inheritance."
            ),
        })
    return result


def build_core_audit(
    audit: list[dict[str, str]], cluster_audit: list[dict[str, str]]
) -> list[dict[str, str]]:
    concept_names = {row["ID"]: row["Name"] for row in read_dicts(CONCEPTS)}
    missing = CORE_CONCEPTS - set(concept_names.values())
    if missing:
        raise RuntimeError(f"core concept names absent from current concept table: {sorted(missing)}")
    lexeme_for_form = {row["Form_ID"]: row["Lexeme_ID"] for row in audit}
    concepts_for_lexeme: dict[str, set[str]] = defaultdict(set)
    for link in read_dicts(FORM_CONCEPTS):
        lexeme_id = lexeme_for_form.get(link["Form_ID"])
        concept = concept_names.get(link["Concept_ID"])
        if lexeme_id and concept in CORE_CONCEPTS:
            concepts_for_lexeme[lexeme_id].add(concept)
    cluster_by_id = {row["Lexeme_ID"]: row for row in cluster_audit}
    exclusion_rows = read_dicts(CORE_EXCLUSIONS)
    exclusions: dict[tuple[str, str], str] = {}
    for exclusion in exclusion_rows:
        key = (exclusion["Lexeme_ID"], exclusion["Concept"])
        if key in exclusions or not exclusion["Reason"]:
            raise RuntimeError(f"invalid duplicate or blank core exclusion: {key}")
        exclusions[key] = exclusion["Reason"]
    observed_exclusions: set[tuple[str, str]] = set()
    result = []
    for lexeme_id, concepts in sorted(concepts_for_lexeme.items()):
        included_concepts = set()
        for concept in concepts:
            key = (lexeme_id, concept)
            if key in exclusions:
                observed_exclusions.add(key)
            else:
                included_concepts.add(concept)
        if not included_concepts:
            continue
        cluster = cluster_by_id[lexeme_id]
        result.append({
            "Lexeme_ID": lexeme_id,
            "Concepts": "; ".join(sorted(included_concepts)),
            "Representative_Form": cluster["Representative_Form"],
            "Representative_Gloss": cluster["Representative_Gloss"],
            "Record_Count": cluster["Record_Count"],
            "Stratum": cluster["Stratum"],
            "Methods": cluster["Methods"],
            "Confidence": cluster["Confidence"],
            "Lexical_Sources": cluster["Lexical_Sources"],
            "Sensitivity_Stratum": "",
            "Sensitivity_Reference_Lexeme_ID": "",
            "Sensitivity_Form_Similarity": "",
            "Sensitivity_Confidence": "",
            "Sensitivity_Rationale": "",
        })
    if observed_exclusions != set(exclusions):
        raise RuntimeError(
            "core exclusion coverage mismatch; "
            f"missing={sorted(set(exclusions) - observed_exclusions)}, "
            f"extra={sorted(observed_exclusions - set(exclusions))}"
        )

    by_lexeme_id = {row["Lexeme_ID"]: row for row in result}
    external_rows = [
        row for row in result if row["Stratum"] not in {"Nihali residue", "Other", ""}
    ]
    sensitivity_candidates: dict[str, tuple[str, float]] = {}
    for row in result:
        if row["Stratum"] != "Nihali residue":
            continue
        concepts = set(row["Concepts"].split("; "))
        matches = []
        for external in external_rows:
            if not concepts & set(external["Concepts"].split("; ")):
                continue
            similarity = form_similarity(
                form_variants(row["Representative_Form"]),
                form_variants(external["Representative_Form"]),
            )
            if similarity >= 0.65:
                matches.append((similarity, external["Lexeme_ID"]))
        if matches:
            similarity, reference_id = max(matches, key=lambda item: item[0])
            sensitivity_candidates[row["Lexeme_ID"]] = (reference_id, similarity)

    review_rows = read_dicts(CORE_VARIANT_REVIEW)
    reviews: dict[str, dict[str, str]] = {}
    for review in review_rows:
        lexeme_id = review["Lexeme_ID"]
        if lexeme_id in reviews or review["Confidence"] not in {"high", "medium", "low"}:
            raise RuntimeError(f"invalid duplicate or confidence in core variant review: {lexeme_id}")
        if not review["Rationale"]:
            raise RuntimeError(f"core variant review lacks rationale: {lexeme_id}")
        reviews[lexeme_id] = review
    # The mechanical candidate set is exhaustive for same-concept pairs.  The review register
    # may additionally record transparent derivational/inflectional families whose English
    # concept labels differ (FULL ~ FILL, SAY ~ SPEAK, BIRD ~ FEATHER).  Such rows remain a
    # sensitivity analysis: they do not create graph edges or promote the donor claim itself.
    if not set(sensitivity_candidates) <= set(reviews):
        raise RuntimeError(
            "core variant sensitivity coverage mismatch; "
            f"missing={sorted(set(sensitivity_candidates) - set(reviews))}"
        )
    for lexeme_id, review in reviews.items():
        row = by_lexeme_id.get(lexeme_id)
        reference_id = review["Reference_Lexeme_ID"]
        reference = cluster_by_id.get(reference_id)
        if not row or row["Stratum"] != "Nihali residue":
            raise RuntimeError(f"core variant target is not residual core: {lexeme_id}")
        if not reference or reference["Stratum"] in {"Nihali residue", "Other", ""}:
            raise RuntimeError(
                f"core variant reference is not externally attributed: {lexeme_id} -> "
                f"{reference_id}"
            )
        similarity = form_similarity(
            form_variants(row["Representative_Form"]),
            form_variants(reference["Representative_Form"]),
        )
        if lexeme_id in sensitivity_candidates:
            expected_reference, expected_similarity = sensitivity_candidates[lexeme_id]
        else:
            expected_reference, expected_similarity = reference_id, similarity
            # A transparent inflectional extension can halve the whole-form score (poy + -ṭa),
            # so explicit cross-concept rows use a lower floor than mechanical discovery.
            if similarity < 0.50:
                raise RuntimeError(
                    f"manual cross-concept core variant is too form-distant: {lexeme_id} -> "
                    f"{reference_id} ({similarity:.3f})"
                )
        if reference_id != expected_reference:
            raise RuntimeError(
                f"core variant reference changed for {lexeme_id}: "
                f"expected {expected_reference}, reviewed {reference_id}"
            )
        row["Sensitivity_Stratum"] = reference["Stratum"]
        row["Sensitivity_Reference_Lexeme_ID"] = reference_id
        row["Sensitivity_Form_Similarity"] = f"{expected_similarity:.3f}"
        row["Sensitivity_Confidence"] = review["Confidence"]
        row["Sensitivity_Rationale"] = review["Rationale"]
    return result


def build_core_residue_root_audit(core_audit: list[dict[str, str]]) -> list[dict[str, str]]:
    """Collapse the post-sensitivity core residue to manually reviewed lexical roots.

    The conservative all-lexicon clusterer intentionally keeps citation forms and close dialect
    variants apart.  That is safe for graph construction but inflates a Swadesh-style diagnostic
    slice.  Here, only rows still classed as residue after the external-variant sensitivity pass
    are grouped, concept by concept.  Multi-cluster concepts require an explicit review row;
    singletons are carried through automatically.
    """
    clusters_by_concept: dict[str, set[str]] = defaultdict(set)
    for row in core_audit:
        if (row["Sensitivity_Stratum"] or row["Stratum"]) != "Nihali residue":
            continue
        for concept in row["Concepts"].split("; "):
            clusters_by_concept[concept].add(row["Lexeme_ID"])
    review_rows = read_dicts(CORE_RESIDUE_ROOT_REVIEW)
    reviews: dict[str, dict[str, str]] = {}
    for review in review_rows:
        concept = review["Concept"]
        if concept in reviews or review["Confidence"] not in {"high", "medium", "low"}:
            raise RuntimeError(f"invalid core residue root review: {concept}")
        if not review["Rationale"]:
            raise RuntimeError(f"core residue root review lacks rationale: {concept}")
        reviews[concept] = review
    expected_reviewed = {
        concept for concept, cluster_ids in clusters_by_concept.items() if len(cluster_ids) > 1
    }
    if set(reviews) != expected_reviewed:
        raise RuntimeError(
            "core residue root review coverage mismatch; "
            f"missing={sorted(expected_reviewed - set(reviews))}, "
            f"extra={sorted(set(reviews) - expected_reviewed)}"
        )
    result = []
    for concept, cluster_ids in sorted(clusters_by_concept.items()):
        if len(cluster_ids) == 1:
            groups = next(iter(cluster_ids))
            root_count = 1
            confidence = "high"
            rationale = "Only one lexeme cluster remains in the effective residue for this concept."
        else:
            review = reviews[concept]
            groups = review["Groups"]
            parsed_groups = [
                [item.strip() for item in group.split("+") if item.strip()]
                for group in groups.split(";") if group.strip()
            ]
            flattened = [item for group in parsed_groups for item in group]
            if (
                set(flattened) != cluster_ids or len(flattened) != len(set(flattened))
                or int(review["Cluster_Count"]) != len(cluster_ids)
                or int(review["Root_Group_Count"]) != len(parsed_groups)
            ):
                raise RuntimeError(
                    f"core residue root grouping mismatch for {concept}: "
                    f"expected={sorted(cluster_ids)}, reviewed={flattened}"
                )
            root_count = len(parsed_groups)
            confidence = review["Confidence"]
            rationale = review["Rationale"]
        result.append({
            "Concept": concept,
            "Cluster_Count": str(len(cluster_ids)),
            "Root_Group_Count": str(root_count),
            "Groups": groups,
            "Confidence": confidence,
            "Rationale": rationale,
        })
    return result


def build_core_residue_root_inventory(
    root_audit: list[dict[str, str]], cluster_audit: list[dict[str, str]],
    audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Expand reviewed core-residue groups into a human-readable root inventory.

    Root grouping is a citation-form denominator correction, not a historical reconstruction.
    Source replication is reported separately because repeated documentation establishes that a
    form is real and stable, but neither that it is inherited nor that it is free of old loans.
    """
    cluster_by_id = {row["Lexeme_ID"]: row for row in cluster_audit}
    audit_by_lexeme: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        audit_by_lexeme[row["Lexeme_ID"]].append(row)
    early_sources = {"konow1906", "bhattacharya1957"}
    result = []
    for concept_row in root_audit:
        parsed_groups = [
            [item.strip() for item in group.split("+") if item.strip()]
            for group in concept_row["Groups"].split(";") if group.strip()
        ]
        for root_number, cluster_ids in enumerate(parsed_groups, start=1):
            clusters = [cluster_by_id[lexeme_id] for lexeme_id in cluster_ids]
            records = [
                row for lexeme_id in cluster_ids for row in audit_by_lexeme[lexeme_id]
            ]
            sources = sorted({row["Lexical_Source"] for row in records})
            forms = sorted({row["Form"] for row in records})
            glosses = sorted({row["Gloss"] for row in records})
            representative = min(
                records, key=lambda row: (len(fold(row["Form"])), row["Form_ID"])
            )
            if len(sources) >= 4:
                replication_grade = "very strong"
            elif len(sources) >= 3:
                replication_grade = "strong"
            elif len(sources) >= 2:
                replication_grade = "moderate"
            else:
                replication_grade = "single-source"
            result.append({
                "Root_ID": (
                    "nihcore-"
                    + re.sub(r"[^a-z0-9]+", "-", concept_row["Concept"].lower()).strip("-")
                    + f"-{root_number}"
                ),
                "Concept": concept_row["Concept"],
                "Root_Number": str(root_number),
                "Cluster_Count": str(len(cluster_ids)),
                "Cluster_IDs": "; ".join(cluster_ids),
                "Representative_Form": representative["Form"],
                "Forms": "; ".join(forms),
                "Glosses": "; ".join(glosses),
                "Record_Count": str(len(records)),
                "Source_Count": str(len(sources)),
                "Lexical_Sources": "; ".join(sources),
                "Early_Source_Attested": (
                    "yes" if early_sources & set(sources) else "no"
                ),
                "Replication_Grade": replication_grade,
                "Grouping_Confidence": concept_row["Confidence"],
                "Grouping_Rationale": concept_row["Rationale"],
                "Methods": "; ".join(sorted({
                    method for cluster in clusters
                    for method in cluster["Methods"].split("; ") if method
                })),
                "Closest_Alternatives": " | ".join(sorted({
                    row["Alternatives"] for row in records if row["Alternatives"]
                })),
                "Interpretation": (
                    "Effective basic-vocabulary residue root hypothesis. Its lack of a resolved "
                    "external assignment is not evidence that it is inherited or uniquely Nihali."
                ),
            })
    expected = sum(int(row["Root_Group_Count"]) for row in root_audit)
    if len(result) != expected or len({row["Root_ID"] for row in result}) != expected:
        raise RuntimeError("core residue root inventory does not match reviewed root count")
    return result


def build_core_concept_profile_audit(
    core_audit: list[dict[str, str]],
    core_residue_root_inventory: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Count each diagnostic basic concept once after reviewed variant sensitivity."""
    by_concept: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in core_audit:
        for concept in row["Concepts"].split("; "):
            by_concept[concept].append(row)
    roots_by_concept: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in core_residue_root_inventory:
        roots_by_concept[row["Concept"]].append(row)
    replication_rank = {
        "single-source": 0, "moderate": 1, "strong": 2, "very strong": 3,
    }
    result = []
    for concept, rows in sorted(by_concept.items()):
        effective_strata = {
            row["Sensitivity_Stratum"] or row["Stratum"] for row in rows
        }
        contact_families = {
            part for stratum in effective_strata for part in stratum.split("+")
            if part not in {"Nihali residue", "Other", ""}
        }
        residue_present = "Nihali residue" in effective_strata
        if residue_present and not contact_families:
            profile = "residue-only"
        elif residue_present:
            profile = "residue-plus-contact"
        elif len(contact_families) == 1:
            profile = "contact-only-single-family"
        else:
            profile = "contact-only-mixed-family"
        concept_roots = roots_by_concept.get(concept, [])
        best_replication = (
            max(
                (root["Replication_Grade"] for root in concept_roots),
                key=lambda grade: replication_rank[grade],
            )
            if concept_roots else ""
        )
        result.append({
            "Concept": concept,
            "Lexeme_Cluster_Count": str(len(rows)),
            "Lexeme_IDs": "; ".join(sorted({row["Lexeme_ID"] for row in rows})),
            "Representative_Forms": "; ".join(sorted({
                row["Representative_Form"] for row in rows
            })),
            "Strict_Strata": "; ".join(sorted({row["Stratum"] for row in rows})),
            "Effective_Strata": "; ".join(sorted(effective_strata)),
            "Residue_Present": "yes" if residue_present else "no",
            "Residual_Root_Hypotheses": str(len(concept_roots)),
            "Best_Residual_Replication": best_replication,
            "Any_Multi_Source_Residual_Root": (
                "yes" if any(
                    root["Replication_Grade"] != "single-source" for root in concept_roots
                ) else "no"
            ),
            "Early_Residual_Root": (
                "yes" if any(
                    root["Early_Source_Attested"] == "yes" for root in concept_roots
                ) else "no"
            ),
            "Contact_Families": "+".join(ordered_strata(contact_families)),
            "Profile_Class": profile,
            "Interpretation": (
                "Concept-level profile after reviewed variant propagation. Multiple synonyms are "
                "shown but the profile counts the concept once; residue means unmatched, not "
                "demonstrated inheritance, and a contact label does not establish direction."
            ),
        })
    return result


def build_replicated_residue_audit(
    audit: list[dict[str, str]], core_audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Inventory residue clusters independently recorded in two or more lexical sources."""
    core_by_id = {row["Lexeme_ID"]: row for row in core_audit}
    by_lexeme: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        by_lexeme[row["Lexeme_ID"]].append(row)
    result = []
    for lexeme_id, group in sorted(by_lexeme.items()):
        source_names = sorted({row["Lexical_Source"] for row in group})
        if len(source_names) < 2 or any(row["Stratum"] != "Nihali residue" for row in group):
            continue
        representative = sorted(
            group, key=lambda row: (len(row["Form"]), row["Form_ID"])
        )[0]
        form_sets = [form_variants(row["Form"]) for row in group]
        glosses = [row["Gloss"] for row in group]
        form_pair_scores = [
            form_similarity(form_sets[i], form_sets[j])
            for i in range(len(group)) for j in range(i + 1, len(group))
        ]
        gloss_pair_scores = [
            gloss_similarity(glosses[i], glosses[j])
            for i in range(len(group)) for j in range(i + 1, len(group))
        ]
        minimum_form = min(form_pair_scores, default=1.0)
        minimum_gloss = min(gloss_pair_scores, default=1.0)
        core = core_by_id.get(lexeme_id)
        effective = bool(
            core and (core["Sensitivity_Stratum"] or core["Stratum"]) == "Nihali residue"
        )
        if len(source_names) >= 4 and minimum_form >= 0.75 and minimum_gloss >= 0.55:
            replication_grade = "very strong"
        elif len(source_names) >= 3 and minimum_form >= 0.70 and minimum_gloss >= 0.45:
            replication_grade = "strong"
        else:
            replication_grade = "moderate"
        if core and not effective:
            interpretation = (
                "Cross-source Nihali item, but the core sensitivity review links it to a "
                "separately clustered external-source variant."
            )
        elif core:
            interpretation = (
                "Cross-source core item still residual after external-variant sensitivity; "
                "replication establishes lexical reality, not inheritance."
            )
        else:
            interpretation = (
                "Cross-source Nihali item; replication establishes lexical reality, not "
                "inheritance or freedom from old borrowing."
            )
        result.append({
            "Lexeme_ID": lexeme_id,
            "Representative_Form": representative["Form"],
            "Representative_Gloss": representative["Gloss"],
            "Record_Count": str(len(group)),
            "Source_Count": str(len(source_names)),
            "Lexical_Sources": "; ".join(source_names),
            "Forms": "; ".join(sorted({row["Form"] for row in group})),
            "Glosses": "; ".join(sorted({row["Gloss"] for row in group})),
            "Minimum_Form_Similarity": f"{minimum_form:.3f}",
            "Minimum_Gloss_Similarity": f"{minimum_gloss:.3f}",
            "Core_Concepts": core["Concepts"] if core else "",
            "Core_Effective_Residue": "yes" if effective else "no",
            "Core_Sensitivity_Stratum": (
                core["Sensitivity_Stratum"] if core and core["Sensitivity_Stratum"] else ""
            ),
            "Methods": "; ".join(sorted({row["Method"] for row in group})),
            "Manual_Decisions": "; ".join(sorted({
                row["Manual_Decision"] for row in group if row["Manual_Decision"]
            })),
            "Closest_Alternatives": next(
                (row["Alternatives"] for row in group if row["Alternatives"]), ""
            ),
            "Replication_Grade": replication_grade,
            "Interpretation": interpretation,
        })
    return result


def build_resolved_contact_shape_audit(
    audit: list[dict[str, str]], languages: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Describe form shape for links that resolve to a non-Nihali database parent.

    This is deliberately not an automatic sound-law detector.  Many targets are reconstructed
    proto-forms rather than surface donors, while near-identical pairs are often exactly what
    recent borrowing predicts.  The table exists to make those distributions inspectable and to
    prevent a handful of attractive matches from masquerading as a correspondence system.
    """
    review_rows = read_dicts(MUNDA_CORRESPONDENCE_REVIEW)
    all_forms = read_dicts(FORMS)
    form_by_id = {row["ID"]: row for row in all_forms}
    edge_rows = read_dicts(EDGES)
    rank1 = {
        row["Child_ID"]: row for row in edge_rows
        if row["Rank"] == "1" and row["Kind"] in {"reflex", "borrowed", "variant"}
    }
    provisional_children = {
        row["Form_ID"] for row in overlay.read_assignments()
        if row.get("Notes", "").startswith(ASSIGNMENT_MARKER)
    }
    curated_rank1 = {
        child_id: row for child_id, row in rank1.items()
        if child_id not in provisional_children
    }
    generic_language_ids = {
        "Indo-Aryan", "Sk", "OIA", "MIA", "Drav", "PDr", "PSTDr", "PSD1",
        "PSD2", "PCDr", "PKMDr", "PNDr", "PMu", "PKher", "Eng",
    }
    surfaces_by_parent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for form in all_forms:
        if (
            form["Language_ID"] == "Ni" or form["Language_ID"] in generic_language_ids
            or form.get("Status") in {"entry", "unlinked"}
            or REFERENCE in form.get("Source", "")
        ):
            continue
        # Index an observed form below every node on its accepted rank-1 path, not just below the
        # ultimate ancestor.  Many CDIAL headwords are themselves linked to Proto-Indo-Iranian;
        # indexing only ``effective_parent`` therefore made their abundant modern reflexes look
        # unavailable whenever a Nihali assignment stopped (appropriately) at the CDIAL node.
        current_id = form["ID"]
        seen = set()
        indexed_parents = set()
        while current_id in rank1 and current_id not in seen:
            seen.add(current_id)
            current_id = rank1[current_id]["Parent_ID"]
            if current_id not in indexed_parents:
                surfaces_by_parent[current_id].append(form)
                indexed_parents.add(current_id)
    reviews: dict[tuple[str, str], dict[str, str]] = {}
    for review in review_rows:
        key = (review["Lexeme_ID"], review["Parent_ID"])
        if key in reviews or review["Assessment"] not in {
            "near-contact-compatible", "possible-correspondence", "weak-comparison",
        }:
            raise RuntimeError(f"invalid Munda correspondence review: {key}")
        if review["Series"] not in {"identity-or-near", "c~s", "other", "none"}:
            raise RuntimeError(f"invalid Munda correspondence series: {key}")
        if review["Confidence"] not in {"high", "medium", "low"} or not review["Rationale"]:
            raise RuntimeError(f"incomplete Munda correspondence review: {key}")
        reviews[key] = review
    link_rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in audit:
        resolved_row = dict(row)
        immediate_parent_id = row["Parent_ID"]
        resolution_path = [immediate_parent_id]
        if row["Parent_Language_ID"] == "Ni":
            terminal_id = immediate_parent_id
            seen = set()
            while terminal_id in curated_rank1 and terminal_id not in seen:
                seen.add(terminal_id)
                terminal_id = curated_rank1[terminal_id]["Parent_ID"]
                resolution_path.append(terminal_id)
            terminal = form_by_id.get(terminal_id)
            if (
                not terminal or terminal_id.startswith("nihprov-")
                or terminal["Language_ID"] == "Ni"
            ):
                continue
            resolved_row.update({
                "Parent_ID": terminal_id,
                "Parent_Form": terminal["Form"],
                "Parent_Language_ID": terminal["Language_ID"],
                "Parent_Language": languages.get(
                    terminal["Language_ID"], {}
                ).get("Name", terminal["Language_ID"]),
                "Method": "curated-internal-chain",
            })
        elif immediate_parent_id.startswith("nihprov-"):
            continue
        resolved_row["Immediate_Parent_ID"] = immediate_parent_id
        resolved_row["Immediate_Parent_Form"] = row["Parent_Form"]
        resolved_row["Resolution_Path"] = " > ".join(resolution_path)
        key = (resolved_row["Lexeme_ID"], resolved_row["Parent_ID"])
        link_rows.setdefault(key, resolved_row)
    parent_counts = Counter(row["Parent_ID"] for row in link_rows.values())
    result = []
    for row in sorted(link_rows.values(), key=lambda item: (item["Lexeme_ID"], item["Parent_ID"])):
        similarity = form_similarity(
            form_variants(row["Form"]), form_variants(row["Parent_Form"])
        )
        if similarity == 1.0:
            shape = "exact"
        elif similarity >= 0.80:
            shape = "near"
        elif similarity >= 0.60:
            shape = "moderate"
        else:
            shape = "distant"
        child_fold, parent_fold = fold(row["Form"]), fold(row["Parent_Form"])
        parent_family = language_family(row["Parent_Language_ID"], languages)
        review = reviews.get((row["Lexeme_ID"], row["Parent_ID"]))
        surface_options = []
        route_families = set(row["Stratum"].split("+"))
        for surface in surfaces_by_parent.get(row["Parent_ID"], []):
            surface_form_similarity = form_similarity(
                form_variants(row["Form"]),
                form_variants(surface["Form"] or surface["Original"]),
            )
            surface_gloss_similarity = gloss_similarity(row["Gloss"], surface["Gloss"])
            surface_family = language_family(surface["Language_ID"], languages)
            route_bonus = 0.05 if surface_family in route_families else 0.0
            surface_options.append((
                0.65 * surface_form_similarity + 0.35 * surface_gloss_similarity + route_bonus,
                surface_form_similarity, surface_gloss_similarity, surface,
            ))
        # Some terminal entries are themselves attested donor lexemes rather than reconstructed
        # heads.  Treat those as observed surfaces when they have no indexed descendant (notably
        # direct English cultural loans), while keeping generic/proto entry nodes excluded.
        direct_parent = form_by_id.get(row["Parent_ID"])
        if (
            direct_parent
            and direct_parent.get("Status") == "entry"
            and direct_parent.get("Language_ID") in {
                "Eng", "Pers", "Port", "Ar", "H", "M", "B", "Ko"
            }
            and REFERENCE not in direct_parent.get("Source", "")
        ):
            direct_form_similarity = form_similarity(
                form_variants(row["Form"]),
                form_variants(direct_parent["Form"] or direct_parent["Original"]),
            )
            direct_gloss_similarity = gloss_similarity(row["Gloss"], direct_parent["Gloss"])
            direct_family = language_family(direct_parent["Language_ID"], languages)
            direct_route_bonus = 0.05 if direct_family in route_families else 0.0
            surface_options.append((
                0.65 * direct_form_similarity + 0.35 * direct_gloss_similarity
                + direct_route_bonus,
                direct_form_similarity, direct_gloss_similarity, direct_parent,
            ))
        # Exact spelling can otherwise select a semantically unrelated descendant over a
        # slightly less similar form that actually preserves the Nihali meaning.  When the
        # etymon has any descendant with minimally compatible semantics, rank only those
        # candidates; retain the unrestricted fallback for poorly glossed etyma.
        preferred_surface_id = PREFERRED_CONTACT_SURFACE_BY_LEXEME.get(row["Lexeme_ID"])
        preferred_surfaces = [
            option for option in surface_options
            if option[3].get("ID") == preferred_surface_id
        ]
        if preferred_surface_id and not preferred_surfaces:
            raise RuntimeError(
                f"preferred contact surface {preferred_surface_id} unavailable for "
                f"{row['Lexeme_ID']} / {row['Parent_ID']}"
            )
        meaning_compatible_surfaces = [
            option for option in surface_options if option[2] >= 0.30
        ]
        ranked_surfaces = (
            preferred_surfaces or meaning_compatible_surfaces or surface_options
        )
        best_surface = max(ranked_surfaces, default=None, key=lambda item: item[:3])
        if best_surface:
            _, surface_form_similarity, surface_gloss_similarity, surface = best_surface
            surface_family = language_family(surface["Language_ID"], languages)
            if surface_form_similarity == 1.0:
                surface_shape = "exact"
            elif surface_form_similarity >= 0.80:
                surface_shape = "near"
            elif surface_form_similarity >= 0.60:
                surface_shape = "moderate"
            else:
                surface_shape = "distant"
        else:
            surface_form_similarity = surface_gloss_similarity = 0.0
            surface = {}
            surface_family = surface_shape = ""
        result.append({
            "Lexeme_ID": row["Lexeme_ID"],
            "Child_Form": row["Form"],
            "Child_Gloss": row["Gloss"],
            "Immediate_Parent_ID": row["Immediate_Parent_ID"],
            "Immediate_Parent_Form": row["Immediate_Parent_Form"],
            "Resolution_Path": row["Resolution_Path"],
            "Parent_ID": row["Parent_ID"],
            "Parent_Form": row["Parent_Form"],
            "Parent_Gloss": form_by_id.get(row["Parent_ID"], {}).get("Gloss", ""),
            "Parent_Language_ID": row["Parent_Language_ID"],
            "Parent_Language": row["Parent_Language"],
            "Parent_Family": parent_family,
            "Method": row["Method"],
            "Confidence": row["Confidence"],
            "Source_Stratum": row["Stratum"],
            "Form_Similarity": f"{similarity:.3f}",
            "Match_Shape": shape,
            "Initial_Correspondence": (
                f"{child_fold[0]}~{parent_fold[0]}" if child_fold and parent_fold else ""
            ),
            "Final_Correspondence": (
                f"{child_fold[-1]}~{parent_fold[-1]}" if child_fold and parent_fold else ""
            ),
            "Matched_Surface_ID": surface.get("ID", ""),
            "Matched_Surface_Form": surface.get("Form", ""),
            "Matched_Surface_Gloss": surface.get("Gloss", ""),
            "Matched_Surface_Language_ID": surface.get("Language_ID", ""),
            "Matched_Surface_Language": languages.get(
                surface.get("Language_ID", ""), {}
            ).get("Name", surface.get("Language_ID", "")),
            "Matched_Surface_Family": surface_family,
            "Surface_Form_Similarity": (
                f"{surface_form_similarity:.3f}" if best_surface else ""
            ),
            "Surface_Gloss_Similarity": (
                f"{surface_gloss_similarity:.3f}" if best_surface else ""
            ),
            "Surface_Match_Shape": surface_shape,
            "Parent_Link_Count": str(parent_counts[row["Parent_ID"]]),
            "Review_Assessment": review["Assessment"] if review else "",
            "Correspondence_Series": review["Series"] if review else "",
            "Review_Confidence": review["Confidence"] if review else "",
            "Review_Rationale": review["Rationale"] if review else "",
            "Interpretive_Caution": (
                "Parent shape and best observed descendant surface are descriptive only. Surface "
                "selection is automated and cannot establish direction, date, or inheritance."
            ),
        })
    munda_keys = {
        (row["Lexeme_ID"], row["Parent_ID"]) for row in result
        if row["Parent_Family"] == "Munda"
    }
    if munda_keys != set(reviews):
        raise RuntimeError(
            "Munda correspondence review coverage mismatch; "
            f"missing={sorted(munda_keys - set(reviews))}, "
            f"extra={sorted(set(reviews) - munda_keys)}"
        )
    return result


def build_dravidian_correspondence_audit(
    resolved_contact_shape_audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Collapse resolved Dravidian links to parent roots and grade their visible shape.

    This is a reproducible diagnostic, not a cognate classifier.  It intentionally keeps semantic
    weakness separate from phonological distance, and it counts proposed initial mappings across
    parent roots rather than repeated dictionary citations.  Identity-heavy matches are especially
    compatible with borrowing and must not be mistaken for an inherited correspondence system.
    """
    by_parent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in resolved_contact_shape_audit:
        if row["Parent_Family"] == "Dravidian":
            by_parent[row["Parent_ID"]].append(row)

    representatives: dict[str, dict[str, str]] = {}
    assessments: dict[str, str] = {}
    for parent_id, rows in by_parent.items():
        # Prefer the link with the strongest joint surface-form and semantic fit.  The surface was
        # selected independently in the resolved-link audit and may itself be a known borrowing.
        representative = max(
            rows,
            key=lambda row: (
                min(
                    float(row["Surface_Form_Similarity"]),
                    float(row["Surface_Gloss_Similarity"]),
                ),
                0.65 * float(row["Surface_Form_Similarity"])
                + 0.35 * float(row["Surface_Gloss_Similarity"]),
                row["Lexeme_ID"],
            ),
        )
        form_score = float(representative["Surface_Form_Similarity"])
        gloss_score = float(representative["Surface_Gloss_Similarity"])
        if form_score >= 0.80 and gloss_score >= 0.70:
            assessment = "near-contact-compatible"
        elif form_score >= 0.60 and gloss_score >= 0.70:
            assessment = "possible-comparison"
        elif gloss_score < 0.40:
            assessment = "weak-semantic-link"
        else:
            assessment = "weak-form-link"
        representatives[parent_id] = representative
        assessments[parent_id] = assessment

    series_counts = Counter(
        row["Initial_Correspondence"] for row in representatives.values()
    )
    result = []
    for parent_id, rows in sorted(by_parent.items()):
        best = representatives[parent_id]
        initial = best["Initial_Correspondence"]
        left, _, right = initial.partition("~")
        if left and left == right:
            series_type = "identity-initial"
        elif series_counts[initial] >= 2:
            series_type = "repeated-nonidentity"
        else:
            series_type = "singleton-nonidentity"
        assessment = assessments[parent_id]
        if assessment == "near-contact-compatible":
            basis = (
                "The best observed descendant has close form and meaning. This supports a real "
                "historical connection but is at least as compatible with borrowing as inheritance."
            )
        elif assessment == "possible-comparison":
            basis = (
                "The best observed descendant has compatible meaning and moderate form overlap; "
                "a historical comparison is plausible but requires independent correspondences."
            )
        elif assessment == "weak-semantic-link":
            basis = (
                "Even the best automatically recovered descendant has weak semantic fit, so this "
                "resolved node should not carry genealogical weight without manual reanalysis."
            )
        else:
            basis = (
                "The meaning is at least compatible, but the best observed descendant remains "
                "formally distant; no regular change is established by this root alone."
            )
        result.append({
            "Parent_ID": parent_id,
            "Parent_Form": best["Parent_Form"],
            "Parent_Gloss": best["Parent_Gloss"],
            "Link_Count": str(len(rows)),
            "Lexeme_Count": str(len({row["Lexeme_ID"] for row in rows})),
            "Lexeme_IDs": "; ".join(sorted({row["Lexeme_ID"] for row in rows})),
            "Child_Forms": "; ".join(sorted({row["Child_Form"] for row in rows})),
            "Child_Glosses": "; ".join(sorted({row["Child_Gloss"] for row in rows})),
            "Source_Strata": "; ".join(sorted({row["Source_Stratum"] for row in rows})),
            "Best_Surface_ID": best["Matched_Surface_ID"],
            "Best_Surface_Form": best["Matched_Surface_Form"],
            "Best_Surface_Gloss": best["Matched_Surface_Gloss"],
            "Best_Surface_Language": best["Matched_Surface_Language"],
            "Best_Surface_Family": best["Matched_Surface_Family"],
            "Best_Surface_Form_Similarity": best["Surface_Form_Similarity"],
            "Best_Surface_Gloss_Similarity": best["Surface_Gloss_Similarity"],
            "Assessment": assessment,
            "Initial_Correspondence": initial,
            "Initial_Series_Root_Count": str(series_counts[initial]),
            "Series_Type": series_type,
            "Interpretation": (
                f"{basis} The {initial or 'unscored'} initial pattern occurs in "
                f"{series_counts[initial]} distinct reviewed parent root(s). This descriptive "
                "count is not a sound law or evidence of genetic inheritance."
            ),
        })
    return result


def build_source_variation_audit(audit: list[dict[str, str]]) -> list[dict[str, str]]:
    review_rows = read_dicts(DISJOINT_SOURCE_REVIEW)
    reviews: dict[str, dict[str, str]] = {}
    for review in review_rows:
        lexeme_id = review["Lexeme_ID"]
        if lexeme_id in reviews:
            raise RuntimeError(f"duplicate disjoint-source review for {lexeme_id}")
        if review["Assessment"] not in {
            "compatible-contact-chain", "favor-indo-aryan", "favor-korku",
            "favor-munda", "favor-dravidian", "unresolved",
        }:
            raise RuntimeError(f"invalid disjoint-source assessment for {lexeme_id}")
        if review["Confidence"] not in {"high", "medium", "low"} or not review["Rationale"]:
            raise RuntimeError(f"incomplete disjoint-source review for {lexeme_id}")
        reviews[lexeme_id] = review
    by_lexeme: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        by_lexeme[row["Lexeme_ID"]].append(row)
    result = []
    for lexeme_id, group in sorted(by_lexeme.items()):
        labels = sorted({
            row["Own_Source_Attribution"] for row in group if row["Own_Source_Attribution"]
        })
        if len(labels) < 2:
            continue
        label_sets = [set(label.split("+")) for label in labels]
        nested = any(
            left < right or right < left
            for index, left in enumerate(label_sets)
            for right in label_sets[index + 1:]
        )
        if nested:
            relationship = "nested-route/ultimate"
            priority = "medium"
        elif set.intersection(*label_sets):
            relationship = "overlapping"
            priority = "medium"
        else:
            relationship = "disjoint"
            priority = "high"
        review = reviews.get(lexeme_id)
        if relationship == "disjoint" and not review:
            raise RuntimeError(f"disjoint source-label case lacks manual review: {lexeme_id}")
        if relationship != "disjoint" and review:
            raise RuntimeError(f"disjoint-source review no longer targets a disjoint case: {lexeme_id}")
        representative = sorted(group, key=lambda row: (len(row["Form"]), row["Form_ID"]))[0]
        evidence = sorted({
            f"{row['Lexical_Source']}: {row['Original_Etymology']}"
            for row in group if row["Own_Source_Attribution"] and row["Original_Etymology"]
        })
        result.append({
            "Lexeme_ID": lexeme_id,
            "Representative_Form": representative["Form"],
            "Representative_Gloss": representative["Gloss"],
            "Record_Count": str(len(group)),
            "Relationship": relationship,
            "Own_Source_Attributions": "; ".join(labels),
            "Lexical_Sources": "; ".join(sorted({row["Lexical_Source"] for row in group})),
            "Source_Evidence": " || ".join(evidence),
            "Current_Stratum": "+".join(ordered_strata({
                part for row in group for part in row["Stratum"].split("+")
                if part not in {"Nihali residue", "Other", ""}
            })) or ("Other" if any(row["Stratum"] == "Other" for row in group) else "Nihali residue"),
            "Current_Parent_IDs": "; ".join(sorted({row["Parent_ID"] for row in group})),
            "Review_Priority": priority,
            "Review_Assessment": review["Assessment"] if review else "",
            "Preferred_Immediate_Donor": review["Preferred_Immediate_Donor"] if review else "",
            "Preferred_Ultimate_Source": review["Preferred_Ultimate_Source"] if review else "",
            "Review_Confidence": review["Confidence"] if review else "",
            "Review_Rationale": review["Rationale"] if review else "",
        })
    reviewed_ids = {row["Lexeme_ID"] for row in result if row["Relationship"] == "disjoint"}
    if reviewed_ids != set(reviews):
        raise RuntimeError(
            "disjoint-source review coverage mismatch; "
            f"missing={sorted(reviewed_ids - set(reviews))}, "
            f"extra={sorted(set(reviews) - reviewed_ids)}"
        )
    return result


def build_source_proxy_quality_audit(
    audit: list[dict[str, str]], core_audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Grade the reproducibility of every unresolved donor-proxy claim.

    A printed family label is useful evidence, but it is not equivalent to a recoverable donor
    form or a numbered comparative-dictionary entry.  This audit keeps those evidential levels
    separate without silently deleting the source's provisional attribution from the graph.
    Uncertainty and directionality are orthogonal fields: a note can name an exact comparison yet
    still question whether it is a loan, or can assert a loan while giving no particular form.
    """
    core_review_rows = read_dicts(CORE_SOURCE_PROXY_REVIEW)
    diagnostic_review_rows = read_dicts(DIAGNOSTIC_SOURCE_PROXY_REVIEW)
    reviews: dict[str, dict[str, str]] = {}
    for review in [*core_review_rows, *diagnostic_review_rows]:
        lexeme_id = review["Lexeme_ID"]
        if lexeme_id in reviews or review["Assessment"] not in {
            "corroborated-contact", "route-ambiguous-contact", "plausible-contact",
            "weak-comparison", "unresolved",
        }:
            raise RuntimeError(f"invalid source-proxy review: {lexeme_id}")
        if review["Confidence"] not in {"high", "medium", "low"} or not review["Rationale"]:
            raise RuntimeError(f"incomplete source-proxy review: {lexeme_id}")
        reviews[lexeme_id] = review
    preexisting_review_ids = set(reviews)
    questioned_decisions = {
        lexeme_id: assessment
        for assessment, lexeme_ids in QUESTIONED_PROXY_REVIEW_IDS.items()
        for lexeme_id in lexeme_ids
    }
    if len(questioned_decisions) != sum(map(len, QUESTIONED_PROXY_REVIEW_IDS.values())):
        raise RuntimeError("questioned source-proxy review assigns a lexeme more than once")
    if set(questioned_decisions) & preexisting_review_ids:
        raise RuntimeError("questioned source-proxy review overlaps an earlier diagnostic review")
    by_lexeme: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        by_lexeme[row["Lexeme_ID"]].append(row)
    core_ids = {row["Lexeme_ID"] for row in core_audit}
    uncertainty_re = re.compile(
        r"\?|\b(?:possibly|perhaps|probably|probable|uncertain|maybe|may be|tentative)\b",
        re.I,
    )
    direction_re = re.compile(r"\b(?:loan|borrow(?:ed|ing)?|from|source)\b", re.I)
    comparison_re = re.compile(
        r"(?:\bcf\.|\bcompare(?:d|s)?\b|\bcomparison\b|\bsimilar\b)", re.I
    )
    catalog_re = re.compile(
        r"\b(?:CDIAL|DEDR|DAD)\s*(?:#|entry\s*)?\d+[a-z]?\b|"
        r"\b(?:CDIAL|DEDR|DAD)\s*#?\s*\d+[a-z]?\b",
        re.I,
    )
    generic_compared = {
        "dialectal", "dictionaryentry", "dr", "frompd", "hindi", "id", "loan",
        "mar", "munda", "possibly", "probably", "same", "source", "uncertain",
    }
    result = []
    for lexeme_id, group in sorted(by_lexeme.items()):
        proxy_rows = [
            row for row in group if row["Method"] in {"source-proxy", "cluster-proxy"}
        ]
        if not proxy_rows:
            continue
        representative = sorted(
            group, key=lambda row: (len(row["Form"]), row["Form_ID"])
        )[0]
        evidence = sorted({
            f"{row['Lexical_Source']}: {row['Original_Etymology']}"
            for row in group if row["Own_Source_Attribution"] and row["Original_Etymology"]
        })
        note_texts = sorted({
            row["Original_Etymology"] for row in group
            if row["Own_Source_Attribution"] and row["Original_Etymology"]
        })
        combined = " || ".join(note_texts)
        compared_forms: list[str] = []
        for note in note_texts:
            for item in source_attribution(note)[2]:
                if fold(item) not in generic_compared and item not in compared_forms:
                    compared_forms.append(item)
        # Some notes use a broad label ("Munda loan") before a semicolon and introduce the
        # actual language forms only after "cf.".  Quoted material after that marker is still a
        # recoverable comparison even when the deliberately narrow source-label parser does not
        # treat the individual language name as a donor label.
        has_quoted_comparison = bool(
            comparison_re.search(combined) and re.search(r"['\"“”]", combined)
        )
        catalog_refs = list(dict.fromkeys(match.group(0) for match in catalog_re.finditer(combined)))
        if catalog_refs:
            evidence_quality = "catalog-indexed"
        elif compared_forms or has_quoted_comparison:
            evidence_quality = "explicit-comparanda"
        elif note_texts:
            evidence_quality = "donor-label-only"
        else:
            evidence_quality = "propagated-only"
        uncertainty = "questioned" if uncertainty_re.search(combined) else "unqualified"
        if direction_re.search(combined):
            directionality = "donor/source asserted"
        elif comparison_re.search(combined):
            directionality = "comparison only"
        else:
            directionality = "attribution without stated direction"
        comparison_scores = [
            (
                form_similarity(form_variants(representative["Form"]), form_variants(item)),
                item,
            )
            for item in compared_forms if form_variants(item)
        ]
        best_similarity, best_compared_form = max(comparison_scores, default=(0.0, ""))
        if not best_compared_form:
            comparison_shape = "unscored"
        elif best_similarity == 1.0:
            comparison_shape = "exact"
        elif best_similarity >= 0.80:
            comparison_shape = "near"
        elif best_similarity >= 0.60:
            comparison_shape = "moderate"
        else:
            comparison_shape = "distant"
        stratum = "+".join(ordered_strata({
            part for row in group for part in row["Stratum"].split("+")
            if part not in {"Nihali residue", "Other", ""}
        })) or ("Other" if any(row["Stratum"] == "Other" for row in group) else "")
        weak = evidence_quality in {"donor-label-only", "propagated-only"} or uncertainty == "questioned"
        historically_diagnostic = any(part in {"Munda", "Dravidian"} for part in stratum.split("+"))
        if lexeme_id in core_ids and weak:
            priority = "critical"
        elif historically_diagnostic and weak:
            priority = "high"
        elif weak:
            priority = "medium"
        else:
            priority = "low"
        review = reviews.get(lexeme_id)
        if not review and uncertainty == "questioned" and lexeme_id in questioned_decisions:
            assessment = questioned_decisions[lexeme_id]
            if assessment == "corroborated-contact":
                confidence = "high"
                rationale = (
                    "Manual review finds a close form-and-meaning match in the printed comparison. "
                    "The contact attribution is corroborated, while historical direction remains "
                    "provisional."
                )
            elif assessment == "route-ambiguous-contact":
                confidence = "medium"
                rationale = (
                    "Manual review supports contact, but the overlapping Korku and Indo-Aryan "
                    "comparanda do not uniquely identify the immediate route or ultimate source."
                )
            elif assessment == "plausible-contact":
                confidence = "medium"
                rationale = (
                    "Manual review finds compatible form and sense, but the comparison requires "
                    "phonological or semantic adjustment and remains provisional."
                )
            elif assessment == "weak-comparison":
                confidence = "low"
                rationale = (
                    "Manual review finds the form or semantic gap too large for positive evidence; "
                    "the printed proposal is retained only as a weak lead."
                )
            else:
                confidence = "low"
                rationale = (
                    "The source supplies only a hedged family label and no recoverable comparandum; "
                    "manual review cannot resolve the donor."
                )
            review = {
                "Assessment": assessment,
                "Preferred_Immediate_Donor": "",
                "Preferred_Ultimate_Source": "",
                "Confidence": confidence,
                "Rationale": rationale,
            }
        direct_note_records = sum(bool(row["Own_Source_Attribution"]) for row in group)
        result.append({
            "Lexeme_ID": lexeme_id,
            "Representative_Form": representative["Form"],
            "Representative_Gloss": representative["Gloss"],
            "Record_Count": str(len(group)),
            "Proxy_Record_Count": str(len(proxy_rows)),
            "Direct_Note_Record_Count": str(direct_note_records),
            "Propagated_Record_Count": str(sum(
                row["Method"] == "cluster-proxy" for row in group
            )),
            "Source_Count": str(len({row["Lexical_Source"] for row in group})),
            "Lexical_Sources": "; ".join(sorted({row["Lexical_Source"] for row in group})),
            "Stratum": stratum,
            "Evidence_Quality": evidence_quality,
            "Uncertainty": uncertainty,
            "Directionality": directionality,
            "Compared_Forms": "; ".join(compared_forms),
            "Catalog_References": "; ".join(catalog_refs),
            "Best_Compared_Form": best_compared_form,
            "Best_Form_Similarity": f"{best_similarity:.3f}" if best_compared_form else "",
            "Comparison_Shape": comparison_shape,
            "Core_Vocabulary": "yes" if lexeme_id in core_ids else "no",
            "Review_Priority": priority,
            "Review_Assessment": review["Assessment"] if review else "",
            "Preferred_Immediate_Donor": review["Preferred_Immediate_Donor"] if review else "",
            "Preferred_Ultimate_Source": review["Preferred_Ultimate_Source"] if review else "",
            "Review_Confidence": review["Confidence"] if review else "",
            "Review_Rationale": review["Rationale"] if review else "",
            "Source_Evidence": " || ".join(evidence),
        })
    critical_ids = {row["Lexeme_ID"] for row in result if row["Review_Priority"] == "critical"}
    core_review_ids = {row["Lexeme_ID"] for row in core_review_rows}
    if critical_ids != core_review_ids:
        raise RuntimeError(
            "core source-proxy review coverage mismatch; "
            f"missing={sorted(critical_ids - core_review_ids)}, "
            f"extra={sorted(core_review_ids - critical_ids)}"
        )
    high_ids = {row["Lexeme_ID"] for row in result if row["Review_Priority"] == "high"}
    diagnostic_review_ids = {row["Lexeme_ID"] for row in diagnostic_review_rows}
    if high_ids != diagnostic_review_ids:
        raise RuntimeError(
            "diagnostic source-proxy review coverage mismatch; "
            f"missing={sorted(high_ids - diagnostic_review_ids)}, "
            f"extra={sorted(diagnostic_review_ids - high_ids)}"
        )
    remaining_questioned_ids = {
        row["Lexeme_ID"] for row in result
        if row["Uncertainty"] == "questioned" and row["Lexeme_ID"] not in preexisting_review_ids
    }
    if remaining_questioned_ids != set(questioned_decisions):
        raise RuntimeError(
            "questioned source-proxy review coverage mismatch; "
            f"missing={sorted(remaining_questioned_ids - set(questioned_decisions))}, "
            f"extra={sorted(set(questioned_decisions) - remaining_questioned_ids)}"
        )
    return result


def build_korku_route_audit(
    source_proxy_quality_audit: list[dict[str, str]],
    languages: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Test printed Korku comparanda against observed Korku forms in Jambu.

    The printed Korku layer is mainly represented by source proxies.  This audit asks the narrower
    and historically useful question whether the printed comparandum can be recovered in the
    independently ingested Korku lexicon.  It then follows an accepted Korku edge upstream when
    one exists.  Failure to find or etymologise a Korku form measures database coverage, not the
    absence or Korku origin of the item.
    """
    all_forms = read_dicts(FORMS)
    form_by_id = {row["ID"]: row for row in all_forms}
    rank1 = {
        row["Child_ID"]: row for row in read_dicts(EDGES)
        if row["Rank"] == "1" and row["Kind"] in {"reflex", "borrowed", "variant"}
    }
    korku_forms = [
        row for row in all_forms
        if row["Language_ID"] == "ko" and REFERENCE not in row.get("Source", "")
        and not row["ID"].startswith("nihprov-")
    ]
    indexed_korku = [
        (row, form_variants(row["Form"] or row["Original"])) for row in korku_forms
    ]
    result = []
    for row in source_proxy_quality_audit:
        if "Korku" not in row["Stratum"].split("+"):
            continue
        query_forms = []
        for compared_form in row["Compared_Forms"].split("; "):
            if compared_form:
                query_forms.extend(form_variants(compared_form))
        best = None
        if query_forms:
            for candidate, candidate_forms in indexed_korku:
                form_score = form_similarity(query_forms, candidate_forms)
                if form_score < 0.35:
                    continue
                gloss_score = gloss_similarity(
                    row["Representative_Gloss"], candidate["Gloss"]
                )
                candidate_key = (
                    0.70 * form_score + 0.30 * gloss_score,
                    form_score, gloss_score, int(candidate["ID"] in rank1), candidate["ID"],
                )
                if best is None or candidate_key > best[0]:
                    best = (candidate_key, candidate)
        if best:
            (_, form_score, gloss_score, _, _), matched = best
        else:
            form_score = gloss_score = 0.0
            matched = {}

        if not query_forms:
            assessment = "no-recoverable-comparandum"
        elif form_score >= 0.80 and gloss_score >= 0.50:
            assessment = "strong-route-match"
        elif form_score >= 0.60 and gloss_score >= 0.40:
            assessment = "possible-route-match"
        elif form_score >= 0.80:
            assessment = "form-only-match"
        else:
            assessment = "weak-or-unmatched"

        matched_id = matched.get("ID", "")
        terminal_id = effective_parent(matched_id, rank1) if matched_id else ""
        terminal = form_by_id.get(terminal_id, {}) if terminal_id != matched_id else {}
        ultimate_family = (
            language_family(terminal.get("Language_ID", ""), languages) if terminal else ""
        )
        if assessment == "strong-route-match":
            basis = (
                "The printed Korku comparandum is independently recoverable with close form and "
                "compatible meaning, supporting Korku as a contact route."
            )
        elif assessment == "possible-route-match":
            basis = (
                "An independently ingested Korku form has compatible form and meaning, but the "
                "match remains approximate."
            )
        elif assessment == "form-only-match":
            basis = (
                "A close Korku string was found without adequate semantic support; homophony or "
                "gloss incompleteness must be resolved manually."
            )
        elif assessment == "no-recoverable-comparandum":
            basis = "The lexical source supplies no machine-recoverable Korku comparison form."
        else:
            basis = (
                "No independently ingested Korku form jointly clears the conservative form and "
                "meaning thresholds."
            )
        result.append({
            "Lexeme_ID": row["Lexeme_ID"],
            "Representative_Form": row["Representative_Form"],
            "Representative_Gloss": row["Representative_Gloss"],
            "Stratum": row["Stratum"],
            "Core_Vocabulary": row["Core_Vocabulary"],
            "Compared_Forms": row["Compared_Forms"],
            "Evidence_Quality": row["Evidence_Quality"],
            "Uncertainty": row["Uncertainty"],
            "Matched_Korku_ID": matched_id,
            "Matched_Korku_Form": matched.get("Form", ""),
            "Matched_Korku_Gloss": matched.get("Gloss", ""),
            "Matched_Korku_Source": matched.get("Source", ""),
            "Form_Similarity": f"{form_score:.3f}" if best else "",
            "Gloss_Similarity": f"{gloss_score:.3f}" if best else "",
            "Route_Assessment": assessment,
            "Korku_Edge_Kind": rank1.get(matched_id, {}).get("Kind", ""),
            "Ultimate_Parent_ID": terminal_id if terminal else "",
            "Ultimate_Parent_Form": terminal.get("Form", ""),
            "Ultimate_Parent_Gloss": terminal.get("Gloss", ""),
            "Ultimate_Parent_Language": languages.get(
                terminal.get("Language_ID", ""), {}
            ).get("Name", terminal.get("Language_ID", "")),
            "Ultimate_Parent_Family": ultimate_family,
            "Interpretation": (
                f"{basis} An empty upstream parent means the matched Korku form is not yet "
                "etymologised in Jambu; it is not evidence that the form originated in Korku."
            ),
        })
    return result


def build_indo_aryan_route_audit(
    audit: list[dict[str, str]], languages: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Separate explicit modern IA comparisons, Sanskrit comparisons, and Korku routes."""
    by_lexeme: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        by_lexeme[row["Lexeme_ID"]].append(row)
    modern_ids = {"H", "M", "B", "Ko"}
    ia_ids = modern_ids | {"Sk", "Indo-Aryan"}
    result = []
    for lexeme_id, rows in sorted(by_lexeme.items()):
        if not any("Indo-Aryan" in row["Stratum"].split("+") for row in rows):
            continue
        mentioned_ids = set()
        for row in rows:
            _strata, language_ids, _forms = source_attribution(row["Original_Etymology"])
            mentioned_ids.update(language_ids)
        specific_ia = mentioned_ids & ia_ids
        modern = bool(specific_ia & modern_ids)
        sanskrit = "Sk" in specific_ia
        generic = "Indo-Aryan" in specific_ia
        resolved_rows = [
            row for row in rows
            if not row["Parent_ID"].startswith("nihprov-")
            and language_family(row["Parent_Language_ID"], languages) == "Indo-Aryan"
        ]
        if sanskrit and modern:
            period_class = "historical-and-modern-explicit"
        elif sanskrit:
            period_class = "sanskrit-explicit"
        elif modern:
            period_class = "modern-ia-explicit"
        elif generic:
            period_class = "generic-ia-explicit"
        elif resolved_rows:
            period_class = "resolved-no-specific-source-language"
        else:
            period_class = "ia-label-no-specific-language"
        representative = min(rows, key=lambda row: (len(row["Form"]), row["Form_ID"]))
        result.append({
            "Lexeme_ID": lexeme_id,
            "Representative_Form": representative["Form"],
            "Representative_Gloss": representative["Gloss"],
            "Stratum": "+".join(ordered_strata({
                part for row in rows for part in row["Stratum"].split("+")
                if part not in {"Nihali residue", "Other", ""}
            })),
            "IA_Source_Language_IDs": "; ".join(sorted(specific_ia)),
            "IA_Source_Languages": "; ".join(sorted({
                languages.get(language_id, {}).get("Name", language_id)
                for language_id in specific_ia
            })),
            "Korku_Route_Mentioned": (
                "yes" if "ko" in mentioned_ids or any(
                    "Korku" in row["Stratum"].split("+") for row in rows
                ) else "no"
            ),
            "English_Mentioned": "yes" if "Eng" in mentioned_ids else "no",
            "Period_Evidence_Class": period_class,
            "Resolved_Parent_IDs": "; ".join(sorted({
                row["Parent_ID"] for row in resolved_rows
            })),
            "Resolved_Parent_Languages": "; ".join(sorted({
                row["Parent_Language"] for row in resolved_rows
            })),
            "Direct_Source_Note": (
                "yes" if any(row["Own_Source_Attribution"] for row in rows) else "no"
            ),
            "Interpretation": (
                "The period class describes languages explicitly named in source comparisons or "
                "the lack of one; it does not date borrowing. Korku route and Indo-Aryan origin "
                "are non-exclusive, and comparison wording does not automatically establish "
                "direction."
            ),
        })
    return result


def build_contact_evidence_tier_audit(
    audit: list[dict[str, str]], source_proxy_quality_audit: list[dict[str, str]],
    resolved_contact_shape_audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Assign each externally labelled lexeme cluster one transparent evidence tier."""
    quality_by_id = {row["Lexeme_ID"]: row for row in source_proxy_quality_audit}
    resolved_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in resolved_contact_shape_audit:
        resolved_by_id[row["Lexeme_ID"]].append(row)
    by_lexeme: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        by_lexeme[row["Lexeme_ID"]].append(row)
    result = []
    contact_families = {"Korku", "Munda", "Indo-Aryan", "Dravidian", "English"}
    for lexeme_id, group in sorted(by_lexeme.items()):
        families = ordered_strata({
            part for row in group for part in row["Stratum"].split("+")
            if part in contact_families
        })
        if not families:
            continue
        representative = sorted(
            group, key=lambda row: (len(row["Form"]), row["Form_ID"])
        )[0]
        quality = quality_by_id.get(lexeme_id)
        resolved = resolved_by_id.get(lexeme_id, [])
        assessment = quality["Review_Assessment"] if quality else ""
        if resolved:
            tier = "resolved-parent"
            basis = "At least one cluster link resolves to an external Jambu parent node."
        elif assessment == "corroborated-contact":
            tier = "manually-corroborated-proxy"
            basis = "An unresolved proxy was independently corroborated in manual review."
        elif assessment in {"route-ambiguous-contact", "plausible-contact"}:
            tier = "manually-qualified-proxy"
            basis = "Manual review retains contact but qualifies the donor route or comparison."
        elif assessment in {"weak-comparison", "unresolved"}:
            tier = "manually-weak-or-unresolved"
            basis = "Manual review finds the comparison weak or leaves the donor unresolved."
        elif quality and quality["Uncertainty"] == "questioned":
            tier = "questioned-unreviewed-proxy"
            basis = "The source comparison is explicitly hedged and is outside the diagnostic review sets."
        elif quality and quality["Evidence_Quality"] in {"catalog-indexed", "explicit-comparanda"}:
            tier = "explicit-unqualified-proxy"
            basis = "The source gives recoverable comparanda without an explicit hedge."
        elif quality and quality["Evidence_Quality"] == "donor-label-only":
            tier = "label-only-unqualified-proxy"
            basis = "The source gives an unhedged donor label but no recoverable compared form."
        else:
            own_source_rows = [row for row in group if row["Own_Source_Attribution"]]
            combined = " ".join(row["Original_Etymology"] for row in own_source_rows)
            uncertainty = bool(re.search(
                r"\?|\b(?:possibly|perhaps|probably|probable|uncertain|maybe|may be|tentative)\b",
                combined, re.I,
            ))
            compared = any(source_attribution(row["Original_Etymology"])[2] for row in own_source_rows)
            if own_source_rows and uncertainty:
                tier = "internal-variant-questioned-source"
                basis = (
                    "A curated internal Nihali variant edge coexists with an explicitly hedged "
                    "external comparison; the internal edge is not external evidence."
                )
            elif own_source_rows and compared:
                tier = "internal-variant-explicit-source"
                basis = (
                    "A curated internal Nihali variant edge coexists with a recoverable unhedged "
                    "external comparison; the source comparison, not the internal edge, carries "
                    "the contact evidence."
                )
            elif own_source_rows:
                tier = "internal-variant-label-only-source"
                basis = (
                    "A curated internal Nihali variant edge coexists with an external donor label "
                    "but no recoverable comparandum."
                )
            else:
                tier = "other-provisional"
                basis = "External label retained provisionally; evidence does not fit a stronger tier."
        result.append({
            "Lexeme_ID": lexeme_id,
            "Representative_Form": representative["Form"],
            "Representative_Gloss": representative["Gloss"],
            "Record_Count": str(len(group)),
            "Source_Count": str(len({row["Lexical_Source"] for row in group})),
            "Lexical_Sources": "; ".join(sorted({row["Lexical_Source"] for row in group})),
            "Stratum": "+".join(families),
            "Families": "; ".join(families),
            "Evidence_Tier": tier,
            "Tier_Basis": basis,
            "Resolved_Parent_Families": "; ".join(sorted({
                row["Parent_Family"] for row in resolved
            })),
            "Proxy_Evidence_Quality": quality["Evidence_Quality"] if quality else "",
            "Proxy_Uncertainty": quality["Uncertainty"] if quality else "",
            "Manual_Assessment": assessment,
            "Manual_Confidence": quality["Review_Confidence"] if quality else "",
        })
    return result


def build_family_contact_evidence_audit(
    audit: list[dict[str, str]], contact_tiers: list[dict[str, str]],
    resolved_links: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Grade each lexeme-family claim independently within mixed contact strata."""
    contact_by_id = {row["Lexeme_ID"]: row for row in contact_tiers}
    audit_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    resolved_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        audit_by_id[row["Lexeme_ID"]].append(row)
    for row in resolved_links:
        resolved_by_id[row["Lexeme_ID"]].append(row)
    uncertainty_re = re.compile(
        r"\?|\b(?:possibly|perhaps|probably|probable|uncertain|maybe|may be|tentative)\b",
        re.I,
    )
    result = []
    for lexeme_id, contact in sorted(contact_by_id.items()):
        group = audit_by_id[lexeme_id]
        links = resolved_by_id.get(lexeme_id, [])
        for family in contact["Families"].split("; "):
            parent_links = [row for row in links if row["Parent_Family"] == family]
            surface_links = [row for row in links if row["Matched_Surface_Family"] == family]
            family_links = list({
                (row["Lexeme_ID"], row["Parent_ID"]): row
                for row in [*parent_links, *surface_links]
            }.values())
            assessments = sorted({
                row["Review_Assessment"] for row in parent_links
                if row["Review_Assessment"]
            })
            if family == "Munda" and parent_links:
                if "near-contact-compatible" in assessments:
                    tier = "resolved-family-corroborated"
                    basis = "A manually reviewed Munda parent link is near contact-compatible."
                elif "possible-correspondence" in assessments:
                    tier = "resolved-family-qualified"
                    basis = (
                        "A Munda parent node is resolved, but manual review retains it only as a "
                        "possible correspondence rather than a demonstrated inherited cognate."
                    )
                elif assessments and set(assessments) == {"weak-comparison"}:
                    tier = "resolved-family-manually-weak"
                    basis = "All resolved Munda parent links for this claim were manually judged weak."
                else:
                    tier = "resolved-family-or-route"
                    basis = "The family is represented by a resolved parent or observed route surface."
            elif parent_links:
                tier = "resolved-parent-family"
                basis = "The family is the family of an identified external database parent."
            elif surface_links:
                tier = "resolved-observed-route"
                basis = (
                    "The family is represented by the best automatically selected observed "
                    "surface under a resolved parent of another or broader family."
                )
            elif contact["Manual_Assessment"] == "corroborated-contact":
                tier = "manually-corroborated-proxy"
                basis = "The unresolved family-labelled comparison was manually corroborated."
            elif contact["Manual_Assessment"] in {
                "route-ambiguous-contact", "plausible-contact",
            }:
                tier = "manually-qualified-proxy"
                basis = "Manual review retains contact but qualifies the route or comparison."
            elif contact["Manual_Assessment"] in {"weak-comparison", "unresolved"}:
                tier = "manually-weak-or-unresolved"
                basis = "Manual review finds the family claim weak or unresolved."
            else:
                own_rows = [
                    row for row in group
                    if family in row["Own_Source_Attribution"].split("+")
                ]
                accepted_manual_component = any(
                    row["Manual_Decision"] == "accept"
                    and family in row["Stratum"].split("+")
                    for row in group
                )
                combined = " ".join(row["Original_Etymology"] for row in own_rows)
                compared = any(
                    source_attribution(row["Original_Etymology"])[2] for row in own_rows
                )
                if own_rows and uncertainty_re.search(combined):
                    tier = "questioned-source"
                    basis = "The family-specific source comparison is explicitly hedged."
                elif own_rows and compared:
                    tier = "explicit-unqualified-source"
                    basis = "The source gives a recoverable family-specific comparison without a hedge."
                elif own_rows:
                    tier = "label-only-source"
                    basis = "The source names the family but gives no recoverable comparandum."
                elif accepted_manual_component:
                    tier = "manually-established-route-or-component"
                    basis = (
                        "An accepted manual review establishes this family as an immediate route "
                        "or a separately analyzed component, although the graph parent represents "
                        "another component or the ultimate source."
                    )
                else:
                    tier = "propagated-cluster-label"
                    basis = (
                        "The family label propagates from a conservatively matched dictionary "
                        "attestation in the same Nihali lexeme cluster."
                    )
            result.append({
                "Lexeme_ID": lexeme_id,
                "Family": family,
                "Representative_Form": contact["Representative_Form"],
                "Representative_Gloss": contact["Representative_Gloss"],
                "Stratum": contact["Stratum"],
                "Evidence_Tier": tier,
                "Tier_Basis": basis,
                "Resolved_Parent_IDs": "; ".join(sorted({
                    row["Parent_ID"] for row in family_links
                })),
                "Resolved_Parent_Families": "; ".join(sorted({
                    row["Parent_Family"] for row in family_links
                })),
                "Matched_Surface_Languages": "; ".join(sorted({
                    row["Matched_Surface_Language"] for row in family_links
                    if row["Matched_Surface_Language"]
                })),
                "Manual_Assessments": "; ".join(assessments),
                "Interpretation": (
                    "Family-specific evidence tier within a potentially mixed contact stratum; "
                    "a resolved route is evidence of contact, not genetic inheritance."
                ),
            })
    return result


def build_family_evidence_bracket_audit(
    family_tiers: list[dict[str, str]], total_lexemes: int,
) -> list[dict[str, str]]:
    """Create transparent evidence brackets; these are not statistical confidence intervals."""
    result = []
    for family in ("Indo-Aryan", "Korku", "Munda", "Dravidian", "English"):
        rows = [row for row in family_tiers if row["Family"] == family]
        tiers = Counter(row["Evidence_Tier"] for row in rows)
        floor = sum(tiers[tier] for tier in (
            "resolved-parent-family", "resolved-family-corroborated",
            "manually-corroborated-proxy",
        ))
        excluded = sum(tiers[tier] for tier in (
            "label-only-source", "manually-weak-or-unresolved", "questioned-source",
            "resolved-family-manually-weak",
        ))
        envelope = len(rows) - excluded
        result.append({
            "Family": family,
            "All_Labelled_Clusters": str(len(rows)),
            "High_Specificity_Floor": str(floor),
            "Supported_Envelope": str(envelope),
            "Weak_Or_Unresolved_Excluded": str(excluded),
            "Floor_Share_Of_All_Lexemes": f"{floor / total_lexemes:.3f}",
            "Envelope_Share_Of_All_Lexemes": f"{envelope / total_lexemes:.3f}",
            "Definition": (
                "Floor = family-specific resolved parent or manually corroborated "
                "proxy. Envelope additionally includes qualified resolved links, explicit "
                "unqualified, propagated, manually established route/component, and manually "
                "plausible/route-ambiguous evidence, while "
                "excluding label-only, questioned, and manually weak/unresolved cases. This is a "
                "sensitivity bracket, not a confidence interval."
            ),
        })
    return result


def build_origin_evidence_matrix(
    cluster_audit: list[dict[str, str]],
    core_concept_profile_audit: list[dict[str, str]],
    core_residue_root_inventory: list[dict[str, str]],
    family_evidence_bracket_audit: list[dict[str, str]],
    family_attribution_replication_audit: list[dict[str, str]],
    korku_route_audit: list[dict[str, str]],
    resolved_contact_shape_audit: list[dict[str, str]],
    dravidian_correspondence_audit: list[dict[str, str]],
    closed_class_audit: list[dict[str, str]],
    cross_source_agreement_audit: list[dict[str, str]],
    residue_threshold_sensitivity: list[dict[str, str]],
    global_variant_sensitivity_audit: list[dict[str, str]],
    residue_contact_component_review: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Make the origin inference auditable without turning qualitative evidence into a score."""
    family_brackets = {row["Family"]: row for row in family_evidence_bracket_audit}
    family_replication = {
        row["Family"]: row for row in family_attribution_replication_audit
    }
    core_profiles = Counter(row["Profile_Class"] for row in core_concept_profile_audit)
    root_replication = Counter(
        row["Replication_Grade"] for row in core_residue_root_inventory
    )
    korku_routes = Counter(row["Route_Assessment"] for row in korku_route_audit)
    korku_upstream = Counter(
        row["Ultimate_Parent_Family"] or "unresolved-korku"
        for row in korku_route_audit if row["Route_Assessment"] == "strong-route-match"
    )
    munda_links = [
        row for row in resolved_contact_shape_audit if row["Parent_Family"] == "Munda"
    ]
    munda_review = Counter(row["Review_Assessment"] for row in munda_links)
    dravidian_review = Counter(row["Assessment"] for row in dravidian_correspondence_audit)
    closed_pronouns = [row for row in closed_class_audit if row["Domain"] == "pronoun"]
    closed_numerals = [row for row in closed_class_audit if row["Domain"] == "low numeral"]
    source_agreement = Counter(row["Agreement_Class"] for row in cross_source_agreement_audit)
    sensitivity_by_label = {
        row["Threshold_Label"]: row for row in residue_threshold_sensitivity
    }
    global_variant_assessments = Counter(
        row["Assessment"] for row in global_variant_sensitivity_audit
    )
    residue_component_assessments = Counter(
        row["Assessment"] for row in residue_contact_component_review
    )
    residue_clusters = sum(row["Stratum"] == "Nihali residue" for row in cluster_audit)
    ia_bracket = family_brackets["Indo-Aryan"]
    munda_bracket = family_brackets["Munda"]

    rows = [
        {
            "Evidence_ID": "E01", "Domain": "stable lexical residue",
            "Finding": (
                f"{residue_clusters} lexeme clusters remain residual; the reviewed core reduces "
                f"to {len(core_residue_root_inventory)} root hypotheses, including "
                f"{root_replication['moderate'] + root_replication['strong'] + root_replication['very strong']} "
                "attested in at least two sources."
            ),
            "Supports_Hypothesis": "independent lineage with layered relexification",
            "Challenges_Hypothesis": "argot-only origin; complete descent from a known donor",
            "Evidential_Weight": "moderate",
            "Limitation": "Unmatched does not mean inherited; old loans can also be stable.",
            "Audit_Or_Source": CORE_RESIDUE_ROOT_INVENTORY_NAME,
        },
        {
            "Evidence_ID": "E02", "Domain": "concept-level basic vocabulary",
            "Finding": (
                f"Of 91 covered concepts, {core_profiles['residue-only']} are residue-only, "
                f"{core_profiles['residue-plus-contact']} mix residue and contact forms, and "
                f"{core_profiles['contact-only-single-family'] + core_profiles['contact-only-mixed-family']} "
                "are contact-only."
            ),
            "Supports_Hypothesis": "layered relexification rather than one-source replacement",
            "Challenges_Hypothesis": "simple affiliation with Indo-Aryan, Dravidian, or Munda",
            "Evidential_Weight": "moderate",
            "Limitation": "Concept links and synonym clustering remain imperfect.",
            "Audit_Or_Source": CORE_CONCEPT_PROFILE_AUDIT_NAME,
        },
        {
            "Evidence_ID": "E03", "Domain": "Korku transmission route",
            "Finding": (
                f"Among {len(korku_route_audit)} Korku-labelled proxies, "
                f"{korku_routes['strong-route-match']} independently recover close Korku forms; "
                f"only {korku_upstream['Munda']} currently trace to an upstream Munda parent."
            ),
            "Supports_Hypothesis": "profound Korku-mediated contact",
            "Challenges_Hypothesis": "counting every Korku-labelled item as ultimately Munda",
            "Evidential_Weight": "moderate",
            "Limitation": "Korku etymological coverage is sparse; route recovery is automated.",
            "Audit_Or_Source": KORKU_ROUTE_AUDIT_NAME,
        },
        {
            "Evidence_ID": "E04", "Domain": "non-Korku Munda specificity",
            "Finding": (
                f"Munda has a high-specificity floor of {munda_bracket['High_Specificity_Floor']} "
                f"among {munda_bracket['All_Labelled_Clusters']} labelled clusters; only "
                f"{family_replication['Munda']['Two_Plus_Direct_Labelled_Sources']} of "
                f"{family_replication['Munda']['Multi_Source_Clusters']} replicated lexemes receive "
                "direct Munda labels in two or more sources."
            ),
            "Supports_Hypothesis": "Munda contact, possibly including an older minor layer",
            "Challenges_Hypothesis": "demonstrated direct Munda descent",
            "Evidential_Weight": "moderate negative",
            "Limitation": "Low specificity can reflect missing comparative resources, not falsehood.",
            "Audit_Or_Source": (
                f"{FAMILY_EVIDENCE_BRACKET_AUDIT_NAME}; "
                f"{FAMILY_ATTRIBUTION_REPLICATION_AUDIT_NAME}"
            ),
        },
        {
            "Evidence_ID": "E05", "Domain": "Nihali–Munda correspondences",
            "Finding": (
                f"{len(munda_links)} resolved links represent "
                f"{len({row['Parent_ID'] for row in munda_links})} parent roots: "
                f"{munda_review['near-contact-compatible']} near contact-compatible, "
                f"{munda_review['possible-correspondence']} possible, and "
                f"{munda_review['weak-comparison']} weak; no recurring sound law is demonstrated."
            ),
            "Supports_Hypothesis": "contact",
            "Challenges_Hypothesis": "a presently demonstrated Munda branch",
            "Evidential_Weight": "moderate negative",
            "Limitation": "The reviewed set is small and cannot exclude a very deep relationship.",
            "Audit_Or_Source": MUNDA_CORRESPONDENCE_REVIEW.name,
        },
        {
            "Evidence_ID": "E06", "Domain": "Indo-Aryan layer",
            "Finding": (
                f"The evidence bracket runs from {ia_bracket['High_Specificity_Floor']} "
                f"high-specificity to {ia_bracket['Supported_Envelope']} supported clusters; the "
                "layer is noun-heavy and includes Korku-mediated routes."
            ),
            "Supports_Hypothesis": "large, chronologically mixed donor layer",
            "Challenges_Hypothesis": "Indo-Aryan genetic affiliation",
            "Evidential_Weight": "strong for contact; weak for origin",
            "Limitation": "Immediate donors, ultimate etyma, and borrowing dates are not identical.",
            "Audit_Or_Source": (
                f"{FAMILY_EVIDENCE_BRACKET_AUDIT_NAME}; {INDO_ARYAN_ROUTE_AUDIT_NAME}; "
                f"{LAYER_CATEGORY_AUDIT_NAME}"
            ),
        },
        {
            "Evidence_ID": "E07", "Domain": "Dravidian layer",
            "Finding": (
                f"{sum(int(row['Link_Count']) for row in dravidian_correspondence_audit)} "
                f"resolved links collapse to {len(dravidian_correspondence_audit)} roots; "
                f"{dravidian_review['near-contact-compatible']} are near contact-compatible and "
                f"{dravidian_review['weak-form-link'] + dravidian_review['weak-semantic-link']} "
                "are weak on form or meaning, without a sound-law system."
            ),
            "Supports_Hypothesis": "smaller heterogeneous contact layer",
            "Challenges_Hypothesis": "Dravidian genetic affiliation",
            "Evidential_Weight": "moderate for contact; negative for origin",
            "Limitation": "The mechanical root screen is not a substitute for expert Dravidian review.",
            "Audit_Or_Source": DRAVIDIAN_CORRESPONDENCE_AUDIT_NAME,
        },
        {
            "Evidence_ID": "E08", "Domain": "closed-class asymmetry",
            "Finding": (
                f"Pronouns have {sum(row['Effective_Stratum'] == 'Nihali residue' for row in closed_pronouns)} "
                f"residual links out of {len(closed_pronouns)}, while low numerals have only "
                f"{sum(row['Effective_Stratum'] == 'Nihali residue' for row in closed_numerals)} "
                f"out of {len(closed_numerals)}."
            ),
            "Supports_Hypothesis": "selective relexification of an older grammatical lexicon",
            "Challenges_Hypothesis": "uniform recent borrowing or wholesale argot replacement",
            "Evidential_Weight": "moderate",
            "Limitation": "Closed classes can be borrowed and the linked inventory contains synonyms.",
            "Audit_Or_Source": CLOSED_CLASS_AUDIT_NAME,
        },
        {
            "Evidence_ID": "E09", "Domain": "source dependence",
            "Finding": (
                f"Among {len(cross_source_agreement_audit)} replicated clusters, only "
                f"{source_agreement['multi-source-exact-agreement']} have exact direct family-label "
                f"agreement; {source_agreement['single-labelled-source']} are labelled by one source."
            ),
            "Supports_Hypothesis": "cautious evidence-tier interpretation",
            "Challenges_Hypothesis": "precise family percentages treated as objective counts",
            "Evidential_Weight": "strong methodological caution",
            "Limitation": "The sources are not fully independent and silence is not disagreement.",
            "Audit_Or_Source": CROSS_SOURCE_AGREEMENT_AUDIT_NAME,
        },
        {
            "Evidence_ID": "E10", "Domain": "surface word shape",
            "Finding": "All five layers have median folded form length 5 and overlapping surface diagnostics.",
            "Supports_Hypothesis": "phonological adaptation across contact layers",
            "Challenges_Hypothesis": "classifying the residue by impressionistic word shape",
            "Evidential_Weight": "negative control",
            "Limitation": "No phonological model or significance test was fitted.",
            "Audit_Or_Source": LAYER_FORM_SHAPE_AUDIT_NAME,
        },
        {
            "Evidence_ID": "E11", "Domain": "morphology and pronouns",
            "Finding": (
                "Zide reports that Nihali pronouns do not resemble Munda pronouns and that proposed "
                "case and verbal analyses do not yield a Proto-Munda-like inherited system."
            ),
            "Supports_Hypothesis": "independent lineage with contact-driven restructuring",
            "Challenges_Hypothesis": "direct Munda affiliation",
            "Evidential_Weight": "moderate",
            "Limitation": "The grammar is underdocumented and contact can restructure morphology.",
            "Audit_Or_Source": "Zide 1996: 93–100; Kuiper 1962",
        },
        {
            "Evidence_ID": "E12", "Domain": "broad structural typology",
            "Finding": "Grambank similarity does not place Nihali uniquely with Munda rather than regional controls.",
            "Supports_Hypothesis": "areal convergence as an alternative to genealogy",
            "Challenges_Hypothesis": "typological resemblance as sufficient Munda proof",
            "Evidential_Weight": "negative control",
            "Limitation": "Synchronic features are dependent, missing, and not phylogenetically modelled.",
            "Audit_Or_Source": "nihali-grambank-structural-sensitivity.csv",
        },
        {
            "Evidence_ID": "E13", "Domain": "argot hypothesis",
            "Finding": "Stable basic terms recur across early and late sources without a demonstrated productive disguise rule.",
            "Supports_Hypothesis": "argot-like substitution as at most a secondary process",
            "Challenges_Hypothesis": "argot-only origin",
            "Evidential_Weight": "moderate negative",
            "Limitation": "The audit did not reconstruct possible historical disguise operations.",
            "Audit_Or_Source": REPLICATED_RESIDUE_AUDIT_NAME,
        },
        {
            "Evidence_ID": "E14", "Domain": "residue threshold sensitivity",
            "Finding": (
                "A production-like score/margin-only relaxation flags "
                f"{sensitivity_by_label['production-like score/margin only']['Flagged_Residue_Clusters']} "
                "residual clusters, but "
                f"{sensitivity_by_label['production-like score/margin only']['Unreviewed']} never "
                "pass the required component gates; the very-high setting leaves "
                f"{sensitivity_by_label['very high score']['Flagged_Residue_Clusters']}."
            ),
            "Supports_Hypothesis": "conservative residue as a review category",
            "Challenges_Hypothesis": "assigning a family by lowering one composite cutoff",
            "Evidential_Weight": "strong methodological caution",
            "Limitation": "The sensitivity is score-only and donor databases have unequal coverage.",
            "Audit_Or_Source": RESIDUE_THRESHOLD_SENSITIVITY_NAME,
        },
        {
            "Evidence_ID": "E15", "Domain": "whole-lexicon variant sensitivity",
            "Finding": (
                "A mechanical exact-gloss and >=0.72 whole-form screen, supplemented by "
                "explicitly reviewed derivational and inflectional families, covered "
                f"{len(global_variant_sensitivity_audit)} residual/contact cluster pairs; "
                f"{global_variant_assessments['variant']} are plausible variants and "
                f"{global_variant_assessments['qualified']} remain qualified leads."
            ),
            "Supports_Hypothesis": "a robust residual core after broader variant consolidation",
            "Challenges_Hypothesis": "treating every provisional residual cluster as independent evidence",
            "Evidential_Weight": "strong methodological caution",
            "Limitation": "The mechanical screen misses non-identical glosses and more remote sound correspondences.",
            "Audit_Or_Source": GLOBAL_VARIANT_SENSITIVITY_AUDIT_NAME,
        },
        {
            "Evidence_ID": "E16", "Domain": "residual expression decomposition",
            "Finding": (
                f"A targeted audit of {len(residue_contact_component_review)} residual expressions "
                f"finds {residue_component_assessments['transparent-component']} transparent and "
                f"{residue_component_assessments['qualified-component']} qualified contact "
                "components, plus one quarantined source/gloss conflict."
            ),
            "Supports_Hypothesis": "mixed morphology and incomplete relexification",
            "Challenges_Hypothesis": "treating the entire residue count as inherited root evidence",
            "Evidential_Weight": "strong methodological caution",
            "Limitation": (
                "This was a targeted semantic review of conspicuous expressions, not a complete "
                "morpheme segmentation of every residual item."
            ),
            "Audit_Or_Source": RESIDUE_CONTACT_COMPONENT_REVIEW.name,
        },
        {
            "Evidence_ID": "E17", "Domain": "synthesis",
            "Finding": "No known donor layer explains the whole core, and no alternative family has a regular inherited correspondence system.",
            "Supports_Hypothesis": "independent lineage with layered relexification",
            "Challenges_Hypothesis": "all simpler single-source origin accounts",
            "Evidential_Weight": "moderate overall, diagnosis by exclusion",
            "Limitation": "An unknown deep relationship remains possible until better grammar and comparative data exist.",
            "Audit_Or_Source": REPORT_NAME,
        },
    ]
    return rows


def build_residue_threshold_sensitivity(
    audit: list[dict[str, str]], languages: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Show what a score-only relaxation would flag among clusters still left residual."""
    form_by_id = {row["ID"]: row for row in read_dicts(FORMS)}
    by_lexeme: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        by_lexeme[row["Lexeme_ID"]].append(row)
    candidates = []
    for lexeme_id, rows in by_lexeme.items():
        if not all(row["Stratum"] == "Nihali residue" for row in rows):
            continue
        candidate_rows = [row for row in rows if row["Alternatives"]]
        if not candidate_rows:
            continue
        best = max(
            candidate_rows,
            key=lambda row: (
                float(row["Score"] or 0), float(row["Margin"] or 0), row["Form_ID"],
            ),
        )
        parent_id = best["Alternatives"].split("; ", 1)[0].split(":", 1)[0]
        parent = form_by_id.get(parent_id, {})
        family = language_family(parent.get("Language_ID", ""), languages)
        candidates.append({
            "Lexeme_ID": lexeme_id,
            "Score": float(best["Score"] or 0),
            "Margin": float(best["Margin"] or 0),
            "Family": family,
            "Manual_Decision": best["Manual_Decision"],
        })
    thresholds = (
        ("very loose", 0.65, 0.025),
        ("loose", 0.70, 0.025),
        ("moderate", 0.75, 0.025),
        ("high score, low margin", 0.80, 0.025),
        ("production-like score/margin only", 0.80, 0.045),
        ("very high score", 0.85, 0.045),
    )
    result = []
    for label, minimum_score, minimum_margin in thresholds:
        flagged = [
            row for row in candidates
            if row["Score"] >= minimum_score and row["Margin"] >= minimum_margin
        ]
        families = Counter(row["Family"] for row in flagged)
        decisions = Counter(row["Manual_Decision"] or "unreviewed" for row in flagged)
        result.append({
            "Threshold_Label": label,
            "Minimum_Composite_Score": f"{minimum_score:.3f}",
            "Minimum_Margin": f"{minimum_margin:.3f}",
            "Flagged_Residue_Clusters": str(len(flagged)),
            "Indo_Aryan_Candidates": str(families["Indo-Aryan"]),
            "Dravidian_Candidates": str(families["Dravidian"]),
            "Munda_Candidates": str(families["Munda"]),
            "Other_Candidates": str(families["Other"]),
            "English_Candidates": str(families["English"]),
            "Manual_Rejected": str(decisions["reject"]),
            "Manual_Deferred": str(decisions["defer"]),
            "Unreviewed": str(decisions["unreviewed"]),
            "Interpretation": (
                "Score-only sensitivity omits the production form, gloss, and donor-specific "
                "component gates. Flagged rows remain review leads, not assignments; family counts "
                "also reflect unequal donor-database coverage, especially sparse etymologised "
                "Korku surfaces."
            ),
        })
    return result


def render_report(
    audit: list[dict[str, str]], proxies: dict[str, dict[str, str]],
    core_audit: list[dict[str, str]],
    global_variant_sensitivity_audit: list[dict[str, str]],
    residue_contact_component_review: list[dict[str, str]],
    source_variation_audit: list[dict[str, str]],
    source_proxy_quality_audit: list[dict[str, str]],
    korku_route_audit: list[dict[str, str]],
    closed_class_audit: list[dict[str, str]],
    core_residue_root_audit: list[dict[str, str]],
    core_residue_root_inventory: list[dict[str, str]],
    core_concept_profile_audit: list[dict[str, str]],
    replicated_residue_audit: list[dict[str, str]],
    resolved_contact_shape_audit: list[dict[str, str]],
    dravidian_correspondence_audit: list[dict[str, str]],
    contact_evidence_tier_audit: list[dict[str, str]],
    family_contact_evidence_audit: list[dict[str, str]],
    source_profile_audit: list[dict[str, str]],
    cross_source_agreement_audit: list[dict[str, str]],
    family_attribution_replication_audit: list[dict[str, str]],
    layer_replication_audit: list[dict[str, str]],
    layer_category_audit: list[dict[str, str]],
    layer_form_shape_audit: list[dict[str, str]],
    indo_aryan_route_audit: list[dict[str, str]],
    origin_evidence_matrix: list[dict[str, str]],
    residue_threshold_sensitivity: list[dict[str, str]],
) -> str:
    methods = Counter(row["Method"] for row in audit)
    strata = Counter(row["Stratum"] for row in audit)
    confidence = Counter(row["Confidence"] for row in audit)
    sources = Counter(row["Lexical_Source"] for row in audit)
    core_strata = Counter(row["Stratum"] for row in core_audit)
    core_sensitivity_strata = Counter(
        row["Sensitivity_Stratum"] or row["Stratum"] for row in core_audit
    )
    global_variant_assessments = Counter(
        row["Assessment"] for row in global_variant_sensitivity_audit
    )
    global_supported_variants = [
        row for row in global_variant_sensitivity_audit
        if row["Assessment"] in {"variant", "qualified"}
    ]
    global_variant_family_involvement = Counter(
        family
        for row in global_supported_variants
        for family in row["Reference_Stratum"].split("+")
        if family in {"Korku", "Munda", "Indo-Aryan", "Dravidian", "English"}
    )
    residue_component_assessments = Counter(
        row["Assessment"] for row in residue_contact_component_review
    )
    residue_component_layers = Counter(
        family
        for row in residue_contact_component_review
        for family in row["Layer"].split("+")
    )
    core_munda_rows = [row for row in core_audit if "Munda" in row["Stratum"]]
    core_munda_concepts = {
        concept for row in core_munda_rows for concept in row["Concepts"].split("; ")
    }
    core_munda_ids = {row["Lexeme_ID"] for row in core_munda_rows}
    core_munda_family_tiers = Counter(
        row["Evidence_Tier"] for row in family_contact_evidence_audit
        if row["Family"] == "Munda" and row["Lexeme_ID"] in core_munda_ids
    )
    sensitivity_munda_rows = [
        row for row in core_audit if "Munda" in (row["Sensitivity_Stratum"] or row["Stratum"])
    ]
    sensitivity_munda_concepts = {
        concept for row in sensitivity_munda_rows for concept in row["Concepts"].split("; ")
    }
    core_concepts_covered = {
        concept for row in core_audit for concept in row["Concepts"].split("; ")
    }
    core_concept_profiles = Counter(
        row["Profile_Class"] for row in core_concept_profile_audit
    )
    core_concept_family_involvement = {
        family: sum(
            family in row["Contact_Families"].split("+")
            for row in core_concept_profile_audit
        )
        for family in ("Korku", "Munda", "Indo-Aryan", "Dravidian")
    }
    residue_only_concepts = [
        row for row in core_concept_profile_audit if row["Profile_Class"] == "residue-only"
    ]
    mixed_residue_concepts = [
        row for row in core_concept_profile_audit
        if row["Profile_Class"] == "residue-plus-contact"
    ]
    variation_relationships = Counter(row["Relationship"] for row in source_variation_audit)
    disjoint_assessments = Counter(
        row["Review_Assessment"] for row in source_variation_audit if row["Review_Assessment"]
    )
    proxy_quality = Counter(row["Evidence_Quality"] for row in source_proxy_quality_audit)
    proxy_uncertainty = Counter(row["Uncertainty"] for row in source_proxy_quality_audit)
    proxy_directionality = Counter(row["Directionality"] for row in source_proxy_quality_audit)
    proxy_comparison_shapes = Counter(
        row["Comparison_Shape"] for row in source_proxy_quality_audit
    )
    proxy_shape_by_uncertainty = {
        uncertainty: Counter(
            row["Comparison_Shape"] for row in source_proxy_quality_audit
            if row["Uncertainty"] == uncertainty
        )
        for uncertainty in ("unqualified", "questioned")
    }
    proxy_priority = Counter(row["Review_Priority"] for row in source_proxy_quality_audit)
    cross_source_agreement = Counter(
        row["Agreement_Class"] for row in cross_source_agreement_audit
    )
    indo_aryan_period_classes = Counter(
        row["Period_Evidence_Class"] for row in indo_aryan_route_audit
    )
    indo_aryan_korku_routes = sum(
        row["Korku_Route_Mentioned"] == "yes" for row in indo_aryan_route_audit
    )
    residue_sensitivity_by_label = {
        row["Threshold_Label"]: row for row in residue_threshold_sensitivity
    }
    korku_route_assessments = Counter(
        row["Route_Assessment"] for row in korku_route_audit
    )
    core_korku_routes = [row for row in korku_route_audit if row["Core_Vocabulary"] == "yes"]
    core_korku_route_assessments = Counter(
        row["Route_Assessment"] for row in core_korku_routes
    )
    korku_strong_ultimate = Counter(
        row["Ultimate_Parent_Family"] or "unresolved-korku"
        for row in korku_route_audit
        if row["Route_Assessment"] == "strong-route-match"
    )
    critical_proxy_assessments = Counter(
        row["Review_Assessment"] for row in source_proxy_quality_audit
        if row["Review_Priority"] == "critical"
    )
    diagnostic_proxy_assessments = Counter(
        row["Review_Assessment"] for row in source_proxy_quality_audit
        if row["Review_Priority"] == "high"
    )
    questioned_review_ids = {
        lexeme_id for ids in QUESTIONED_PROXY_REVIEW_IDS.values() for lexeme_id in ids
    }
    questioned_proxy_assessments = Counter(
        row["Review_Assessment"] for row in source_proxy_quality_audit
        if row["Lexeme_ID"] in questioned_review_ids
    )
    replicated_residue_grades = Counter(
        row["Replication_Grade"] for row in replicated_residue_audit
    )
    replicated_core_residue = sum(
        row["Core_Effective_Residue"] == "yes" for row in replicated_residue_audit
    )
    resolved_parent_families = Counter(
        row["Parent_Family"] for row in resolved_contact_shape_audit
    )
    resolved_parent_languages = Counter(
        row["Parent_Language"] for row in resolved_contact_shape_audit
    )
    resolved_match_shapes = {
        family: Counter(
            row["Match_Shape"] for row in resolved_contact_shape_audit
            if row["Parent_Family"] == family
        )
        for family in resolved_parent_families
    }
    resolved_surface_match_shapes = {
        family: Counter(
            row["Surface_Match_Shape"] for row in resolved_contact_shape_audit
            if row["Parent_Family"] == family and row["Surface_Match_Shape"]
        )
        for family in resolved_parent_families
    }
    route_family_involvement = {
        parent_family: Counter({
            route_family: sum(
                row["Parent_Family"] == parent_family
                and route_family in row["Source_Stratum"].split("+")
                for row in resolved_contact_shape_audit
            )
            for route_family in ("Korku", "Munda", "Indo-Aryan", "Dravidian")
        })
        for parent_family in resolved_parent_families
    }
    munda_link_rows = [
        row for row in resolved_contact_shape_audit if row["Parent_Family"] == "Munda"
    ]
    munda_review = Counter(row["Review_Assessment"] for row in munda_link_rows)
    munda_series = Counter(row["Correspondence_Series"] for row in munda_link_rows)
    munda_parent_roots = len({row["Parent_ID"] for row in munda_link_rows})
    dravidian_review = Counter(
        row["Assessment"] for row in dravidian_correspondence_audit
    )
    dravidian_series_types = Counter(
        row["Series_Type"] for row in dravidian_correspondence_audit
    )
    dravidian_repeated_nonidentity = Counter(
        row["Initial_Correspondence"] for row in dravidian_correspondence_audit
        if row["Series_Type"] == "repeated-nonidentity"
    )
    dravidian_repeated_series = ", ".join(
        f"{series} ({count} roots)"
        for series, count in dravidian_repeated_nonidentity.most_common()
    )
    contact_tiers = Counter(row["Evidence_Tier"] for row in contact_evidence_tier_audit)
    contact_family_tiers = {
        family: Counter(
            row["Evidence_Tier"] for row in family_contact_evidence_audit
            if row["Family"] == family
        )
        for family in ("Korku", "Munda", "Indo-Aryan", "Dravidian", "English")
    }
    reproducible_proxy = sum(
        row["Evidence_Quality"] in {"catalog-indexed", "explicit-comparanda"}
        and row["Uncertainty"] == "unqualified"
        for row in source_proxy_quality_audit
    )
    by_lexeme: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit:
        by_lexeme[row["Lexeme_ID"]].append(row)

    lexeme_strata: Counter[str] = Counter()
    lexeme_methods: Counter[str] = Counter()
    manual_reviews: Counter[str] = Counter()
    manual_triggers: Counter[str] = Counter()
    propagated_clusters = 0
    propagated_records = 0
    for group in by_lexeme.values():
        labels = {row["Stratum"] for row in group}
        external_parts: set[str] = set()
        for label in labels:
            if label not in {"Nihali residue", "Other", ""}:
                external_parts.update(label.split("+"))
        if external_parts:
            label = "+".join(ordered_strata(external_parts))
        elif "Other" in labels:
            label = "Other"
        else:
            label = "Nihali residue"
        lexeme_strata[label] += 1
        group_methods = {row["Method"] for row in group}
        manual_decision = next(
            (row["Manual_Decision"] for row in group if row["Manual_Decision"]), ""
        )
        if manual_decision:
            manual_reviews[manual_decision] += 1
            manual_trigger = next(
                (row["Manual_Trigger"] for row in group if row["Manual_Trigger"]), ""
            )
            manual_triggers[manual_trigger] += 1
        cluster_propagated = sum(
            row["Method"] in {"cluster-resolved", "cluster-proxy"} for row in group
        )
        propagated_records += cluster_propagated
        propagated_clusters += bool(cluster_propagated)
        if "existing-curated" in group_methods:
            lexeme_methods["existing-curated"] += 1
        elif group_methods & {"source-resolved", "source-proxy"}:
            lexeme_methods["record-local source/editorial note"] += 1
        elif group_methods & {"cluster-resolved", "cluster-proxy"}:
            lexeme_methods["cross-dictionary propagation"] += 1
        elif "manual-resolved" in group_methods:
            lexeme_methods["manually accepted source-free lead"] += 1
        elif "manual-rejected" in group_methods:
            lexeme_methods["manually rejected computational lead"] += 1
        elif "manual-deferred" in group_methods:
            lexeme_methods["manually deferred computational lead"] += 1
        else:
            lexeme_methods["unresolved residue"] += 1
    source_parent_decisions = Counter(
        next(row["Manual_Decision"] for row in group if row["Manual_Trigger"])
        for group in by_lexeme.values()
        if any(row["Manual_Trigger"] == "source-attributed-parent" for row in group)
    )

    family_involvement: Counter[str] = Counter()
    for label, count in lexeme_strata.items():
        for family in label.split("+"):
            if family in {"Korku", "Munda", "Indo-Aryan", "Dravidian", "English"}:
                family_involvement[family] += count

    external = sum(n for label, n in strata.items() if label not in {"Nihali residue", "Other", ""})
    residue = strata["Nihali residue"]
    external_lexemes = sum(
        count for label, count in lexeme_strata.items()
        if label not in {"Nihali residue", "Other", ""}
    )
    residue_lexemes = lexeme_strata["Nihali residue"]
    multi_source_clusters = {
        lexeme_id: group for lexeme_id, group in by_lexeme.items()
        if len({row["Lexical_Source"] for row in group}) >= 2
    }
    multi_source_residue = sum(
        all(row["Stratum"] == "Nihali residue" for row in group)
        for group in multi_source_clusters.values()
    )
    explicit = sum(
        methods[name] for name in (
            "existing-curated", "source-resolved", "source-proxy",
            "cluster-resolved", "cluster-proxy", "manual-resolved",
        )
    )
    stable_residue = [
        (
            row["Representative_Form"], row["Representative_Gloss"],
            int(row["Record_Count"]), row["Lexical_Sources"].replace("; ", ", "),
        )
        for row in replicated_residue_audit
        if int(row["Source_Count"]) >= 3
        and (not row["Core_Concepts"] or row["Core_Effective_Residue"] == "yes")
    ]
    stable_residue.sort(key=lambda item: (-item[2], item[0]))

    lines = [
        "# Provisional etymology of the Jambu Nihali lexicon",
        "",
        "## Scope, units, and headline",
        "",
        f"This analysis covers **{len(audit):,} attested Nihali database records** and assigns each "
        f"one a provisional rank-1 hypothesis. A conservative form-plus-meaning comparison across "
        f"the five lexical sources collapses them to **{len(by_lexeme):,} normalized lexeme "
        f"clusters**. These clusters, rather than raw dictionary rows or proxy IDs, are the least "
        f"misleading unit for historical proportions. The overlay creates {len(proxies):,} explicit "
        "proxy entries where no existing Jambu etymon can safely carry the claim.",
        "",
        f"At lexeme-cluster level, {external_lexemes:,} "
        f"({external_lexemes/len(by_lexeme):.1%}) have an externally attributed contact source and "
        f"{residue_lexemes:,} ({residue_lexemes/len(by_lexeme):.1%}) remain in the Nihali residue. "
        f"The remaining {lexeme_strata['Other']:,} cluster is retained as Other. "
        f"At record level the corresponding figures are {external:,} ({external/len(audit):.1%}) "
        f"external and {residue:,} ({residue/len(audit):.1%}) residue, with "
        f"{strata['Other']:,} Other. Source-attributed, "
        f"cross-dictionary-propagated, manually adjudicated, or previously curated evidence "
        f"supports {explicit:,} records.",
        "",
        "The resulting historical hypothesis is that Nihali is best treated as the sole documented "
        "survivor of an **independent Central Indian lineage that underwent layered relexification**. "
        "Indo-Aryan is the numerically widest attribution in the expanded database, while "
        "Korku/Munda is the most historically diagnostic contact channel; other Munda comparisons "
        "may include an older layer, and Dravidian is smaller but real. Lexical comparison alone "
        "does not locate the independent lineage's "
        "earlier homeland, date it, or demonstrate a relationship to another isolate.",
        "",
        "## Lexeme-cluster results",
        "",
        "| Stratum | Lexeme clusters | Share |",
        "|---|---:|---:|",
    ]
    for label, count in lexeme_strata.most_common():
        lines.append(f"| {label or 'Unclassified'} | {count:,} | {count/len(by_lexeme):.1%} |")
    lines += [
        "",
        "### Family involvement (non-exclusive)",
        "",
        "Mixed labels count under each named family in this table; rows therefore do not sum to "
        "the lexeme total.",
        "",
        "| Named contact family | Lexeme clusters | Share of all clusters |",
        "|---|---:|---:|",
    ]
    for label, count in family_involvement.most_common():
        lines.append(f"| {label} | {count:,} | {count/len(by_lexeme):.1%} |")
    lines += [
        "",
        "### Contact evidence tiers",
        "",
        "The family counts above preserve all printed attributions. This second view separates "
        "resolved parents from recheckable but unresolved comparisons and from explicitly "
        "qualified or weak evidence. Families remain non-exclusive.",
        "",
        "| Family | Total labelled | Resolved parent/route | Recheckable source | Qualified | Questioned/weak |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, total in family_involvement.most_common():
        tiers = contact_family_tiers[family]
        recheckable = (
            tiers["explicit-unqualified-source"] + tiers["manually-corroborated-proxy"]
            + tiers["propagated-cluster-label"]
            + tiers["manually-established-route-or-component"]
        )
        qualified = (
            tiers["manually-qualified-proxy"] + tiers["resolved-family-qualified"]
        )
        resolved_or_chain = (
            tiers["resolved-parent-family"] + tiers["resolved-observed-route"]
            + tiers["resolved-family-corroborated"]
        )
        weak_tier = (
            tiers["label-only-source"] + tiers["questioned-source"]
            + tiers["manually-weak-or-unresolved"]
            + tiers["resolved-family-manually-weak"]
        )
        lines.append(
            f"| {family} | {total:,} | {resolved_or_chain:,} | {recheckable:,} | "
            f"{qualified:,} | {weak_tier:,} |"
        )
    lines += [
        "",
        f"Across all families, {contact_tiers['resolved-parent']:,} clusters have an external "
        "parent node. An internal variant edge is not counted as external evidence unless its "
        "curated chain actually terminates at such a node; otherwise the cluster is graded from "
        "its source proxy. The remaining tiers preserve source hypotheses without pretending they "
        f"are equally resolved. Every cluster and basis is listed in "
        f"`{CONTACT_EVIDENCE_TIER_AUDIT_NAME}`; `{FAMILY_CONTACT_EVIDENCE_AUDIT_NAME}` grades "
        "each family separately so one resolved member of a mixed label cannot promote the others.",
        "",
        "A sensitivity bracket makes the consequence of those tiers explicit. The floor retains "
        "only family-specific resolved parents and manually corroborated proxies. "
        "The supported envelope additionally retains qualified resolved links, explicit "
        "unqualified and propagated source evidence, and manually plausible or route-ambiguous "
        "proxies, but excludes label-only, questioned, and manually weak/unresolved cases. "
        "These are evidence thresholds, not statistical confidence intervals.",
        "",
        "| Family | High-specificity floor | Supported envelope | All labelled |",
        "|---|---:|---:|---:|",
    ]
    family_bracket_values = {}
    for family, total in family_involvement.most_common():
        tiers = contact_family_tiers[family]
        floor = sum(tiers[tier] for tier in (
            "resolved-parent-family", "resolved-family-corroborated",
            "manually-corroborated-proxy",
        ))
        excluded = sum(tiers[tier] for tier in (
            "label-only-source", "manually-weak-or-unresolved", "questioned-source",
            "resolved-family-manually-weak",
        ))
        family_bracket_values[family] = (floor, total - excluded, total)
        lines.append(f"| {family} | {floor:,} | {total - excluded:,} | {total:,} |")
    lines += [
        "",
        f"The row-level definitions and shares are in `{FAMILY_EVIDENCE_BRACKET_AUDIT_NAME}`. "
        "The wide gaps, especially for Indo-Aryan and Korku, are a warning against presenting "
        "the headline family labels as solved etymologies.",
        "",
        f"The contrast is historically informative: non-Korku Munda has only "
        f"{family_bracket_values['Munda'][0]:,} high-specificity claims, compared with "
        f"{family_bracket_values['Dravidian'][0]:,} Dravidian and "
        f"{family_bracket_values['Indo-Aryan'][0]:,} Indo-Aryan. Korku's floor is only "
        f"{family_bracket_values['Korku'][0]:,}, but its supported envelope reaches "
        f"{family_bracket_values['Korku'][1]:,}: most Korku evidence identifies a proximate "
        "route in the lexical sources rather than a terminal Korku parent node. This is why the "
        "audit supports profound Korku contact without converting the much smaller non-Korku "
        "Munda set into proof of Munda descent.",
        "",
        f"A separate route-recovery test checks all {len(korku_route_audit):,} Korku-labelled "
        "source-proxy clusters against the independently ingested Korku lexicon, excluding the "
        "provisional proxy forms themselves. It recovers "
        f"{korku_route_assessments['strong-route-match']:,} close form-and-meaning matches and "
        f"{korku_route_assessments['possible-route-match']:,} possible matches; "
        f"{korku_route_assessments['form-only-match']:,} have only a close string, "
        f"{korku_route_assessments['weak-or-unmatched']:,} remain weak or unmatched, and "
        f"{korku_route_assessments['no-recoverable-comparandum']:,} lack a recoverable printed "
        "comparison form. This is deliberately stricter than accepting a dictionary's donor "
        "label at face value.",
        "",
        f"Among the {korku_route_assessments['strong-route-match']:,} strongest route matches, "
        f"{korku_strong_ultimate['Munda']:,} Korku forms have an accepted upstream Munda parent "
        f"and {korku_strong_ultimate['unresolved-korku']:,} have no upstream etymology in the "
        "current graph. The latter count is a Korku etymological-coverage gap, not evidence that "
        "those words originated in Korku. It sharply limits any attempt to convert an immediate "
        "Korku route into an ultimate Munda percentage. Every selected Korku form, score, source, "
        f"and upstream path is in `{KORKU_ROUTE_AUDIT_NAME}`.",
        "",
        f"The Indo-Aryan layer is equally non-monolithic. Of {len(indo_aryan_route_audit):,} "
        f"Indo-Aryan-involved clusters, {indo_aryan_period_classes['modern-ia-explicit']:,} "
        "explicitly name modern Hindi, Marathi, Bengali, or Konkani comparanda, "
        f"{indo_aryan_period_classes['sanskrit-explicit']:,} name Sanskrit without a modern IA "
        f"language, and {indo_aryan_period_classes['historical-and-modern-explicit']:,} name both. "
        f"Another {indo_aryan_period_classes['generic-ia-explicit']:,} use only a generic IA label, "
        f"{indo_aryan_period_classes['resolved-no-specific-source-language']:,} resolve without a "
        f"specific language in the source note, and {indo_aryan_period_classes['ia-label-no-specific-language']:,} "
        f"remain label-only at that granularity. {indo_aryan_korku_routes:,} clusters also mention "
        "a Korku route. These labels distinguish comparative evidence, not borrowing dates, but "
        "they make a single prehistoric Indo-Aryan layer untenable. Row-level named languages, "
        f"parents, and cautions are in `{INDO_ARYAN_ROUTE_AUDIT_NAME}`.",
        "",
        "### Diagnostic basic vocabulary",
        "",
        f"After removing {len(read_dicts(CORE_EXCLUSIONS))} manually verified concept-linking "
        "false positives, a fixed 93-domain "
        f"Swadesh-style screen finds {len(core_audit):,} lexeme clusters linked to "
        f"{len(core_concepts_covered)} covered concepts (HEART and SWIM have no linked Nihali "
        "record). Exclusions are listed in `nihali-core-vocabulary-exclusions.csv`. This is a "
        "diagnostic slice, not a replacement-rate calculation: synonyms and "
        "multiple source forms can create more than one cluster per concept.",
        "",
        "| Stratum | Conservative baseline | Variant-propagation sensitivity |",
        "|---|---:|---:|",
    ]
    core_labels = list(dict.fromkeys([
        *[label for label, _count in core_strata.most_common()],
        *[label for label, _count in core_sensitivity_strata.most_common()],
    ]))
    for label in core_labels:
        baseline_count = core_strata[label]
        sensitivity_count = core_sensitivity_strata[label]
        lines.append(
            f"| {label} | {baseline_count:,} ({baseline_count/len(core_audit):.1%}) | "
            f"{sensitivity_count:,} ({sensitivity_count/len(core_audit):.1%}) |"
        )
    core_residue = core_strata["Nihali residue"]
    sensitivity_residue = core_sensitivity_strata["Nihali residue"]
    residue_root_count = sum(
        int(row["Root_Group_Count"]) for row in core_residue_root_audit
    )
    residue_root_concepts = len(core_residue_root_audit)
    root_replication = Counter(
        row["Replication_Grade"] for row in core_residue_root_inventory
    )
    root_early_attested = sum(
        row["Early_Source_Attested"] == "yes" for row in core_residue_root_inventory
    )
    closed_domain_order = (
        "pronoun", "demonstrative", "interrogative", "polarity", "low numeral",
    )
    closed_class_table = [
        "| Domain | Linked clusters | Residue | Indo-Aryan | Korku | Munda | Dravidian |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in closed_domain_order:
        rows = [row for row in closed_class_audit if row["Domain"] == domain]
        closed_class_table.append(
            f"| {domain} | {len(rows):,} | "
            f"{sum(row['Effective_Stratum'] == 'Nihali residue' for row in rows):,} | "
            f"{sum('Indo-Aryan' in row['Effective_Stratum'].split('+') for row in rows):,} | "
            f"{sum('Korku' in row['Effective_Stratum'].split('+') for row in rows):,} | "
            f"{sum('Munda' in row['Effective_Stratum'].split('+') for row in rows):,} | "
            f"{sum('Dravidian' in row['Effective_Stratum'].split('+') for row in rows):,} |"
        )
    lines += [
        "",
        f"The residue accounts for {core_residue:,} of {len(core_audit):,} core-vocabulary "
        f"clusters ({core_residue/len(core_audit):.1%}), compared with "
        f"{residue_lexemes/len(by_lexeme):.1%} of all lexeme clusters under the strict baseline. "
        f"However, {sum(bool(row['Sensitivity_Stratum']) for row in core_audit):,} manually "
        "reviewed forms are plausible variants or transparent derivational/inflectional relatives "
        "of separately clustered source-attributed Nihali forms. Propagating those comparisons "
        "only in a sensitivity pass "
        f"reduces the core residue to {sensitivity_residue:,} ({sensitivity_residue/len(core_audit):.1%}). "
        "The apparent core enrichment is therefore not robust to conservative under-clustering and "
        "must not be used as positive evidence for an independent lineage. The repeated residue "
        "remains historically important, but its interpretation rests on item-level evidence and "
        "the absence of demonstrated regular correspondences rather than this proportion.",
        "",
        "The same under-clustering risk was screened across the full lexicon. An exhaustive "
        f"review register contains {len(global_variant_sensitivity_audit)} residual-to-contact "
        f"pairs: {global_variant_assessments['variant']} are judged lexical variants, "
        f"{global_variant_assessments['qualified']} remain qualified, and "
        f"{global_variant_assessments['reject']} is rejected. Propagating only the clearer variants "
        f"would reduce the strict residue from {residue_lexemes:,} to "
        f"{residue_lexemes - global_variant_assessments['variant']:,} clusters; including qualified "
        f"cases gives a broad sensitivity floor of "
        f"{residue_lexemes - len(global_supported_variants):,}. The {len(global_supported_variants)} "
        "supported or qualified rows involve "
        f"{global_variant_family_involvement['Indo-Aryan']} Indo-Aryan, "
        f"{global_variant_family_involvement['Korku']} Korku, "
        f"{global_variant_family_involvement['Munda']} Munda, and "
        f"{global_variant_family_involvement['Dravidian']} Dravidian reference strata "
        "(non-exclusive). This sensitivity does not alter installed graph edges: it measures how "
        "much source-level attribution is hidden by conservative lexical clustering. Every pair, "
        "score, decision, and rationale is in "
        f"`{GLOBAL_VARIANT_SENSITIVITY_AUDIT_NAME}`.",
        "",
        "Counting each covered basic concept once gives a more stable picture: "
        f"{core_concept_profiles['residue-only']} concepts are residue-only, "
        f"{core_concept_profiles['residue-plus-contact']} retain both residue and contact forms, "
        f"{core_concept_profiles['contact-only-single-family']} are contact-only with one named "
        f"family, and {core_concept_profiles['contact-only-mixed-family']} are contact-only with "
        "multiple named families. Thus "
        f"{core_concept_profiles['residue-only'] + core_concept_profiles['residue-plus-contact']}/"
        f"{len(core_concept_profile_audit)} concepts retain at least one effective residual root, "
        f"while {core_concept_profiles['contact-only-single-family'] + core_concept_profiles['contact-only-mixed-family']}/"
        f"{len(core_concept_profile_audit)} are contact-only under the current hypotheses. Contact involvement is also "
        f"distributed rather than unitary: Korku occurs in {core_concept_family_involvement['Korku']} "
        f"concepts, Indo-Aryan in {core_concept_family_involvement['Indo-Aryan']}, Munda in "
        f"{core_concept_family_involvement['Munda']}, and Dravidian in "
        f"{core_concept_family_involvement['Dravidian']}. This balance fits layered replacement "
        "better than descent from any single donor layer, but residual presence still means "
        "unmatched rather than inherited. Crucially, "
        f"{sum(row['Any_Multi_Source_Residual_Root'] == 'yes' for row in residue_only_concepts)}/"
        f"{len(residue_only_concepts)} residue-only concepts have at least one root attested in "
        "multiple sources and "
        f"{sum(row['Early_Residual_Root'] == 'yes' for row in residue_only_concepts)}/"
        f"{len(residue_only_concepts)} have a root in Konow or Bhattacharya; the residue-only "
        "concepts supported solely by one source are "
        f"{', '.join(row['Concept'] for row in residue_only_concepts if row['Any_Multi_Source_Residual_Root'] == 'no')}. "
        f"Among the {len(mixed_residue_concepts)} mixed residue-plus-contact concepts, "
        f"{sum(row['Any_Multi_Source_Residual_Root'] == 'yes' for row in mixed_residue_concepts)} "
        "have a replicated residual root. "
        "This establishes a stable residual lexicon without converting it into a demonstrated "
        "lineage. The 91 concept-level rows are in "
        f"`{CORE_CONCEPT_PROFILE_AUDIT_NAME}`.",
        "",
        f"The stricter Korku route-recovery screen contains {len(core_korku_routes)} core-vocabulary "
        f"proxy clusters: {core_korku_route_assessments['strong-route-match']} independently "
        f"recover a close Korku form with compatible meaning, "
        f"{core_korku_route_assessments['possible-route-match']} are possible, "
        f"{core_korku_route_assessments['form-only-match']} recover only a misleading or "
        f"semantically unsupported string, {core_korku_route_assessments['weak-or-unmatched']} "
        f"are weak/unmatched, and {core_korku_route_assessments['no-recoverable-comparandum']} "
        "lack a recoverable form. The source attributions remain recorded, but only the first two "
        "categories independently corroborate a Korku route in the current database.",
        "",
        f"A second manual sensitivity pass collapses the {sensitivity_residue:,} still-residual "
        "clusters to citation-form and dialect-variant root groups within each concept. It yields "
        f"{residue_root_count:,} provisional root hypotheses across {residue_root_concepts:,} "
        "concepts. This is not a Proto-Nihali reconstruction: it is a denominator correction that "
        "prevents forms such as five separately clustered transcriptions of 'head' from counting "
        f"as five independent roots. Groupings and rationales are in "
        f"`{CORE_RESIDUE_ROOT_AUDIT_NAME}`; multi-cluster decisions are maintained in "
        f"`{CORE_RESIDUE_ROOT_REVIEW.name}`.",
        "",
        f"The resulting root inventory contains {root_replication['very strong']:,} roots "
        f"attested in four or five sources, {root_replication['strong']:,} in three, "
        f"{root_replication['moderate']:,} in two, and "
        f"{root_replication['single-source']:,} in one. "
        f"{root_early_attested:,} are present in Konow (1906) or Bhattacharya (1957). "
        "This measures documentary replication, not linguistic age: old loans can be stable, "
        "and single-source items can be genuine. Actual forms, source lists, alternatives, and "
        f"a caution on every row are in `{CORE_RESIDUE_ROOT_INVENTORY_NAME}`.",
        "",
        "Closed-class material is not uniformly replaced. The concept-linked inventory gives "
        "the following effective-stratum profile after applying the reviewed core-variant "
        "sensitivity where available (family columns are non-exclusive):",
        "",
        "\n".join(closed_class_table),
        "",
        "Pronouns, deixis, interrogatives, and polarity retain many residual forms, while the "
        "low numerals are much more heavily assigned to Indo-Aryan or Dravidian. This asymmetry "
        "is compatible with relexification of a pre-contact grammatical lexicon, but it is not "
        "a genetic proof: these linked clusters contain under-merged variants and synonyms, and "
        "contact can affect closed classes too. The complete, row-level diagnostic is in "
        f"`{CLOSED_CLASS_AUDIT_NAME}`.",
        "",
        f"The strict core slice contains {len(core_munda_rows)} Munda-involved clusters covering "
        f"only {len(core_munda_concepts)} concepts. The family-specific audit finds just "
        f"{core_munda_family_tiers['resolved-family-corroborated']} high-specificity corroborated "
        f"Munda claims; {core_munda_family_tiers['resolved-family-qualified']} resolved link is "
        f"qualified, {core_munda_family_tiers['resolved-family-manually-weak']} is manually weak, "
        f"{core_munda_family_tiers['explicit-unqualified-source']} are explicit unhedged source "
        f"comparisons, and {core_munda_family_tiers['label-only-source']} are label-only. "
        "Variant propagation raises the row count to "
        f"{len(sensitivity_munda_rows)} but adds no new concepts "
        f"({len(sensitivity_munda_concepts)} total). This is replication of the same proposed "
        "comparisons, not an expanding cognate set, and it supplies no regular Nihali–Munda sound "
        "correspondence system.",
        "", "### Evidence basis by lexeme cluster", "", "| Evidence basis | Lexeme clusters |",
        "|---|---:|",
    ]
    for label, count in lexeme_methods.most_common():
        lines.append(f"| {label} | {count:,} |")
    lines += [
        "",
        f"The evidence-basis buckets are mutually exclusive. One of the "
        f"{manual_reviews['accept']} accepted reviewed "
        "leads belongs to a multi-record cluster that already contains a curated edge, so it is "
        "counted under `existing-curated` rather than again under manually accepted leads.",
        "",
        f"There are {len(source_variation_audit):,} clusters for which the lexical sources preserve "
        f"different donor labels. In {variation_relationships['nested-route/ultimate']} cases one "
        "label nests inside another, typically reflecting immediate Korku transmission versus an "
        "Indo-Aryan or other ultimate source. The remaining "
        f"{variation_relationships['disjoint']} cases have disjoint labels. Manual adjudication "
        f"treats {disjoint_assessments['compatible-contact-chain']} as plausible staged contact "
        "chains, favors Indo-Aryan in "
        f"{disjoint_assessments['favor-indo-aryan']}, Korku in "
        f"{disjoint_assessments['favor-korku']}, and Munda in "
        f"{disjoint_assessments['favor-munda']}; {disjoint_assessments['unresolved']} remain "
        "unresolved. The union labels stay in the audit so this adjudication does not erase the "
        f"printed alternatives. Full evidence and rationales are in `{SOURCE_VARIATION_AUDIT_NAME}`.",
        "",
        f"Cross-dictionary propagation extends source/editorial evidence to "
        f"{propagated_records:,} records "
        f"in {propagated_clusters:,} conservatively matched clusters.",
        "",
        f"All {manual_triggers['separated-candidate'] + manual_triggers['low-margin-family-tie']} "
        "threshold-generated source-free leads received an explicit human decision: "
        f"{manual_triggers['separated-candidate']} had a clear margin over alternatives and "
        f"{manual_triggers['low-margin-family-tie']} were family-level ties. A separate reviewed "
        f"escape hatch covers {manual_triggers['transparent-cultural-loan']} transparent cultural "
        "loans that string scoring misses. A fourth register manually checks "
        f"{manual_triggers['source-attributed-parent']} low-fit or semantically deceptive parent "
        "resolutions where "
        "the donor attribution itself remains source-supported. Across the four registers, "
        f"{manual_reviews['accept']} were accepted, {manual_reviews['reject']} rejected, and "
        f"{manual_reviews['defer']} deferred. Decisions and rationales are preserved in the audit "
        f"tables, including `{TRANSPARENT_LOAN_REVIEW.name}` for the cultural-loan exceptions "
        f"and `{SOURCE_PARENT_REVIEW.name}` for source-parent corrections. Rejected and deferred "
        "source-free leads remain Nihali-residue proxy hypotheses; rejected source-parent matches "
        "retain their printed donor stratum as unresolved proxies. Within the parent-head register, "
        f"{source_parent_decisions['accept']} links were redirected to the source-supported head "
        f"and {source_parent_decisions['reject']} were downgraded to unresolved proxies.",
        "",
        f"A separate morpheme-level review finds {len(residue_contact_component_review)} "
        "otherwise residual expressions with recognizable contact material: "
        f"{residue_component_assessments['transparent-component']} transparent components, "
        f"{residue_component_assessments['qualified-component']} qualified components, and "
        f"{residue_component_assessments['source-gloss-conflict']} quarantined source/gloss "
        "conflict. Layer mentions are non-exclusive ("
        + ", ".join(
            f"{layer} {count}" for layer, count in sorted(residue_component_layers.items())
        )
        + "). These rows remain residual in the strict graph because component analysis alone "
        "does not create a whole-expression rank-1 edge. Most retain unresolved Nihali material; "
        "a fully contact-composed case is removed only in the explicit sensitivity pass. See "
        f"`{RESIDUE_CONTACT_COMPONENT_REVIEW.name}`.",
        "",
        "### Residue threshold sensitivity",
        "",
        "A score-only stress test deliberately relaxes the production gates for clusters that "
        "still remain residual. It does not create assignments:",
        "",
        "| Threshold | Minimum score | Minimum margin | Flagged | Indo-Aryan | Dravidian | Munda | Unreviewed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in residue_threshold_sensitivity:
        lines.append(
            f"| {row['Threshold_Label']} | {float(row['Minimum_Composite_Score']):.3f} | "
            f"{float(row['Minimum_Margin']):.3f} | {int(row['Flagged_Residue_Clusters']):,} | "
            f"{int(row['Indo_Aryan_Candidates']):,} | {int(row['Dravidian_Candidates']):,} | "
            f"{int(row['Munda_Candidates']):,} | {int(row['Unreviewed']):,} |"
        )
    production_like = residue_sensitivity_by_label["production-like score/margin only"]
    very_high = residue_sensitivity_by_label["very high score"]
    lines += [
        "",
        "At the production-like composite score and margin alone, "
        f"{production_like['Flagged_Residue_Clusters']} residual clusters would be flagged, but "
        f"{production_like['Unreviewed']} never passed the required component-level form, meaning, "
        "and donor-specific gates. At the very-high score setting, "
        f"{very_high['Flagged_Residue_Clusters']} remain, of which "
        f"{int(very_high['Manual_Rejected']) + int(very_high['Manual_Deferred'])} were already "
        "rejected or deferred by manual review. Looser settings mainly generate Indo-Aryan and "
        "Dravidian candidates, reflecting donor-database size as much as history; etymologised "
        "Korku coverage is especially sparse. This is why lowering one cutoff cannot convert the "
        f"residue into evidence for a particular family. Full definitions are in "
        f"`{RESIDUE_THRESHOLD_SENSITIVITY_NAME}`.",
        "",
        "### Shape of resolved external links",
        "",
        "The following table counts distinct lexeme-to-parent links that reach an actual external "
        "Jambu node, excluding unresolved donor proxies. Parent shape compares the Nihali form "
        "with the stored parent, which may be reconstructed or ultimate. Surface shape instead "
        "uses the best automatically selected observed descendant under that parent. Both are "
        "descriptive, not sound-law tests.",
        "",
        "| Parent family | Links | Parent exact/near | Surface exact/near | Surface moderate | "
        "Surface distant/unavailable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, count in resolved_parent_families.most_common():
        shapes = resolved_match_shapes[family]
        surface_shapes = resolved_surface_match_shapes[family]
        surface_unavailable = count - sum(surface_shapes.values())
        lines.append(
            f"| {family} | {count:,} | {shapes['exact'] + shapes['near']:,} | "
            f"{surface_shapes['exact'] + surface_shapes['near']:,} | "
            f"{surface_shapes['moderate']:,} | "
            f"{surface_shapes['distant'] + surface_unavailable:,} |"
        )
    lines += [
        "",
        "Resolution is mostly to a comparative index or reconstructed node, not to an observed "
        "surface donor: "
        f"{resolved_parent_languages['Indo-Aryan']:,} links terminate at the generic "
        f"Indo-Aryan node, {resolved_parent_languages['Proto-Indo-Iranian']:,} at "
        f"Proto-Indo-Iranian, all {resolved_parent_languages['Proto-Dravidian']:,} Dravidian "
        f"links at Proto-Dravidian, and the Munda links split between "
        f"Proto-Kherwarian ({resolved_parent_languages['Proto-Kherwarian']:,}) and Proto-Munda "
        f"({resolved_parent_languages['Proto-Munda']:,}), with "
        f"{resolved_parent_languages['English']:,} direct English and "
        f"{resolved_parent_languages['Persian']:,} Persian links. Thus 'resolved' means that a "
        "database "
        "etymon was identified; it does not by itself identify the immediate donor, borrowing "
        "date, or direction.",
        "The observed-surface column is a reproducible diagnostic rather than manual donor "
        "identification. It can recover a relevant reflex hidden beneath an awkward canonical "
        "display form, but dense etymological families can also supply an accidentally attractive "
        "surface. The selected IDs, forms, languages, glosses, and both similarity scores remain "
        f"inspectable in `{RESOLVED_CONTACT_SHAPE_AUDIT_NAME}`.",
    ]
    lines += [
        "",
        "The source labels and resolved parents answer different historical questions. The matrix "
        "below counts which named route families occur in the source stratum of each resolved "
        "ultimate-parent link; route columns are non-exclusive.",
        "",
        "| Resolved parent family | Links | Korku-labelled route | Munda-labelled route | "
        "Indo-Aryan-labelled route | Dravidian-labelled route |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in ("Indo-Aryan", "Dravidian", "Munda", "Other", "English"):
        routes = route_family_involvement[family]
        lines.append(
            f"| {family} | {resolved_parent_families[family]:,} | {routes['Korku']:,} | "
            f"{routes['Munda']:,} | {routes['Indo-Aryan']:,} | {routes['Dravidian']:,} |"
        )
    lines += [
        "",
        f"Most importantly, {route_family_involvement['Indo-Aryan']['Korku']} of the "
        f"{resolved_parent_families['Indo-Aryan']} links that resolve to an Indo-Aryan parent "
        "carry a Korku route label, and "
        f"{route_family_involvement['Munda']['Korku']} of the "
        f"{resolved_parent_families['Munda']} Munda-parent links do so. This is direct database "
        "evidence that Korku is often the proximate conduit rather than a sufficient statement of "
        "ultimate ancestry.",
    ]
    munda_shapes = resolved_match_shapes.get("Munda", Counter())
    lines += [
        "",
        f"The Munda subset has {resolved_parent_families['Munda']:,} resolved links, of which "
        f"{munda_shapes['exact'] + munda_shapes['near']:,} are exact or near. Near identity is "
        "compatible with borrowing, while the more distant pairs require recurring correspondences "
        "before they can support inheritance. Manual review reduces the "
        f"{resolved_parent_families['Munda']} links to "
        f"{munda_parent_roots} distinct Munda parent roots and classifies "
        f"{munda_review['near-contact-compatible']} as near contact-compatible, "
        f"{munda_review['possible-correspondence']} as possible correspondences, and "
        f"{munda_review['weak-comparison']} as weak. The only repeated non-identity proposal is "
        f"initial c~s in {munda_series['c~s']} links representing four parent roots; one root "
        "('dance') is duplicated across two Nihali citation clusters. Four semantic sets are too "
        "few, and their remaining segments too heterogeneous, to establish a Nihali–Munda sound "
        "law. The full form-shape inventory and all "
        f"{resolved_parent_families['Munda']} rationales are in "
        f"`{RESOLVED_CONTACT_SHAPE_AUDIT_NAME}` and `{MUNDA_CORRESPONDENCE_REVIEW.name}`.",
        "",
        f"The parallel Dravidian diagnostic collapses "
        f"{sum(int(row['Link_Count']) for row in dravidian_correspondence_audit)} resolved links to "
        f"{len(dravidian_correspondence_audit)} parent roots. On a deliberately mechanical "
        f"joint form-and-meaning screen, {dravidian_review['near-contact-compatible']} roots are "
        f"near contact-compatible and {dravidian_review['possible-comparison']} are possible "
        f"comparisons, while {dravidian_review['weak-form-link']} have weak form fit and "
        f"{dravidian_review['weak-semantic-link']} have weak semantic fit. "
        f"{dravidian_series_types['identity-initial']} roots begin with identity mappings; the "
        f"only repeated non-identity initials are {dravidian_repeated_series}. These sparse "
        "patterns mix unrelated meanings and do not establish a Nihali–Dravidian sound law. "
        "The diagnostic nevertheless supports a genuine contact layer: close matches such as "
        "the low numerals, 'dog', 'cat', and 'cotton' are exactly the shapes that borrowing can "
        "preserve. Root-level rows and the reproducible thresholds are in "
        f"`{DRAVIDIAN_CORRESPONDENCE_AUDIT_NAME}`.",
        "",
        "## Record-level audit results",
        "",
        "| Stratum | Records | Share |",
        "|---|---:|---:|",
    ]
    for label, count in strata.most_common():
        lines.append(f"| {label or 'Unclassified'} | {count:,} | {count/len(audit):.1%} |")
    lines += [
        "", "### Method", "", "| Method | Records |", "|---|---:|",
    ]
    for label, count in methods.most_common():
        lines.append(f"| {label} | {count:,} |")
    lines += ["", "### Confidence", "", "| Confidence | Records |", "|---|---:|"]
    for label, count in confidence.most_common():
        lines.append(f"| {label} | {count:,} |")
    weak_count = confidence["low"] + confidence["unresolved"]
    lines += [
        "",
        f"Low or unresolved analyses account for {weak_count:,} records "
        f"({weak_count/len(audit):.1%}). Confidence grades the resolution of a particular parent, "
        "not the mere presence of a donor label in an editorial/source note; these proportions "
        "must not be presented as a fully solved etymological dictionary.",
        "",
        "### Resolution quality within donor proxies",
        "",
        f"The {len(source_proxy_quality_audit):,} lexeme clusters containing at least one unresolved "
        "donor proxy are separately graded below. A catalog-indexed or explicit comparison can be "
        "rechecked against a named form; a donor-label-only proxy preserves a source judgment but "
        "does not reveal the comparison that motivated it.",
        "",
        "| Proxy evidence | Clusters | Share of proxy clusters |",
        "|---|---:|---:|",
    ]
    for label, count in proxy_quality.most_common():
        lines.append(
            f"| {label} | {count:,} | {count/len(source_proxy_quality_audit):.1%} |"
        )
    lines += [
        "",
        "A surface-form check against the best recoverable printed comparandum provides a second, "
        "independent triage. It is deliberately descriptive: compounds, historical sound change, "
        "and reconstructed citation forms can make genuine contacts look distant, while near "
        "identity alone cannot establish direction or inheritance.",
        "",
        "| Best surface shape | All proxies | Unqualified | Questioned |",
        "|---|---:|---:|---:|",
    ]
    for shape in ("exact", "near", "moderate", "distant", "unscored"):
        lines.append(
            f"| {shape} | {proxy_comparison_shapes[shape]:,} | "
            f"{proxy_shape_by_uncertainty['unqualified'][shape]:,} | "
            f"{proxy_shape_by_uncertainty['questioned'][shape]:,} |"
        )
    lines += [
        "",
        f"Only {reproducible_proxy:,} proxy clusters "
        f"({reproducible_proxy/len(source_proxy_quality_audit):.1%}) combine recoverable comparanda "
        "with no explicit uncertainty marker. "
        f"{proxy_uncertainty['questioned']:,} contain an explicit question or hedge; "
        f"{proxy_quality['donor-label-only'] + proxy_quality['propagated-only']:,} give no "
        "recoverable compared form at cluster level. Direction is explicitly asserted for "
        f"{proxy_directionality['donor/source asserted']:,}, while "
        f"{proxy_directionality['comparison only']:,} are comparison-only and the remainder are "
        "attributions without a stated direction. These distinctions are tabulated in "
        f"`{SOURCE_PROXY_QUALITY_AUDIT_NAME}`. The "
        f"{proxy_priority['critical']:,} critical-priority rows are weakly supported core-vocabulary "
        "proxies. Manual review corroborates contact in "
        f"{critical_proxy_assessments['corroborated-contact']}, finds the route ambiguous in "
        f"{critical_proxy_assessments['route-ambiguous-contact']}, retains "
        f"{critical_proxy_assessments['plausible-contact']} as plausible, and leaves "
        f"{critical_proxy_assessments['unresolved']} unresolved. The unresolved cases do not count "
        "as positive evidence for genetic classification; the row-level reasoning is preserved in "
        f"`{CORE_SOURCE_PROXY_REVIEW.name}` and repeated in the quality audit.",
        "",
        f"The {proxy_priority['high']:,} high-priority weak proxies involving Munda or Dravidian "
        "also received full manual review: "
        f"{diagnostic_proxy_assessments['corroborated-contact']} corroborated contact items, "
        f"{diagnostic_proxy_assessments['route-ambiguous-contact']} route-ambiguous items, "
        f"{diagnostic_proxy_assessments['plausible-contact']} plausible items, "
        f"{diagnostic_proxy_assessments['weak-comparison']} weak comparisons, and "
        f"{diagnostic_proxy_assessments['unresolved']} unresolved labels. This prevents questioned "
        "family tags from being mistaken for equal-strength evidence; preferred donors and full "
        f"rationales are in `{DIAGNOSTIC_SOURCE_PROXY_REVIEW.name}`.",
        "",
        f"The remaining {len(questioned_review_ids):,} explicitly hedged proxies were then reviewed "
        "one by one: "
        f"{questioned_proxy_assessments['corroborated-contact']} corroborated, "
        f"{questioned_proxy_assessments['route-ambiguous-contact']} route-ambiguous, "
        f"{questioned_proxy_assessments['plausible-contact']} plausible, "
        f"{questioned_proxy_assessments['weak-comparison']} weak, and "
        f"{questioned_proxy_assessments['unresolved']} unresolved. Thus no explicitly questioned "
        "proxy remains unreviewed. These decisions grade the source comparison without silently "
        "rewriting its graph label; the complete register is "
        f"`{QUESTIONED_SOURCE_PROXY_REVIEW_NAME}`.",
    ]
    lines += [
        "", "### Lexical sources and ascertainment", "",
        "The sources differ sharply in whether they print or inherit etymological commentary. "
        "Rows therefore measure documentation practices as well as the language.",
        "",
        "| Source | Records | Direct donor note | Current external label | Residue |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, count in sources.most_common():
        source_group = [row for row in audit if row["Lexical_Source"] == label]
        own = sum(bool(row["Own_Source_Attribution"]) for row in source_group)
        source_external = sum(
            row["Stratum"] not in {"Nihali residue", "Other", ""} for row in source_group
        )
        source_residue = sum(row["Stratum"] == "Nihali residue" for row in source_group)
        lines.append(
            f"| {label} | {count:,} | {own:,} ({own/count:.1%}) | "
            f"{source_external:,} ({source_external/count:.1%}) | "
            f"{source_residue:,} ({source_residue/count:.1%}) |"
        )
    lines += [
        "",
        "Deduplicating within each source gives the following source-normalized profile. Family "
        "columns are non-exclusive, and cross-dictionary propagation is retained because the "
        "question here is the current hypothesis coverage of each source, not authorship of the "
        "label.",
        "",
        "| Source | Clusters | External | Korku | Indo-Aryan | Munda | Dravidian | Residue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source_profile_audit:
        lines.append(
            f"| {row['Lexical_Source']} | {int(row['Lexeme_Clusters']):,} | "
            f"{float(row['External_Share']):.1%} | {float(row['Korku_Share']):.1%} | "
            f"{float(row['Indo_Aryan_Share']):.1%} | {float(row['Munda_Share']):.1%} | "
            f"{float(row['Dravidian_Share']):.1%} | {float(row['Residue_Share']):.1%} |"
        )
    source_profile_by_name = {
        row["Lexical_Source"]: row for row in source_profile_audit
    }
    munda_shares = [float(row["Munda_Share"]) for row in source_profile_audit]
    lines += [
        "",
        "The normalized contrast is too large to read chronologically. Konow has only "
        f"{float(source_profile_by_name['konow1906']['External_Share']):.1%} external coverage, "
        "while Nagaraja has "
        f"{float(source_profile_by_name['nagaraja2014']['External_Share']):.1%}; Nagaraja also "
        "supplies the densest Korku commentary. Non-Korku Munda remains a small share in every "
        f"source ({min(munda_shares):.1%}–{max(munda_shares):.1%}), rather than "
        "becoming a dominant layer when dictionary size is controlled. Location, elicitation "
        "scope, and editorial policy remain confounded; complete counts are in "
        f"`{SOURCE_PROFILE_AUDIT_NAME}`.",
        "",
        f"Direct attribution agreement is still sparser. Among the "
        f"{len(cross_source_agreement_audit):,} clusters attested in at least two sources, only "
        f"{cross_source_agreement['multi-source-exact-agreement']:,} receive the same direct "
        "family label from multiple sources and "
        f"{cross_source_agreement['multi-source-nested-compatible']:,} receive compatible nested "
        f"labels. {cross_source_agreement['multi-source-disjoint']:,} have disjoint direct labels; "
        f"{cross_source_agreement['single-labelled-source']:,} are labelled by only one source, "
        f"and {cross_source_agreement['no-direct-label']:,} have no direct label in their own "
        "replicated rows. Silence is not disagreement, and matching labels are not fully "
        "independent because later dictionaries can repeat earlier analyses. The result explains "
        "why source hypotheses are retained but not treated as independently replicated cognate "
        f"judgments; see `{CROSS_SOURCE_AGREEMENT_AUDIT_NAME}`.",
        "",
        "Family-specific replication makes that distinction explicit:",
        "",
        "| Family | All labelled | Lexeme in 2+ sources | Family directly labelled in 2+ sources |",
        "|---|---:|---:|---:|",
    ]
    for row in family_attribution_replication_audit:
        lines.append(
            f"| {row['Family']} | {int(row['All_Labelled_Clusters']):,} | "
            f"{int(row['Multi_Source_Clusters']):,} | "
            f"{int(row['Two_Plus_Direct_Labelled_Sources']):,} "
            f"({float(row['Two_Plus_Share_Of_Multi_Source']):.1%}) |"
        )
    lines += [
        "",
        "Only 5 of the 51 replicated Munda-labelled lexemes receive a direct Munda label in two "
        "or more sources, compared with 58/323 for Korku and 162/348 for Indo-Aryan. Thus repeated "
        "attestation of a proposed Munda item usually replicates the Nihali word, not the Munda "
        "analysis. Full zero/one/two-plus counts and the dependency warning are in "
        f"`{FAMILY_ATTRIBUTION_REPLICATION_AUDIT_NAME}`.",
        "",
        "Cross-source replication by layer is also non-exclusive for the named contact families. "
        "‘Early source’ here means attested in Konow (1906) or Bhattacharya (1957), not that the "
        "etymon itself has been dated.",
        "",
        "| Layer | Clusters | In 2+ sources | Early-source attested | Nagaraja 2014 | All five |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in layer_replication_audit:
        lines.append(
            f"| {row['Layer']} | {int(row['Total_Clusters']):,} | "
            f"{int(row['Multi_Source_Clusters']):,} ({float(row['Multi_Source_Share']):.1%}) | "
            f"{int(row['Konow_Or_Bhattacharya_Attested']):,} | "
            f"{int(row['Nagaraja_Attested']):,} | {int(row['All_Five_Sources']):,} |"
        )
    lines += [
        "",
        "The contact layers replicate across sources at roughly 27–33%, versus 12% for the strict "
        "residue. That does not make the loans older than the residue: named comparisons are easier "
        "to propagate across dictionaries, and the five sources sample different places, times, "
        "and editorial traditions. The table is therefore a documentation-stability check, not a "
        f"loan chronology; its rows are in `{LAYER_REPLICATION_AUDIT_NAME}`.",
    ]
    lines += [
        "",
        "The database's linked concept categories show that the strata are not distributed "
        "uniformly across the lexicon. Categories are non-exclusive.",
        "",
        "| Layer | Concept-linked | Noun | Verb | Adjective | Numeral | Other |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in layer_category_audit:
        lines.append(
            f"| {row['Layer']} | {int(row['Concept_Linked_Clusters']):,} | "
            f"{int(row['Noun_Clusters']):,} | {int(row['Verb_Clusters']):,} | "
            f"{int(row['Adjective_Clusters']):,} | {int(row['Numeral_Clusters']):,} | "
            f"{int(row['Other_Clusters']):,} |"
        )
    layer_category_by_name = {row["Layer"]: row for row in layer_category_audit}
    ia_category = layer_category_by_name["Indo-Aryan"]
    korku_category = layer_category_by_name["Korku"]
    residue_category = layer_category_by_name["Nihali residue"]
    lines += [
        "",
        "Among concept-linked clusters, Indo-Aryan is strongly noun-heavy "
        f"({ia_category['Noun_Clusters']} nouns versus {ia_category['Verb_Clusters']} verbs), "
        f"as is Korku ({korku_category['Noun_Clusters']} versus "
        f"{korku_category['Verb_Clusters']}), whereas the residue is more predicate-rich "
        f"({residue_category['Noun_Clusters']} nouns versus "
        f"{residue_category['Verb_Clusters']} verbs). This is compatible with layered lexical "
        "replacement rather than "
        "a single uniform donor process. It is not decisive: concept links are incomplete, source "
        "editors differ in coverage, and Dravidian and non-Korku Munda labels are themselves "
        "verb-rich. Full counts and cautions are in "
        f"`{LAYER_CATEGORY_AUDIT_NAME}`.",
        "",
        "Surface word shape likewise fails to isolate a unique residual phonotactic system:",
        "",
        "| Layer | Mean folded length | Final vowel | Multiword/compound | Retroflex | Aspiration | Nasalization |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in layer_form_shape_audit:
        lines.append(
            f"| {row['Layer']} | {float(row['Mean_Folded_Length']):.2f} | "
            f"{float(row['Final_Vowel_Share']):.1%} | "
            f"{float(row['Multiword_Or_Compound_Share']):.1%} | "
            f"{float(row['Retroflex_Share']):.1%} | "
            f"{float(row['Aspiration_Share']):.1%} | "
            f"{float(row['Nasalization_Share']):.1%} |"
        )
    lines += [
        "",
        "Every layer has median folded length 5; residue and contact strata broadly overlap on "
        "final vowels, retroflexion, and compounding. Residual aspiration is lower than in the "
        "Korku and Indo-Aryan layers, but this unmodelled difference can reflect loan adaptation, "
        "morphological suffixes, and transcription practice. The negative result prevents the "
        "residue's surface profile from being used as a surrogate family classifier. Definitions "
        f"and counts are in `{LAYER_FORM_SHAPE_AUDIT_NAME}`.",
    ]
    lines += [
        "",
        f"Only {len(multi_source_clusters):,} of {len(by_lexeme):,} clusters are independently "
        f"attested in at least two lexical sources; {multi_source_residue:,} of those replicated "
        "clusters remain residue throughout. Cross-source stability confirms that these are real "
        "Nihali lexical items, but it does not by itself distinguish inheritance from an old "
        "unrecognized loan. The complete replicated-residue register contains "
        f"{replicated_residue_grades['very strong']:,} very-strong, "
        f"{replicated_residue_grades['strong']:,} strong, and "
        f"{replicated_residue_grades['moderate']:,} moderate replication grades; "
        f"{replicated_core_residue:,} are core clusters that remain residual after variant "
        f"sensitivity. See `{REPLICATED_RESIDUE_AUDIT_NAME}`.",
    ]
    lines += [
        "",
        "## Interpretation",
        "",
        "1. **Indo-Aryan is widest; Korku is the diagnostic contact center.** Indo-Aryan appears "
        "in the largest number of expanded-database clusters, but Korku dominates Kuiper's older "
        "core sample and supplies the clearest community-specific transmission layer. A "
        "Korku-labelled match is best "
        "read as the route of transmission, not necessarily the ultimate origin: Korku itself "
        "contains Indo-Aryan loans, so counting every such item as ultimately Munda would inflate "
        "the Munda layer.",
        "2. **The Indo-Aryan layer is chronologically mixed.** It contains inherited Indo-Aryan "
        "etyma reached through Marathi/Hindi reflexes, recent Hindi/Marathi cultural vocabulary, "
        "and English loans often mediated by Indo-Aryan. Surface donor and ultimate etymon must "
        "therefore remain separate questions.",
        "3. **The Dravidian layer is real but comparatively small and heterogeneous.** Some proposed "
        "comparisons are explicitly uncertain, and apparent Dravidian material may have passed "
        "through neighbouring Indo-Aryan or Munda varieties.",
        "4. **The residue is historically important.** Repeated basic vocabulary survives there, "
        "but this "
        "analysis cannot turn a set of unmatched forms into a demonstrated Proto-Nihali lexicon. "
        "That requires recurrent sound correspondences across internal dialect evidence or an "
        "external relative, neither of which is presently available at the necessary scale.",
        "5. **An argot-only account is unnecessary.** Some disguising or replacement vocabulary "
        "may exist, but a wholesale argot hypothesis predicts neither the persistent basic residue "
        "nor the layered, source-specific contact profile as economically as relexification of an "
        "independent language does.",
        "",
        "### Repeated residue examples",
        "",
        "The following are examples attested by at least three distinct lexical sources and not "
        "assigned an external source here. Stability across dictionaries supports their reality as "
        "Nihali lexemes; it does **not** by itself prove inheritance.",
        "",
        "| Form | Gloss | Records | Sources |",
        "|---|---|---:|---|",
    ]
    for form, gloss, count, source_names in stable_residue[:20]:
        lines.append(f"| {form} | {gloss.replace('|', '/')} | {count} | {source_names} |")
    lines += [
        "",
        "## Method and safeguards",
        "",
        "Existing rank-1 Jambu edges were retained. Repeated attestations were clustered only when "
        "their normalized forms and meanings agree, or when a near-identical form has compatible "
        "gloss evidence across distinct sources. Record-local donor evidence may propagate within such a "
        "cluster. Candidate resolution still requires compatible donor family, semantic overlap, "
        "phonological similarity, and a margin over alternatives. Unresolved printed comparisons "
        "become donor proxies rather than invented links.",
        "The clustering deliberately under-merges synonym-only glosses rather than consulting the "
        "generated concept table, so a later database build cannot change its own lexical units.",
        "",
        "Source spellings and diacritics are retained in the installed forms; accent folding and "
        "Unicode normalization are used only for candidate retrieval and clustering. The one "
        "catalog-number emendation made during this audit is `tongre/ṭongre` 'knee(-cup)': its "
        "printed DED(S) 2419 citation corresponds to DEDR 2983, whose Naikri `ṭoŋgre` is the exact "
        "comparison, rather than DEDR 2419 'neck'. The correction is documented in the source row. "
        "No printed lexical form was silently respelled. The contradictory `katʰarnāk` source/gloss "
        "case remains quarantined rather than normalized into a convenient etymology.",
        "",
        "Records with no printed or cross-dictionary donor evidence are searched for review leads. "
        "Every machine lead that cleared the conservative threshold was then manually accepted, "
        "rejected, or deferred in `nihali-computational-candidate-review.csv` and "
        "`nihali-low-margin-candidate-review.csv`. Only accepted leads "
        "become external links; rejected and deferred leads remain Nihali-residue proxies. This "
        "prevents accidental short-form and remote-language coincidences from driving the historical "
        "conclusion. The complete scores, margins, alternatives, source wording, manual rationales, "
        f"cluster IDs, and record IDs are in `{AUDIT_NAME}`; `{CLUSTER_AUDIT_NAME}` provides the "
        "deduplicated review view.",
        "",
        "## Comparison with Kuiper's baseline",
        "",
        "Kuiper's 1962 study classified a 503-item vocabulary as 180 direct Korku loans (36%), "
        "about 20 possible remnants of an earlier Munda layer (roughly 4%), 47 Dravidian items "
        "(9%), and 123 items without any known Indian correspondence (about 24%). He explicitly "
        "warned that incomplete Korku documentation and subjective borderline decisions make these "
        "figures approximate. The present database is over eight times larger, combines five "
        "partly overlapping sources, and distinguishes immediate donor from ultimate ancestry; its "
        "percentages therefore test the shape of his model but are not a direct replication.",
        "",
        "## Evidence synthesis",
        "",
        "No single percentage decides the origin question. The matrix below records what each "
        "independent diagnostic can and cannot support.",
        "",
        "| Domain | Finding | Evidential weight |",
        "|---|---|---|",
    ]
    for row in origin_evidence_matrix:
        lines.append(
            f"| {row['Domain']} | {row['Finding']} | {row['Evidential_Weight']} |"
        )
    lines += [
        "",
        "The full matrix also states the hypothesis supported, the alternative challenged, the "
        f"limitation, and the underlying audit or publication for every row: "
        f"`{ORIGIN_EVIDENCE_MATRIX_NAME}`.",
        "",
        "## Competing origin hypotheses",
        "",
        "| Hypothesis | Fit to this lexical audit | Provisional confidence | Main unresolved test |",
        "|---|---|---|---|",
        "| Independent lineage with layered relexification | Best overall fit: stable residue plus "
        "separable Indo-Aryan, Korku/Munda, and Dravidian contact layers | Moderate, as a diagnosis "
        "by exclusion | Demonstrate internal history or identify an external relative with regular "
        "correspondences |",
        "| Direct Munda affiliation | Explains some lexicon and areal/morphological traits, but the "
        "resolved lexical set lacks a recurring inherited correspondence system | Low on present "
        "lexical evidence; not excluded | Reconstruct shared innovations not attributable to Korku "
        "contact |",
        "| Indo-Aryan or Dravidian affiliation | Poor fit: both behave as stratified donor layers, "
        "not as the source of the whole basic lexicon | Very low | Find inherited morphology and "
        "regular core cognates outside the known loan strata |",
        "| Argot-only origin | Can explain some substitution or deformation but not the whole "
        "cross-source lexical system | Low as a complete account; plausible as a secondary process "
        "| Identify productive disguise rules and their recoverable base forms |",
        "| Relationship to another isolate or macrofamily | Published comparisons are sparse and "
        "non-systematic | Unsupported | Establish multiple exclusive, semantically controlled "
        "correspondence series |",
        "",
        "- **Independent-lineage/relexification hypothesis.** Kuiper's unidentified component and "
        "Zide's later appraisal treat the non-loan residue as potentially representing a lineage "
        "without a demonstrated living relative. This best predicts a stable core residue together "
        "with several donor-specific layers, but it remains a diagnosis by exclusion rather than a "
        "comparative reconstruction.",
        "- **Munda-branch hypothesis.** Mundlay argued in the same 1996 volume for placing Nihali "
        "directly under Proto-Munda but outside the Northern and Southern branches, using lexical, "
        "grammatical, and ethnographic evidence. The present lexical audit finds substantial "
        "Korku/Munda material, but much of it is explicitly contact-attributed and it does not "
        "produce the regular inherited correspondence set needed to choose this genetic account. "
        "Ilia Peiros's independent core-list appraisal in the same volume found only nine Munda "
        "comparisons with a scattered distribution and likewise judged the relationship "
        "unconvincing. Morphology remains the strongest counterargument to an isolate analysis, "
        "but Mundlay's own presentation says the grammatical evidence is largely negative, the "
        "resemblances are not close, and structural erosion obscures the system. The lexical "
        "database cannot adjudicate inherited versus contact-induced grammar, so a future "
        "morpheme-by-morpheme reconstruction is essential.",
        "- **Argot or deliberately disguised register.** Socially restricted vocabulary and "
        "semantic replacement may explain some forms, but do not by themselves explain the "
        "cross-source stability of residue terms for body parts, natural phenomena, pronouns, and "
        "basic predicates. The audit therefore treats argot formation as a secondary process, not "
        "a complete origin account.",
        "",
        "### Structural sensitivity check (not a family test)",
        "",
        "As an external control on the grammatical argument, 164 coded Nihali features in "
        "[Grambank](https://github.com/grambank/grambank/commit/"
        "37f73da55cf8b426c82383f46a972bc59ce6cf76) were compared with selected regional languages. "
        "Pairwise raw agreement is 74.4% with Korku (129 shared coded features), 73.2–77.1% with "
        "five other Munda languages, 73.8% with Marathi, 78.5% with Hindi, 77.4–81.0% with four "
        "selected Dravidian languages, and 79.6% with Kusunda. Chance-corrected kappa and "
        "positive-feature Jaccard measures preserve the same basic warning: no uniquely Munda "
        "cluster emerges from this small control.",
        "Each pair uses only features coded for both languages; kappa corrects for marginal-value "
        "agreement and Jaccard compares features whose value is 1. No imputation, feature "
        "weighting, or optimization was used.",
        "",
        "These figures cannot identify ancestry. Grambank features are synchronic, structurally "
        "dependent, unevenly missing, and highly susceptible to areal convergence; the comparison "
        "also lacks a phylogenetic or spatial model. Its value is negative: a broad claim that "
        "Nihali 'looks Munda' is not by itself discriminating evidence. Only shared innovations and "
        "morpheme histories can decide the morphological question. Inputs, shared-feature counts, "
        "and all three descriptive measures are preserved in "
        "`nihali-grambank-structural-sensitivity.csv`.",
        "",
        "The provisional adjudication is consequently **independent lineage with profound Munda "
        "contact**, not because Munda affiliation is impossible, but because the positive evidence "
        "required to demonstrate it is still missing.",
        "",
        "## Scholarly context",
        "",
        "The interpretation agrees in broad outline with Kuiper's stratified treatment and with "
        "Zide's later caution: massive Korku-mediated relexification can coexist with an independent "
        "residue, while isolated similarities to Tibeto-Burman or wider Austroasiatic do not by "
        "themselves establish genetic affiliation. Zide further stresses that directionality is "
        "not always obvious for the small set of South Munda parallels. Those cautions are built "
        "into the proxy and confidence scheme here.",
        "The current [Glottolog 5.3 classification](https://glottolog.org/resource/languoid/id/niha1238) "
        "also leaves Nihali as a standalone top-level entry rather than placing it inside Munda. "
        "That is corroborating scholarly practice, not independent proof of the hypothesis.",
        "Shailendra Mohan's 2016 documentation report likewise describes several historical and "
        "local contact layers and says that proposed links to Kusunda, Ainu, Nostratic-Dravidian, "
        "and Greater Austric had not yielded conclusive genetic evidence. This audit does not "
        "retest those remote proposals form by form; it treats them as unsupported until a regular "
        "correspondence system is demonstrated.",
        "John Peterson's 2021 areal-typological account treats the Satpura Range as a residual or "
        "accretion zone where Nihali, Korku, and Gondi survived successive regional expansions. "
        "He explicitly allows that pre-Munda languages were already present in such hill zones. "
        "That geography makes survival of an independent Nihali lineage historically plausible, "
        "but it is contextual fit rather than comparative proof and cannot date the language.",
        "A 2025 methodological survey of language isolates uses Nihali as an example of an isolate "
        "whose grammar has been extensively remodeled through sustained multilingualism. Its "
        "recommended historical toolkit—dialect comparison, internal reconstruction, philological "
        "source study, and explicit contact analysis—also explains the boundary of the present "
        "result: this lexical audit advances the last two, but cannot substitute for the first two.",
        "A Mohan-led [Endangered Language Documentation Programme project](https://cultureincrisis.org/"
        "projects/documentation-and-description-of-nihali-a-critically-endangered-language-isolate-"
        "of-india) targets a descriptive grammar, trilingual dictionary, and 20 hours of archived "
        "audio/video. Those materials are the right basis for testing shared morphological "
        "innovations, but they are not silently treated as part of this five-source lexical "
        "database. The distinction matters: this report's negative finding is absence of a "
        "demonstrated relationship in the audited evidence, not proof that no relationship can "
        "ever be found.",
        "A 2017 genome-wide study reports excess haplotype sharing and recent population ancestry "
        "between sampled Bhil and Nihali groups. That makes Zide and Shafer's lost-Bhil-language "
        "scenario geographically and demographically interesting, but genes do not classify "
        "languages. The result is compatible with regional population continuity or interaction; "
        "it neither proves that old Nihali was the Bhils' language nor supports a specific linguistic "
        "family. In particular, this report makes no 'Paleolithic' or ancestry-component claim from "
        "the lexical residue.",
        "",
        "Primary/contextual references: F. B. J. Kuiper, [*Nahali: A Comparative Study* "
        "(1962)](https://dwc.knaw.nl/DL/publications/PU00009788.pdf), especially pp. 48-51; "
        "Kuiper, [*The Sources of the Nahali Vocabulary* "
        "(1966)](https://sealang.net/sala/archives/pdf8/kuiper1966sources.pdf); "
        "Norman H. Zide, [\"On Nihali\" (1996)](https://www.mother-tongue-journal.org/"
        "wp-content/uploads/2025/08/2-Mother-Tongue-II-1996_text.pdf), pp. 93–100; K. S. "
        "Nagaraja, [*The Nihali Language* (2014)](https://hdl.handle.net/20.500.14705/8151); "
        "and Asha Mundlay, \"Nihali Lexicon\" (1996), in the same *Mother Tongue* volume, "
        "pp. 17–40; and Ilia Peiros, \"Nihali and Austroasiatic\" (1996), in the same volume, "
        "pp. 75–76; Shailendra Mohan, [\"Describing Endangered Languages: Experiences from Nihali "
        "Documentation Project\" (2016)](https://files.core.ac.uk/download/pdf/141880631.pdf), "
        "pp. 182–187; John Peterson, [\"The Spread of Munda in Prehistoric South Asia: The View "
        "from Areal Typology\" (2021)](https://www.isfas.uni-kiel.de/de/linguistik-und-phonetik/"
        "team/uploads/Peterson_Spread_of_Munda.pdf), pp. 109–130; Iker Salaberri et al., "
        "[\"State of the Art of Research on Language Isolates\" "
        "(2025)](https://doi.org/10.1075/tsl.135.intro), pp. 2–19; and Gyaneshwer Chaubey et al., "
        "[\"The Genome-Wide Analysis of the Bhils\" "
        "(2017)](https://www.isw.unibe.ch/e41142/e41180/e523709/e523717/2017b_ger.pdf), "
        "especially pp. 279–285.",
        "",
        "## Limits and next tests",
        "",
        "This is a hypothesis inventory, not a completed comparative reconstruction. The strongest "
        "next lexical step is independent verification of the unqualified source proxies against "
        f"primary donor lexica, prioritizing the {korku_route_assessments['form-only-match']:,} "
        f"Korku form-only matches and {korku_route_assessments['weak-or-unmatched']:,} "
        "weak/unmatched routes. Korku itself needs etymological expansion for the "
        f"{korku_strong_ultimate['unresolved-korku']:,} strongest route forms that "
        f"currently stop without an upstream parent. The {residue_root_count:,} core residue-root "
        "hypotheses then need "
        "dialectal comparison and correspondence discovery rather than another round of string "
        "matching. On the grammatical side, person marking, case allomorphy, and verb morphology "
        "require morpheme-by-morpheme reconstruction from the new documentation corpus. Neither "
        "the residue percentage nor the typological profile can currently distinguish an ancient "
        "isolate from an unrecognized deep relationship whose comparanda have been lost.",
        "",
    ]
    return "\n".join(lines)


def build(output_dir: Path, install: bool) -> dict[str, object]:
    forms = read_dicts(FORMS)
    edges = read_dicts(EDGES)
    languages = {row["ID"]: row for row in read_dicts(LANGUAGES)}
    by_id = {row["ID"]: row for row in forms}
    provisional_children = {
        row["Form_ID"] for row in overlay.read_assignments()
        if row.get("Notes", "").startswith(ASSIGNMENT_MARKER)
    }
    rank1 = {
        row["Child_ID"]: row for row in edges
        if row["Rank"] == "1" and row["Kind"] in {"reflex", "borrowed", "variant"}
        and row["Child_ID"] not in provisional_children
    }
    targets = [
        row for row in forms
        if row["Language_ID"] == "Ni" and row["Status"] != "entry"
        and not row["ID"].startswith("nihprov-")
    ]
    clusters, cluster_for_form = build_lexeme_clusters(targets)
    candidates, by_token, by_gloss = build_candidates(forms, rank1, languages)

    review_rows, reviews = load_candidate_reviews(MANUAL_REVIEW)
    tie_review_rows, tie_reviews = load_candidate_reviews(LOW_MARGIN_REVIEW)
    transparent_review_rows, transparent_reviews = load_candidate_reviews(
        TRANSPARENT_LOAN_REVIEW
    )
    source_parent_review_rows, source_parent_reviews = load_candidate_reviews(
        SOURCE_PARENT_REVIEW
    )
    review_sets = [
        set(reviews), set(tie_reviews), set(transparent_reviews), set(source_parent_reviews),
    ]
    if any(
        review_sets[i] & review_sets[j]
        for i in range(len(review_sets)) for j in range(i + 1, len(review_sets))
    ):
        raise RuntimeError("candidate review registers overlap")
    for review in [
        *review_rows, *tie_review_rows, *transparent_review_rows, *source_parent_review_rows,
    ]:
        if review["Decision"] == "accept" and review["Parent_ID"] not in by_id:
            raise RuntimeError(
                f"manual review parent missing from forms for {review['Lexeme_ID']}"
            )

    ranked_by_form: dict[str, list[tuple[float, float, float, Candidate]]] = {}
    machine_candidate_clusters: set[str] = set()
    tie_candidate_clusters: set[str] = set()
    review_basis_by_cluster: dict[str, list[tuple[float, float, float, Candidate]]] = {}
    for row in targets:
        if row["ID"] in rank1:
            continue
        cluster = clusters[cluster_for_form[row["ID"]]]
        pool = candidate_pool(row["Gloss"], by_token, by_gloss)
        ranked = rank_candidates(
            row, set(cluster["query_forms"]), set(cluster["strata"]),
            set(cluster["language_ids"]), candidates, pool,
        )
        ranked_by_form[row["ID"]] = ranked
        top = ranked[0] if ranked else None
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        base_candidate = not cluster["strata"] and top and (
            (top[0] >= 0.80 and top[1] >= 0.78 and top[2] >= 0.50)
            # Transparent modern cultural loans can carry a small adapted suffix that depresses
            # whole-form similarity.  Keep a narrow, review-only escape hatch for exact semantics.
            or (top[0] >= 0.85 and top[1] >= 0.75 and top[2] >= 0.95)
        )
        if base_candidate:
            lexeme_id = str(cluster["id"])
            if top[0] - second_score >= 0.045:
                machine_candidate_clusters.add(lexeme_id)
            else:
                tie_candidate_clusters.add(lexeme_id)
            previous = review_basis_by_cluster.get(lexeme_id)
            if previous is None or ranked[0][0] > previous[0][0]:
                review_basis_by_cluster[lexeme_id] = ranked
    if set(reviews) != machine_candidate_clusters:
        missing = sorted(machine_candidate_clusters - set(reviews))
        extra = sorted(set(reviews) - machine_candidate_clusters)
        raise RuntimeError(f"manual review coverage mismatch; missing={missing}, extra={extra}")
    if set(tie_reviews) != tie_candidate_clusters:
        missing = sorted(tie_candidate_clusters - set(tie_reviews))
        extra = sorted(set(tie_reviews) - tie_candidate_clusters)
        raise RuntimeError(
            f"low-margin manual review coverage mismatch; missing={missing}, extra={extra}"
        )
    # Computational review registers are decisions over a reproducibly generated candidate set.
    # A durable-looking form ID can still be stale or mistyped while denoting an unrelated word;
    # fail instead of silently substituting the current top candidate's score for that parent.
    for review_set in (reviews, tie_reviews):
        for lexeme_id, review in review_set.items():
            if review["Decision"] != "accept":
                continue
            current_parent_ids: set[str] = set()
            for item in review_basis_by_cluster.get(lexeme_id, []):
                candidate = item[3]
                current_parent_ids.add(candidate.parent_id)
                current_id = candidate.surface_id
                seen = set()
                while current_id in rank1 and current_id not in seen:
                    seen.add(current_id)
                    current_id = rank1[current_id]["Parent_ID"]
                    current_parent_ids.add(current_id)
            if review["Parent_ID"] not in current_parent_ids:
                raise RuntimeError(
                    f"accepted manual parent is not a current candidate for {lexeme_id}: "
                    f"{review['Parent_ID']}"
                )
    all_reviews = reviews | tie_reviews | transparent_reviews | source_parent_reviews

    audit: list[dict[str, str]] = []
    assignments: list[dict[str, str]] = []
    proxies: dict[str, dict[str, str]] = {}

    for row in sorted(targets, key=lambda item: item["ID"]):
        cluster = clusters[cluster_for_form[row["ID"]]]
        existing = rank1.get(row["ID"])
        etymology = row.get("Etymology", "")
        own_strata, _own_language_ids, _own_compared_forms = source_attribution(etymology)
        strata = list(cluster["strata"])
        language_ids = list(cluster["language_ids"])
        compared_forms = list(cluster["compared_forms"])
        source_label = "+".join(strata)
        own_source_label = "+".join(own_strata)
        manual_review = all_reviews.get(str(cluster["id"]))
        manual_trigger = (
            "transparent-cultural-loan" if str(cluster["id"]) in transparent_reviews else
            "source-attributed-parent" if str(cluster["id"]) in source_parent_reviews else
            "separated-candidate" if str(cluster["id"]) in reviews else
            "low-margin-family-tie" if str(cluster["id"]) in tie_reviews else ""
        )
        attribution_basis = (
            "direct source note" if own_strata else
            "cross-dictionary lexeme cluster" if strata else
            "none"
        )
        if existing:
            parent = by_id[effective_parent(row["ID"], rank1)]
            stratum = (
                source_label
                or (manual_review["Stratum"] if manual_review and manual_review["Decision"] == "accept" else "")
                or language_family(parent["Language_ID"], languages)
            )
            method, confidence = "existing-curated", "high"
            score, margin, alternatives = "1.000", "1.000", ""
            evidence = "Existing accepted Jambu edge retained."
            if manual_review and manual_review["Decision"] == "accept":
                evidence += (
                    " Its conservative lexeme-cluster peer was manually resolved externally: "
                    + manual_review["Rationale"]
                )
            parent_id, kind = existing["Parent_ID"], existing["Kind"]
            parent_display = by_id.get(parent_id, parent)
        else:
            ranked = ranked_by_form[row["ID"]]
            top = ranked[0] if ranked else None
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0
            review_ranked = (
                review_basis_by_cluster.get(str(cluster["id"]), ranked)
                if manual_review else ranked
            )
            review_top = review_ranked[0] if review_ranked else None
            review_second_score = review_ranked[1][0] if len(review_ranked) > 1 else 0.0
            accepted = False
            if top and strata:
                top_score, fsim, gsim, _candidate = top
                accepted = (
                    top_score >= 0.70 and fsim >= 0.68 and gsim >= 0.30
                    and top_score - second_score >= 0.025
                )
            if str(cluster["id"]) in source_parent_reviews and manual_review["Decision"] != "accept":
                accepted = False
            if manual_review and manual_review["Decision"] == "accept":
                parent_id = manual_review["Parent_ID"]
                parent_display = by_id[parent_id]
                stratum = manual_review["Stratum"]
                method = "manual-resolved"
                confidence = manual_review["Confidence"]
                reviewed_match = next(
                    (item for item in review_ranked if item[3].parent_id == parent_id), None
                )
                if reviewed_match:
                    reviewed_score = reviewed_match[0]
                elif manual_trigger in {"transparent-cultural-loan", "source-attributed-parent"}:
                    parent_forms = form_variants(
                        parent_display.get("Form") or parent_display.get("Original") or ""
                    )
                    manual_fsim = form_similarity(set(cluster["query_forms"]), parent_forms)
                    manual_gsim = gloss_similarity(row["Gloss"], parent_display.get("Gloss", ""))
                    reviewed_score = 0.64 * manual_fsim + 0.36 * manual_gsim
                else:
                    reviewed_score = review_top[0] if review_top else 0.0
                score = f"{reviewed_score:.3f}"
                margin = f"{reviewed_score - review_second_score:.3f}"
                evidence = (
                    ("Transparent cultural-loan review accepted: "
                     if manual_trigger == "transparent-cultural-loan" else
                     "Source-attributed parent correction accepted after manual review: "
                     if manual_trigger == "source-attributed-parent" else
                     "Source-free computational lead accepted after manual review: ")
                    + manual_review["Rationale"]
                )
                alternatives = concise_alternatives(review_ranked)
                kind = "borrowed"
            elif accepted and top:
                top_score, fsim, gsim, chosen = top
                parent_id = chosen.parent_id
                parent_display = by_id[parent_id]
                stratum = source_label or chosen.family
                method = "source-resolved" if own_strata else "cluster-resolved"
                confidence = (
                    "high" if own_strata and top_score >= 0.86 else
                    "medium" if top_score >= 0.82 else "low"
                )
                score = f"{top_score:.3f}"
                margin = f"{top_score - second_score:.3f}"
                evidence = (
                    f"Compatible Jambu candidate using {attribution_basis}, via "
                    f"{chosen.language_id} surface {chosen.form!r}; "
                    f"form={fsim:.3f}, gloss={gsim:.3f}."
                )
                alternatives = concise_alternatives(ranked)
                kind = "borrowed"
            else:
                if strata:
                    stratum = source_label
                    method = "source-proxy" if own_strata else "cluster-proxy"
                    confidence, kind = "low", "borrowed"
                    donor_language = language_ids[0] if language_ids else {
                        "Korku": "ko", "Munda": "PMu", "Indo-Aryan": "Indo-Aryan",
                        "Dravidian": "Drav", "English": "Eng",
                    }.get(strata[0], "Ni")
                    proxy_form = source_proxy_display_form(
                        str(cluster["id"]), compared_forms, row["Form"],
                    )
                    proxy_key = "source|" + str(cluster["id"]) + "|" + source_label
                    evidence = (
                        f"Donor attribution from {attribution_basis} retained as a proxy; no "
                        "compatible existing etymon cleared resolution thresholds."
                    )
                    if manual_trigger == "source-attributed-parent":
                        evidence += " Automatic parent rejected after manual review: " + manual_review["Rationale"]
                else:
                    stratum = "Nihali residue"
                    plausible_candidate = bool(
                        top
                        and (
                            (top[0] >= 0.80 and top[1] >= 0.78 and top[2] >= 0.50)
                            or (top[0] >= 0.85 and top[1] >= 0.75 and top[2] >= 0.95)
                        )
                        and top[0] - second_score >= 0.045
                    )
                    if manual_review:
                        method = "manual-" + (
                            "rejected" if manual_review["Decision"] == "reject" else "deferred"
                        )
                        confidence = (
                            "unresolved" if manual_review["Decision"] == "reject" else "low"
                        )
                    else:
                        method = "residue-proxy"
                        confidence = "unresolved"
                    kind = "reflex"
                    donor_language = "Ni"
                    proxy_form = row["Form"]
                    proxy_key = "residue|" + str(cluster["id"])
                    if manual_review:
                        decision_verb = (
                            "rejected" if manual_review["Decision"] == "reject" else "deferred"
                        )
                        evidence = (
                            f"Source-free computational lead {decision_verb} after manual review: "
                            f"{manual_review['Rationale']} The record remains in the Nihali residue "
                            "and is not linked to the suggested external parent."
                        )
                    elif plausible_candidate:
                        raise RuntimeError(
                            "unreviewed computational candidate unexpectedly reached output: "
                            f"{cluster['id']}"
                        )
                    else:
                        evidence = (
                            "No source attribution and no external Jambu candidate cleared "
                            "conservative review thresholds; retained as Nihali residue, not "
                            "asserted inherited."
                        )
                parent_id = proxy_id(proxy_key)
                display_top = review_top if manual_review else top
                display_second = review_second_score if manual_review else second_score
                display_ranked = review_ranked if manual_review else ranked
                score = f"{display_top[0]:.3f}" if display_top else "0.000"
                margin = (
                    f"{display_top[0] - display_second:.3f}" if display_top else "0.000"
                )
                alternatives = concise_alternatives(display_ranked)
                if parent_id not in proxies:
                    note = (
                        f"{ASSIGNMENT_MARKER}. {evidence} "
                        f"Source comparison: {etymology or 'none printed'}. "
                        "This grouping is provisional and is not a reconstruction."
                    )
                    proxies[parent_id] = {
                        "ID": parent_id, "Language_ID": donor_language, "Form": proxy_form,
                        "Gloss": row["Gloss"], "Source": REFERENCE, "Etymology": note,
                    }
                parent_display = {
                    "Form": proxies[parent_id]["Form"], "Language_ID": proxies[parent_id]["Language_ID"],
                    "Gloss": proxies[parent_id]["Gloss"],
                }
            assignments.append({
                "Form_ID": row["ID"], "Etymon_ID": parent_id, "Kind": kind, "Rank": "1",
                "Status": "accepted", "Source": REFERENCE,
                "Notes": f"{ASSIGNMENT_MARKER}; method={method}; confidence={confidence}",
            })
        parent_lang_id = parent_display["Language_ID"]
        parent_lang = languages.get(parent_lang_id, {}).get("Name", parent_lang_id)
        parent_clade = languages.get(parent_lang_id, {}).get("Clade", language_family(parent_lang_id, languages))
        audit.append({
            "Form_ID": row["ID"], "Lexeme_ID": str(cluster["id"]),
            "Lexeme_Size": str(cluster["size"]), "Form": row["Form"], "Gloss": row["Gloss"],
            "Lexical_Source": lexical_source(row["Source"]), "Form_Source": row["Source"],
            "Original_Etymology": etymology, "Method": method, "Stratum": stratum,
            "Confidence": confidence, "Score": score, "Margin": margin, "Parent_ID": parent_id,
            "Parent_Form": parent_display["Form"], "Parent_Language_ID": parent_lang_id,
            "Parent_Language": parent_lang, "Parent_Clade": parent_clade, "Kind": kind,
            "Source_Attribution": source_label, "Own_Source_Attribution": own_source_label,
            "Attribution_Basis": attribution_basis,
            "Manual_Trigger": manual_trigger,
            "Manual_Decision": manual_review["Decision"] if manual_review else "",
            "Manual_Rationale": manual_review["Rationale"] if manual_review else "",
            "Evidence": evidence,
            "Alternatives": alternatives,
        })

    if len(audit) != len(targets) or len({row["Form_ID"] for row in audit}) != len(targets):
        raise RuntimeError("Nihali audit is not one-to-one with target records")
    if {row["Form_ID"] for row in assignments} & set(rank1):
        raise RuntimeError("generated assignments would overwrite existing accepted edges")

    params_path = output_dir / PARAMS_NAME
    audit_path = output_dir / AUDIT_NAME
    cluster_audit_path = output_dir / CLUSTER_AUDIT_NAME
    core_audit_path = output_dir / CORE_AUDIT_NAME
    global_variant_sensitivity_audit_path = (
        output_dir / GLOBAL_VARIANT_SENSITIVITY_AUDIT_NAME
    )
    core_concept_profile_audit_path = output_dir / CORE_CONCEPT_PROFILE_AUDIT_NAME
    core_residue_root_audit_path = output_dir / CORE_RESIDUE_ROOT_AUDIT_NAME
    core_residue_root_inventory_path = output_dir / CORE_RESIDUE_ROOT_INVENTORY_NAME
    closed_class_audit_path = output_dir / CLOSED_CLASS_AUDIT_NAME
    replicated_residue_audit_path = output_dir / REPLICATED_RESIDUE_AUDIT_NAME
    resolved_contact_shape_audit_path = output_dir / RESOLVED_CONTACT_SHAPE_AUDIT_NAME
    dravidian_correspondence_audit_path = output_dir / DRAVIDIAN_CORRESPONDENCE_AUDIT_NAME
    source_variation_audit_path = output_dir / SOURCE_VARIATION_AUDIT_NAME
    source_proxy_quality_audit_path = output_dir / SOURCE_PROXY_QUALITY_AUDIT_NAME
    korku_route_audit_path = output_dir / KORKU_ROUTE_AUDIT_NAME
    indo_aryan_route_audit_path = output_dir / INDO_ARYAN_ROUTE_AUDIT_NAME
    questioned_source_proxy_review_path = output_dir / QUESTIONED_SOURCE_PROXY_REVIEW_NAME
    layer_replication_audit_path = output_dir / LAYER_REPLICATION_AUDIT_NAME
    source_profile_audit_path = output_dir / SOURCE_PROFILE_AUDIT_NAME
    cross_source_agreement_audit_path = output_dir / CROSS_SOURCE_AGREEMENT_AUDIT_NAME
    family_attribution_replication_audit_path = (
        output_dir / FAMILY_ATTRIBUTION_REPLICATION_AUDIT_NAME
    )
    layer_category_audit_path = output_dir / LAYER_CATEGORY_AUDIT_NAME
    layer_form_shape_audit_path = output_dir / LAYER_FORM_SHAPE_AUDIT_NAME
    contact_evidence_tier_audit_path = output_dir / CONTACT_EVIDENCE_TIER_AUDIT_NAME
    family_contact_evidence_audit_path = output_dir / FAMILY_CONTACT_EVIDENCE_AUDIT_NAME
    family_evidence_bracket_audit_path = output_dir / FAMILY_EVIDENCE_BRACKET_AUDIT_NAME
    summary_path = output_dir / SUMMARY_NAME
    report_path = output_dir / REPORT_NAME
    origin_evidence_matrix_path = output_dir / ORIGIN_EVIDENCE_MATRIX_NAME
    residue_threshold_sensitivity_path = output_dir / RESIDUE_THRESHOLD_SENSITIVITY_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    with params_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for proxy in sorted(proxies.values(), key=lambda item: item["ID"]):
            writer.writerow([
                proxy["ID"], proxy["Language_ID"], proxy["Form"], proxy["Gloss"], proxy["Source"]
            ])
    write_dicts(audit_path, AUDIT_FIELDS, audit)
    cluster_audit = build_cluster_audit(audit)
    write_dicts(cluster_audit_path, CLUSTER_AUDIT_FIELDS, cluster_audit)
    residue_contact_component_review = load_residue_contact_component_review(cluster_audit)
    global_variant_sensitivity_audit = build_global_variant_sensitivity_audit(cluster_audit)
    write_dicts(
        global_variant_sensitivity_audit_path,
        GLOBAL_VARIANT_SENSITIVITY_AUDIT_FIELDS,
        global_variant_sensitivity_audit,
    )
    source_profile_audit = build_source_profile_audit(audit)
    write_dicts(
        source_profile_audit_path, SOURCE_PROFILE_AUDIT_FIELDS, source_profile_audit,
    )
    cross_source_agreement_audit = build_cross_source_agreement_audit(audit)
    write_dicts(
        cross_source_agreement_audit_path, CROSS_SOURCE_AGREEMENT_AUDIT_FIELDS,
        cross_source_agreement_audit,
    )
    family_attribution_replication_audit = build_family_attribution_replication_audit(audit)
    write_dicts(
        family_attribution_replication_audit_path,
        FAMILY_ATTRIBUTION_REPLICATION_AUDIT_FIELDS,
        family_attribution_replication_audit,
    )
    layer_replication_audit = build_layer_replication_audit(cluster_audit)
    write_dicts(
        layer_replication_audit_path, LAYER_REPLICATION_AUDIT_FIELDS,
        layer_replication_audit,
    )
    layer_category_audit = build_layer_category_audit(audit, cluster_audit)
    write_dicts(
        layer_category_audit_path, LAYER_CATEGORY_AUDIT_FIELDS, layer_category_audit,
    )
    layer_form_shape_audit = build_layer_form_shape_audit(cluster_audit)
    write_dicts(
        layer_form_shape_audit_path, LAYER_FORM_SHAPE_AUDIT_FIELDS, layer_form_shape_audit,
    )
    core_audit = build_core_audit(audit, cluster_audit)
    write_dicts(core_audit_path, CORE_AUDIT_FIELDS, core_audit)
    closed_class_audit = build_closed_class_audit(audit, cluster_audit, core_audit)
    write_dicts(
        closed_class_audit_path, CLOSED_CLASS_AUDIT_FIELDS, closed_class_audit,
    )
    core_residue_root_audit = build_core_residue_root_audit(core_audit)
    write_dicts(
        core_residue_root_audit_path, CORE_RESIDUE_ROOT_AUDIT_FIELDS,
        core_residue_root_audit,
    )
    core_residue_root_inventory = build_core_residue_root_inventory(
        core_residue_root_audit, cluster_audit, audit
    )
    write_dicts(
        core_residue_root_inventory_path, CORE_RESIDUE_ROOT_INVENTORY_FIELDS,
        core_residue_root_inventory,
    )
    core_concept_profile_audit = build_core_concept_profile_audit(
        core_audit, core_residue_root_inventory
    )
    write_dicts(
        core_concept_profile_audit_path, CORE_CONCEPT_PROFILE_AUDIT_FIELDS,
        core_concept_profile_audit,
    )
    replicated_residue_audit = build_replicated_residue_audit(audit, core_audit)
    write_dicts(
        replicated_residue_audit_path, REPLICATED_RESIDUE_AUDIT_FIELDS,
        replicated_residue_audit,
    )
    resolved_contact_shape_audit = build_resolved_contact_shape_audit(audit, languages)
    write_dicts(
        resolved_contact_shape_audit_path, RESOLVED_CONTACT_SHAPE_AUDIT_FIELDS,
        resolved_contact_shape_audit,
    )
    dravidian_correspondence_audit = build_dravidian_correspondence_audit(
        resolved_contact_shape_audit
    )
    write_dicts(
        dravidian_correspondence_audit_path, DRAVIDIAN_CORRESPONDENCE_AUDIT_FIELDS,
        dravidian_correspondence_audit,
    )
    source_variation_audit = build_source_variation_audit(audit)
    write_dicts(
        source_variation_audit_path, SOURCE_VARIATION_AUDIT_FIELDS, source_variation_audit
    )
    source_proxy_quality_audit = build_source_proxy_quality_audit(audit, core_audit)
    write_dicts(
        source_proxy_quality_audit_path, SOURCE_PROXY_QUALITY_AUDIT_FIELDS,
        source_proxy_quality_audit,
    )
    korku_route_audit = build_korku_route_audit(source_proxy_quality_audit, languages)
    write_dicts(
        korku_route_audit_path, KORKU_ROUTE_AUDIT_FIELDS, korku_route_audit,
    )
    indo_aryan_route_audit = build_indo_aryan_route_audit(audit, languages)
    write_dicts(
        indo_aryan_route_audit_path, INDO_ARYAN_ROUTE_AUDIT_FIELDS,
        indo_aryan_route_audit,
    )
    questioned_review_ids = {
        lexeme_id for ids in QUESTIONED_PROXY_REVIEW_IDS.values() for lexeme_id in ids
    }
    questioned_source_proxy_review = [
        row for row in source_proxy_quality_audit if row["Lexeme_ID"] in questioned_review_ids
    ]
    write_dicts(
        questioned_source_proxy_review_path, SOURCE_PROXY_QUALITY_AUDIT_FIELDS,
        questioned_source_proxy_review,
    )
    contact_evidence_tier_audit = build_contact_evidence_tier_audit(
        audit, source_proxy_quality_audit, resolved_contact_shape_audit
    )
    write_dicts(
        contact_evidence_tier_audit_path, CONTACT_EVIDENCE_TIER_AUDIT_FIELDS,
        contact_evidence_tier_audit,
    )
    family_contact_evidence_audit = build_family_contact_evidence_audit(
        audit, contact_evidence_tier_audit, resolved_contact_shape_audit
    )
    write_dicts(
        family_contact_evidence_audit_path, FAMILY_CONTACT_EVIDENCE_AUDIT_FIELDS,
        family_contact_evidence_audit,
    )
    family_evidence_bracket_audit = build_family_evidence_bracket_audit(
        family_contact_evidence_audit, len(cluster_audit)
    )
    write_dicts(
        family_evidence_bracket_audit_path, FAMILY_EVIDENCE_BRACKET_AUDIT_FIELDS,
        family_evidence_bracket_audit,
    )
    residue_threshold_sensitivity = build_residue_threshold_sensitivity(audit, languages)
    write_dicts(
        residue_threshold_sensitivity_path, RESIDUE_THRESHOLD_SENSITIVITY_FIELDS,
        residue_threshold_sensitivity,
    )
    origin_evidence_matrix = build_origin_evidence_matrix(
        cluster_audit, core_concept_profile_audit, core_residue_root_inventory,
        family_evidence_bracket_audit, family_attribution_replication_audit,
        korku_route_audit, resolved_contact_shape_audit, dravidian_correspondence_audit,
        closed_class_audit, cross_source_agreement_audit, residue_threshold_sensitivity,
        global_variant_sensitivity_audit, residue_contact_component_review,
    )
    write_dicts(
        origin_evidence_matrix_path, ORIGIN_EVIDENCE_MATRIX_FIELDS,
        origin_evidence_matrix,
    )
    lexeme_strata = Counter(row["Stratum"] for row in cluster_audit)
    summary = {
        "records": len(audit), "existing_rank1": sum(row["Method"] == "existing-curated" for row in audit),
        "generated_assignments": len(assignments), "proxy_entries": len(proxies),
        "lexeme_clusters": len(clusters), "cluster_audit_rows": len(cluster_audit),
        "core_audit_rows": len(core_audit),
        "global_variant_sensitivity_rows": len(global_variant_sensitivity_audit),
        "global_variant_sensitivity_assessments": Counter(
            row["Assessment"] for row in global_variant_sensitivity_audit
        ),
        "residue_contact_component_review_rows": len(residue_contact_component_review),
        "residue_contact_component_assessments": Counter(
            row["Assessment"] for row in residue_contact_component_review
        ),
        "closed_class_rows": len(closed_class_audit),
        "closed_class_effective_strata": Counter(
            row["Effective_Stratum"] for row in closed_class_audit
        ),
        "closed_class_by_domain": {
            domain: Counter(
                row["Effective_Stratum"] for row in closed_class_audit
                if row["Domain"] == domain
            )
            for domain in sorted({row["Domain"] for row in closed_class_audit})
        },
        "layer_replication": {
            row["Layer"]: {
                "total_clusters": int(row["Total_Clusters"]),
                "multi_source_clusters": int(row["Multi_Source_Clusters"]),
                "early_source_attested": int(row["Konow_Or_Bhattacharya_Attested"]),
                "nagaraja_attested": int(row["Nagaraja_Attested"]),
                "all_five_sources": int(row["All_Five_Sources"]),
            }
            for row in layer_replication_audit
        },
        "source_normalized_profile": {
            row["Lexical_Source"]: {
                "lexeme_clusters": int(row["Lexeme_Clusters"]),
                "external_clusters": int(row["External_Clusters"]),
                "residue_clusters": int(row["Residue_Clusters"]),
                "korku_clusters": int(row["Korku_Clusters"]),
                "munda_clusters": int(row["Munda_Clusters"]),
                "indo_aryan_clusters": int(row["Indo_Aryan_Clusters"]),
                "dravidian_clusters": int(row["Dravidian_Clusters"]),
            }
            for row in source_profile_audit
        },
        "cross_source_agreement_rows": len(cross_source_agreement_audit),
        "cross_source_agreement_classes": Counter(
            row["Agreement_Class"] for row in cross_source_agreement_audit
        ),
        "family_attribution_replication": {
            row["Family"]: {
                "all_labelled": int(row["All_Labelled_Clusters"]),
                "multi_source": int(row["Multi_Source_Clusters"]),
                "no_direct_family_label": int(row["No_Direct_Family_Label"]),
                "one_direct_labelled_source": int(row["One_Direct_Labelled_Source"]),
                "two_plus_direct_labelled_sources": int(
                    row["Two_Plus_Direct_Labelled_Sources"]
                ),
            }
            for row in family_attribution_replication_audit
        },
        "layer_category_profile": {
            row["Layer"]: {
                "concept_linked_clusters": int(row["Concept_Linked_Clusters"]),
                "noun_clusters": int(row["Noun_Clusters"]),
                "verb_clusters": int(row["Verb_Clusters"]),
                "adjective_clusters": int(row["Adjective_Clusters"]),
                "numeral_clusters": int(row["Numeral_Clusters"]),
                "other_clusters": int(row["Other_Clusters"]),
            }
            for row in layer_category_audit
        },
        "layer_form_shape_profile": {
            row["Layer"]: {
                "total_clusters": int(row["Total_Clusters"]),
                "mean_folded_length": float(row["Mean_Folded_Length"]),
                "median_folded_length": float(row["Median_Folded_Length"]),
                "final_vowel_count": int(row["Final_Vowel_Count"]),
                "compound_count": int(row["Multiword_Or_Compound_Count"]),
                "retroflex_count": int(row["Retroflex_Count"]),
                "aspiration_count": int(row["Aspiration_Count"]),
                "nasalization_count": int(row["Nasalization_Count"]),
            }
            for row in layer_form_shape_audit
        },
        "core_mapping_exclusions": len(read_dicts(CORE_EXCLUSIONS)),
        "core_concept_profile_rows": len(core_concept_profile_audit),
        "core_concept_profile_classes": Counter(
            row["Profile_Class"] for row in core_concept_profile_audit
        ),
        "core_concept_family_involvement": {
            family: sum(
                family in row["Contact_Families"].split("+")
                for row in core_concept_profile_audit
            )
            for family in ("Korku", "Munda", "Indo-Aryan", "Dravidian")
        },
        "core_concepts_covered": len({
            concept for row in core_audit for concept in row["Concepts"].split("; ")
        }),
        "core_strata": Counter(row["Stratum"] for row in core_audit),
        "core_variant_sensitivity_rows": sum(
            bool(row["Sensitivity_Stratum"]) for row in core_audit
        ),
        "core_sensitivity_strata": Counter(
            row["Sensitivity_Stratum"] or row["Stratum"] for row in core_audit
        ),
        "core_residue_root_concepts": len(core_residue_root_audit),
        "core_residue_root_hypotheses": sum(
            int(row["Root_Group_Count"]) for row in core_residue_root_audit
        ),
        "core_residue_root_replication": Counter(
            row["Replication_Grade"] for row in core_residue_root_inventory
        ),
        "core_residue_root_early_attested": sum(
            row["Early_Source_Attested"] == "yes" for row in core_residue_root_inventory
        ),
        "core_residue_multi_cluster_reviews": sum(
            int(row["Cluster_Count"]) > 1 for row in core_residue_root_audit
        ),
        "replicated_residue_rows": len(replicated_residue_audit),
        "replicated_residue_grades": Counter(
            row["Replication_Grade"] for row in replicated_residue_audit
        ),
        "replicated_effective_core_residue": sum(
            row["Core_Effective_Residue"] == "yes" for row in replicated_residue_audit
        ),
        "resolved_contact_shape_rows": len(resolved_contact_shape_audit),
        "resolved_contact_parent_families": Counter(
            row["Parent_Family"] for row in resolved_contact_shape_audit
        ),
        "resolved_contact_parent_languages": Counter(
            row["Parent_Language"] for row in resolved_contact_shape_audit
        ),
        "resolved_contact_match_shapes": {
            family: Counter(
                row["Match_Shape"] for row in resolved_contact_shape_audit
                if row["Parent_Family"] == family
            )
            for family in sorted({row["Parent_Family"] for row in resolved_contact_shape_audit})
        },
        "resolved_contact_surface_match_shapes": {
            family: Counter(
                row["Surface_Match_Shape"] for row in resolved_contact_shape_audit
                if row["Parent_Family"] == family and row["Surface_Match_Shape"]
            )
            for family in sorted({row["Parent_Family"] for row in resolved_contact_shape_audit})
        },
        "resolved_parent_route_family_involvement": {
            parent_family: Counter({
                route_family: sum(
                    row["Parent_Family"] == parent_family
                    and route_family in row["Source_Stratum"].split("+")
                    for row in resolved_contact_shape_audit
                )
                for route_family in ("Korku", "Munda", "Indo-Aryan", "Dravidian")
            })
            for parent_family in sorted({
                row["Parent_Family"] for row in resolved_contact_shape_audit
            })
        },
        "munda_correspondence_review": Counter(
            row["Review_Assessment"] for row in resolved_contact_shape_audit
            if row["Parent_Family"] == "Munda"
        ),
        "munda_correspondence_series": Counter(
            row["Correspondence_Series"] for row in resolved_contact_shape_audit
            if row["Parent_Family"] == "Munda"
        ),
        "munda_resolved_parent_roots": len({
            row["Parent_ID"] for row in resolved_contact_shape_audit
            if row["Parent_Family"] == "Munda"
        }),
        "dravidian_correspondence_roots": len(dravidian_correspondence_audit),
        "dravidian_correspondence_assessments": Counter(
            row["Assessment"] for row in dravidian_correspondence_audit
        ),
        "dravidian_correspondence_series_types": Counter(
            row["Series_Type"] for row in dravidian_correspondence_audit
        ),
        "source_variation_rows": len(source_variation_audit),
        "source_variation_relationships": Counter(
            row["Relationship"] for row in source_variation_audit
        ),
        "source_proxy_quality_rows": len(source_proxy_quality_audit),
        "source_proxy_quality": Counter(
            row["Evidence_Quality"] for row in source_proxy_quality_audit
        ),
        "source_proxy_uncertainty": Counter(
            row["Uncertainty"] for row in source_proxy_quality_audit
        ),
        "source_proxy_directionality": Counter(
            row["Directionality"] for row in source_proxy_quality_audit
        ),
        "source_proxy_comparison_shapes": Counter(
            row["Comparison_Shape"] for row in source_proxy_quality_audit
        ),
        "source_proxy_review_priority": Counter(
            row["Review_Priority"] for row in source_proxy_quality_audit
        ),
        "korku_route_rows": len(korku_route_audit),
        "korku_route_assessments": Counter(
            row["Route_Assessment"] for row in korku_route_audit
        ),
        "core_korku_route_assessments": Counter(
            row["Route_Assessment"] for row in korku_route_audit
            if row["Core_Vocabulary"] == "yes"
        ),
        "korku_strong_route_ultimate_families": Counter(
            row["Ultimate_Parent_Family"] or "unresolved-korku"
            for row in korku_route_audit
            if row["Route_Assessment"] == "strong-route-match"
        ),
        "indo_aryan_route_rows": len(indo_aryan_route_audit),
        "indo_aryan_period_evidence_classes": Counter(
            row["Period_Evidence_Class"] for row in indo_aryan_route_audit
        ),
        "indo_aryan_korku_route_clusters": sum(
            row["Korku_Route_Mentioned"] == "yes" for row in indo_aryan_route_audit
        ),
        "core_source_proxy_review": Counter(
            row["Review_Assessment"] for row in source_proxy_quality_audit
            if row["Review_Priority"] == "critical"
        ),
        "diagnostic_source_proxy_review": Counter(
            row["Review_Assessment"] for row in source_proxy_quality_audit
            if row["Review_Priority"] == "high"
        ),
        "questioned_source_proxy_review": Counter(
            row["Review_Assessment"] for row in questioned_source_proxy_review
        ),
        "contact_evidence_tier_rows": len(contact_evidence_tier_audit),
        "contact_evidence_tiers": Counter(
            row["Evidence_Tier"] for row in contact_evidence_tier_audit
        ),
        "contact_family_by_tier": {
            family: Counter(
                row["Evidence_Tier"] for row in family_contact_evidence_audit
                if row["Family"] == family
            )
            for family in ("Korku", "Munda", "Indo-Aryan", "Dravidian", "English")
        },
        "family_contact_evidence_rows": len(family_contact_evidence_audit),
        "family_evidence_brackets": {
            row["Family"]: {
                "all_labelled": int(row["All_Labelled_Clusters"]),
                "high_specificity_floor": int(row["High_Specificity_Floor"]),
                "supported_envelope": int(row["Supported_Envelope"]),
                "weak_or_unresolved_excluded": int(row["Weak_Or_Unresolved_Excluded"]),
            }
            for row in family_evidence_bracket_audit
        },
        "origin_evidence_matrix_rows": len(origin_evidence_matrix),
        "residue_threshold_sensitivity": {
            row["Threshold_Label"]: {
                "flagged": int(row["Flagged_Residue_Clusters"]),
                "indo_aryan": int(row["Indo_Aryan_Candidates"]),
                "dravidian": int(row["Dravidian_Candidates"]),
                "munda": int(row["Munda_Candidates"]),
                "unreviewed": int(row["Unreviewed"]),
            }
            for row in residue_threshold_sensitivity
        },
        "disjoint_source_review": Counter(
            row["Review_Assessment"] for row in source_variation_audit
            if row["Review_Assessment"]
        ),
        "lexeme_strata": lexeme_strata,
        "manual_review": Counter(
            row["Decision"]
            for row in [
                *review_rows, *tie_review_rows, *transparent_review_rows,
                *source_parent_review_rows,
            ]
        ),
        "manual_review_triggers": {
            "separated-candidate": len(review_rows),
            "low-margin-family-tie": len(tie_review_rows),
            "transparent-cultural-loan": len(transparent_review_rows),
            "source-attributed-parent": len(source_parent_review_rows),
        },
        "methods": Counter(row["Method"] for row in audit),
        "strata": Counter(row["Stratum"] for row in audit),
        "confidence": Counter(row["Confidence"] for row in audit),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(
        render_report(
            audit, proxies, core_audit, global_variant_sensitivity_audit,
            residue_contact_component_review,
            source_variation_audit, source_proxy_quality_audit,
            korku_route_audit,
            closed_class_audit, core_residue_root_audit, core_residue_root_inventory,
            core_concept_profile_audit,
            replicated_residue_audit,
            resolved_contact_shape_audit, dravidian_correspondence_audit,
            contact_evidence_tier_audit, family_contact_evidence_audit,
            source_profile_audit, cross_source_agreement_audit,
            family_attribution_replication_audit,
            layer_replication_audit, layer_category_audit, layer_form_shape_audit,
            indo_aryan_route_audit,
            origin_evidence_matrix,
            residue_threshold_sensitivity,
        ),
        encoding="utf-8",
    )

    if install:
        # The analysis bundle keeps a reviewable copy; the build consumes params only from this
        # canonical directory.  Keeping the copy step explicit prevents a successful analysis run
        # from leaving assignments whose proxy parents are absent from the CLDF input graph.
        canonical_params = ROOT / "data/other/params" / PARAMS_NAME
        canonical_params.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(params_path, canonical_params)

        # Replace only this script's earlier rows.  The per-source sidecars remain the build
        # inputs, so no pipeline special-case or generated side table is required; new rows are
        # filed under the Nihali source that owns each form.
        existing_assignments = [
            row for row in overlay.read_assignments()
            if not row.get("Notes", "").startswith(ASSIGNMENT_MARKER)
            and not row.get("Etymon_ID", "").startswith("nihprov-")
        ]
        combined = existing_assignments + assignments
        combined.sort(key=lambda row: (row["Form_ID"], int(row["Rank"] or 1), row["Etymon_ID"]))
        overlay.write_assignments(combined, overlay.SidecarResolver())

        existing_etyma = []
        with ETYMOLOGIES.open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                if row and not row[0].startswith("nihprov-"):
                    existing_etyma.append(row)
        with ETYMOLOGIES.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerows(existing_etyma)
            for proxy in sorted(proxies.values(), key=lambda item: item["ID"]):
                writer.writerow([proxy["ID"], proxy["Etymology"]])

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.install:
        output_dir = args.output_dir or ROOT / "data/other/analysis/nihali-provisional"
    else:
        output_dir = args.output_dir or ROOT / "tmp/nihali-provisional"
    summary = build(output_dir, args.install)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
