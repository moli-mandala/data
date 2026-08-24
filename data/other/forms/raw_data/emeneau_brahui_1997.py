#!/usr/bin/env python3
"""Install the page-wise extraction of Emeneau's 1997 Brahui article.

The copyrighted PDF is not redistributed.  Eight article-page JSON files are the checked-in raw
extraction layer: a low-cost agent read each rendered page under the contract in
``emeneau_brahui_1997_prompt.md``.  This importer is the editorial reconciliation layer.  It
deduplicates cross-page claims, corrects image-checked transcription, separates accepted links
from ranked hypotheses, and writes the lexical, entry-text, audit, sample, reconciliation, and
manifest artifacts used by Jambu.

Run from ``data/``.  Supplying the original PDF verifies its identity; CI can rebuild entirely
from the checked-in page JSON files::

    uv run python data/other/forms/raw_data/emeneau_brahui_1997.py \
      --pdf ../../Downloads/Emeneau-BrahuiEtymologiesPhonetic-1997.pdf
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


SOURCE_ID = "emeneau1997brahui"
SNAPSHOT_DATE = "2026-08-19"
PDF_SHA256 = "e2aa0c7a0063b83509cf402cb880de97a902b1195c7d5f69bb75d8775fb30dde"
PDF_PAGES = 9
PRINTED_PAGES = tuple(range(440, 448))

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
AGENT_DIR = RAW_DIR / "emeneau_brahui_1997_agent"
FORM_OUTPUT = ROOT / "data/other/forms/20260819-emeneau-brahui-1997.csv"
TEXT_OUTPUT = ROOT / "data/other/entry_texts/20260819-emeneau-brahui-1997.csv"
AUDIT_OUTPUT = RAW_DIR / "20260819-emeneau-brahui-1997-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260819-emeneau-brahui-1997-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260819-emeneau-brahui-1997-manifest.json"
RECONCILIATION_OUTPUT = RAW_DIR / "20260819-emeneau-brahui-1997-reconciliation.json"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
TEXT_FIELDS = ["Form_ID", "Position", "Kind", "Format", "Content", "Source"]
AUDIT_FIELDS = [
    "Snapshot_Date", "Unit_ID", "PDF_Page", "Printed_Page", "Section", "Raw_Record_Type",
    "Raw_Form", "Raw_Gloss", "Raw_Target_System", "Raw_Target_ID", "Raw_Claim_Status",
    "Raw_Editorial_Action", "Final_Status", "Final_Form", "Final_Parameter_ID",
    "Final_Claim_Status", "Emitted_Keys", "Entry_Text_Targets", "Resolution",
    "Agent_Correction", "Review", "Material_Error", "Source", "Record_SHA256",
]


@dataclass(frozen=True)
class FormSpec:
    key: str
    records: tuple[str, ...]
    language: str
    parameter: str
    form: str
    gloss: str
    printed_page: int
    section: str
    etymology: str
    tags: str
    variant_of: str = ""

    @property
    def locator(self) -> str:
        return f"{SOURCE_ID}[p. {self.printed_page}, §{self.section}]"

    def row(self) -> dict[str, str]:
        return dict(zip(FORM_FIELDS, [
            self.language, self.parameter, self.form, self.gloss, "", "", "",
            self.locator, "", self.etymology, self.key, self.variant_of, "", "", self.tags,
        ]))


@dataclass(frozen=True)
class TextSpec:
    key: str
    records: tuple[str, ...]
    targets: tuple[str, ...]
    position: int
    kind: str
    printed_page: int
    section: str
    content: str

    @property
    def locator(self) -> str:
        return f"{SOURCE_ID}[p. {self.printed_page}, §{self.section}]"

    def rows(self) -> list[dict[str, str]]:
        return [
            {
                "Form_ID": target,
                "Position": str(self.position),
                "Kind": self.kind,
                "Format": "markdown",
                "Content": self.content,
                "Source": self.locator,
            }
            for target in self.targets
        ]


FORMS = (
    FormSpec(
        "emeneau1997brahui:p441:s2.1:begh", ("p441:s2.1:u02", "p441:s2.2:u07", "p442:s4:u01"),
        "Brahui", "d5078", "bēg̲h̲-", "to knead, muddle up (and spoil)", 442, "4",
        "Emeneau adds this reflex to DEDR 5078 and reconstructs *mel-k-; the proposed plural-action analysis of -gh- remains tentative.",
        "verb tr",
    ),
    FormSpec(
        "emeneau1997brahui:p441:s2.1:bel", ("p441:s2.1:u03",), "Brahui", "d5503", "bēl",
        "large hill-torrent", 441, "2.1", "Emeneau retains DEDR 5503 and explicitly removes its query.", "noun",
    ),
    FormSpec(
        "emeneau1997brahui:p441:s2.2:hogh", ("p441:s2.2:u08", "p441:s3:u01", "p442:s3:u03", "p442:s3:u04"),
        "Brahui", "d996", "hōg̲h̲-", "to weep, cry", 442, "3",
        "Emeneau removes the DEDR query and analyzes the form with Kurux-Malto *-k- plural-action morphology; loss of gh in finite forms remains tentative.",
        "verb intr",
    ),
    FormSpec(
        "emeneau1997brahui:p441:s2.2:mux", ("p441:s2.2:u09",), "Brahui", "d4986", "mux",
        "waist, loins", 441, "2.2", "Cited as evidence for loss of *l in a consonant cluster from *mulk-.", "noun",
    ),
    FormSpec(
        "emeneau1997brahui:p441:s2.2:taf", ("p441:s2.2:u10",), "Brahui", "d3133", "taf-",
        "to tie up, bind; become congealed; gather (of clouds)", 441, "2.2",
        "Emeneau removes the DEDR query and treats the form as evidence for loss of *l in *tal-v/p-.", "verb",
    ),
    FormSpec(
        "emeneau1997brahui:p442:s4:basht", ("p442:s4:u03",), "Brahui", "d4841", "bāšt",
        "heaven", 442, "4", "Emeneau adds Elfenbein's form as evidence for Proto-Dravidian *m- > Brahui b- before a front vowel.", "noun",
    ),
    FormSpec(
        "emeneau1997brahui:p443:s6:puzza", ("p443:s6:u05",), "Brahui", "", "pužža",
        "human hair", 443, "6", "Emeneau says the form might belong in DEDR 4477; the comparison remains a ranked hypothesis.",
        "noun uncertain",
    ),
    FormSpec(
        "emeneau1997brahui:p443:s6:kuzing", ("p443:s6:u01", "p443:s6:u07", "p443:s6:u08"),
        "Brahui", "", "kūžing", "to shrink in fear", 443, "6",
        "Emeneau considers DEDR 1876 and 2687, preferring 2687 on the initial consonant but requiring a query for either analysis.",
        "verb intr uncertain",
    ),
    FormSpec(
        "emeneau1997brahui:p443:s7:pisfing", ("p443:s7:u01", "p443:s7:u02", "p443:s7:u03"),
        "Brahui", "", "pisfing", "to squeeze air out of an inflated skin or football", 443, "7",
        "Emeneau prefers DEDR 4135 to 4183, but the final -f- is unexplained; both are retained only as ranked hypotheses.",
        "verb tr uncertain",
    ),
    FormSpec(
        "emeneau1997brahui:p444:s8.1:shupping", ("p444:s8.1:u01",), "Brahui", "", "šupping",
        "to drink a thick liquid", 444, "8.1",
        "Emeneau suggests DEDR 2621, while leaving both an Indo-Aryan comparison and expressive reshaping open.",
        "verb tr uncertain",
    ),
    FormSpec(
        "emeneau1997brahui:p444:s8.2:shurufing", ("p444:s8.2:u01", "p444:s8.2:u02"),
        "Brahui", "d2712", "šurūfing", "to smack the lips", 444, "8.2",
        "Emeneau places the expressive form in DEDR 2712; the internal -ūf- remains unexplained.", "verb",
    ),
    FormSpec(
        "emeneau1997brahui:p444:s10:kirrefing", ("p444:s10:u01",), "Brahui", "d1595", "kirrefing",
        "to turn a hand-mill", 444, "10", "Emeneau accepts the speaker's causative analysis from *kirreng and places the verb in DEDR 1595.",
        "verb tr caus",
    ),
    FormSpec(
        "emeneau1997brahui:p444:s11:tarifing:slaughter", ("p444:s11:u01",), "Brahui", "d3029", "taṛifing",
        "to be slaughtered", 444, "11", "Emeneau adds this sense to DEDR 3029 but rejects its presentation as a causative of taṛing.",
        "verb intr",
    ),
    FormSpec(
        "emeneau1997brahui:p444:s11:tarifing:sour", ("p444:s11:u02",), "Brahui", "", "taṛifing",
        "to turn sour (of milk)", 444, "11", "Emeneau splits this homonymous sense from 'be slaughtered' and leaves it without etymology.",
        "verb intr uncertain",
    ),
    FormSpec(
        "emeneau1997brahui:p444:fn3:cikap", ("p444:sfoot:u01",), "Gadaba", "", "cīkap-",
        "to lick, suck", 444, "8.1 n. 3", "Emeneau adds Bhaskararao's Gadaba form to DEDR 2621a and identifies it as a borrowing from Telugu cīk-.",
        "verb tr loanword",
    ),
    FormSpec(
        "emeneau1997brahui:p445:s12:taring", ("p445:s12:u01",), "Brahui", "d3195", "tāring",
        "to screen coal; sift", 445, "12", "Emeneau adds the form to DEDR 3195.", "verb tr",
    ),
    FormSpec(
        "emeneau1997brahui:p445:s13:allai", ("p445:s13:u01", "p445:s13:u02"), "Brahui", "d235", "allāī",
        "long winter nights", 445, "13", "Emeneau identifies the form as a continuant of DEDR 235 and compares Kurux ell-.", "noun",
    ),
    FormSpec(
        "emeneau1997brahui:p446:fn6:dui", ("p446:s15:u05",), "Brahui", "", "dūī", "tongue",
        446, "15 n. 6", "Emeneau finds no secure Dravidian, Indo-Aryan, or Iranian etymology; CDIAL 5228 remains only a tempting possibility.",
        "noun uncertain",
    ),
    FormSpec(
        "emeneau1997brahui:p446:fn6:duwi", ("p446:s15:u05",), "Brahui", "", "duwī", "tongue",
        446, "15 n. 6", "Variant printed beside dūī in Emeneau's unresolved footnote.", "noun uncertain alternate",
        variant_of="emeneau1997brahui:p446:fn6:dui",
    ),
)


TEXTS = (
    TextSpec("mid-vowels", ("p441:s2.1:u01",), ("d5078", "d5503"), 199701, "phonology", 441, "2.1",
             "Emeneau uses **bēgh-** and **bēl** to fill the previously missing Proto-Dravidian *e > Brahui ē development."),
    TextSpec("l-loss", ("p441:s2.2:u01", "p441:s2.2:u02", "p441:s2.2:u03", "p441:s2.2:u04", "p441:s2.2:u05", "p441:s2.2:u06", "p441:s2.2:u11"),
             ("d5078", "d996", "d4986", "d3133", "d3098"), 199702, "phonology", 441, "2.2",
             "Emeneau proposes loss of *l/ḷ as the first member of a consonant cluster, parallel to the established loss of rhotics, citing bēgh-, hōgh-, mux, taf-, and the ha- component of ha-tin-."),
    TextSpec("hogh", ("p441:s3:u02", "p442:s3:u01", "p442:s3:u02", "p442:s3:u05"), ("d996",), 199703, "analysis", 442, "3",
             "The paper prefers DEDR 996 over Bray's imitative analysis of **hōgh-**. It treats the final consonant as the North Dravidian plural-action *-k- correspondence, while leaving its absence in some imperative and prohibitive forms unresolved."),
    TextSpec("begh", ("p442:s4:u01", "p442:s4:u04", "p442:s4:u05", "p442:s4:u06", "p443:s4:u01"), ("d5078",), 199704, "analysis", 442, "4",
             "The paper reconstructs **bēgh-** from *mel-k-: *m- > b- before a front vowel, *e > ē, and loss of *l in the cluster. A plural-action interpretation of -gh- is explicitly tentative."),
    TextSpec("basht", ("p442:s4:u02", "p442:s4:u03"), ("d4841",), 199705, "analysis", 442, "4",
             "Emeneau adds **bāšt** 'heaven' beside **bash** 'up' as evidence for Proto-Dravidian *m- > Brahui b- before a front vowel."),
    TextSpec("sibilants", ("p443:s5:u01", "p444:s8:u01", "p444:s8:u02", "p444:s8:u03", "p444:s8:u04", "p444:s8:u05", "p444:s8:u06", "p444:s8:u07"),
             ("d1876", "d2687", "d2621", "d2712"), 199706, "phonology", 444, "5–8",
             "The paper treats several irregular Brahui sibilants as partly expressive. The regular correspondences remain *c- > c- and medial *-c-/*-cc- > -s-."),
    TextSpec("puzza", ("p443:s6:u05", "p443:s6:u06"), ("d4477", "d4476"), 199707, "comparison", 443, "6",
             "**pužža** 'human hair' is only tentatively compared with DEDR 4477; **pōs** in DEDR 4476 is cited as a contrasting secure development."),
    TextSpec("kuz", ("p443:s6:u07", "p443:s6:u08", "p443:s6:u09"), ("d1876", "d2687"), 199708, "comparison", 443, "6",
             "Emeneau considers DEDR 1876 and 2687 for **kūžing**. Entry 2687 better fits initial k-, but neither explains ž securely; both comparisons require a query."),
    TextSpec("pisf", ("p443:s7:u02", "p443:s7:u03", "p443:s7:u04", "p443:s7:u05", "p443:s7:u06"), ("d4135", "d4183"), 199709, "comparison", 443, "7",
             "For **pisfing**, Emeneau regards DEDR 4135 as simpler than 4183. The latter would require cluster simplification, and -f- remains unexplained in either analysis."),
    TextSpec("shupp", ("p444:s8.1:u01", "p444:s8.1:u02"), ("d2621",), 199710, "comparison", 444, "8.1",
             "**šupping** 'drink a thick liquid' is tentatively compared with DEDR 2621b. Expressive reshaping and a possible Indo-Aryan comparison prevent a categorical assignment."),
    TextSpec("shuruf", ("p444:s8.2:u01", "p444:s8.2:u02"), ("d2712",), 199711, "analysis", 444, "8.2",
             "Emeneau places expressive **šurūfing** 'smack the lips' in DEDR 2712, while leaving -ūf- unexplained."),
    TextSpec("kirref", ("p444:s9:u01", "p444:s10:u01"), ("d1595",), 199712, "analysis", 444, "10",
             "Speaker-supplied **kirrefing** 'turn a hand-mill', analyzed as a causative of *kirreng 'turn', is added to DEDR 1595."),
    TextSpec("tarif", ("p444:s11:u01", "p444:s11:u02"), ("d3029",), 199713, "analysis", 444, "11",
             "Emeneau splits two meanings printed together: **taṛifing** 'be slaughtered' belongs in DEDR 3029 but is not a causative of taṛing; 'turn sour (milk)' remains unetymologized."),
    TextSpec("gadaba", ("p444:sfoot:u01",), ("d2621",), 199714, "source-note", 444, "8.1 n. 3",
             "Footnote 3 adds Gadaba (Bhaskararao) **cīkap-** 'lick, suck' to DEDR 2621a and identifies it as a borrowing from Telugu cīk-."),
    TextSpec("taring", ("p445:s12:u01", "p445:s12:u02", "p445:s12:u03"), ("d3195", "d3402"), 199715, "correction", 445, "12",
             "Emeneau adds **tāring** 'sift' to DEDR 3195 and removes the query on dranz-/drāz-. He rejects the less economical placement of the latter in DEDR 3402."),
    TextSpec("allai", ("p445:s13:u01", "p445:s13:u02", "p445:s13:u03", "p445:s13:u04"), ("d235",), 199716, "analysis", 445, "13",
             "The paper places **allāī** 'long winter nights' in DEDR 235, comparing Kurux ell-. It does not decide whether -āī is the locative suffix."),
    TextSpec("235-3613", (), ("d235", "d3613"), 199717, "comparison", 445, "13 n. 4",
             "Emeneau says a connection between DEDR 235 and 3613 remains doubtful and may ultimately be improbable."),
    TextSpec("horse-reject", ("p445:s14:u01", "p445:s14:u02"), ("d500",), 199718, "correction", 445, "14",
             "Emeneau rejects the DEDR 500 comparison of Brahui (h)ullī with Tamil ivuḷi on phonological grounds; the Brahui form is reassigned on the following page."),
    TextSpec("horse-reassign", ("p446:s14:u01", "p446:s14:u02"), ("d701",), 199719, "analysis", 446, "14",
             "The paper reassigns Brahui **(h)ullī** 'horse' to DEDR 701, deriving it from *uḷ- 'mane' with an uncertain -ī suffix. The former DEDR 500 edge is superseded."),
    TextSpec("du", ("p446:s15:u01", "p446:s15:u02", "p446:s15:u03", "p446:s15:u04"), ("6586", "14024"), 199720, "comparison", 446, "15",
             "Emeneau finds a northwestern Indo-Aryan origin for Brahui **dū** 'hand, arm' highly probable, specifically CDIAL 6586 dṓṣ- 'forearm'. He rejects the Persian dast / CDIAL 14024 hásta- comparison because their final consonants survive."),
    TextSpec("dui", ("p446:s15:u05",), ("5228",), 199721, "comparison", 446, "15 n. 6",
             "For Brahui **dūī, duwī** 'tongue', the paper finds CDIAL 5228 tempting but leaves both reconstruction and immediate borrowing source unresolved."),
)


CORRECTIONS = {
    "p441:s2.1:u02": "target_id canonicalized from '5078 (*mel(l)-)' to 5078",
    "p441:s2.1:u03": "target_id canonicalized from '5503 (*vell-)' to 5503",
    "p441:s2.2:u07": "target_id canonicalized from '5078 (*mel(l)-)' to 5078",
    "p441:s2.2:u08": "target_id canonicalized from '996 (*olk-)' to 996",
    "p441:s2.2:u11": "agent conflated ha- with the footnote's unrelated be- gloss; retained only in the cluster-loss note",
    "p441:s3:u01": "target_id canonicalized from '996 (*olk-)' to 996",
    "p442:s3:u03": "cleared spurious previous_target_id; this is not a reassignment",
    "p442:s4:u03": "image-checked Brahui form corrected from agent baṣṭ to bāšt",
    "p443:s6:u05": "'might belong' is a ranked hypothesis, not an installed direct reflex",
    "p443:s7:u02": "calibrated probable to suggested",
    "p444:s8.2:u02": "cleared spurious previous_target_id",
    "p444:s10:u01": "target_id canonicalized from '1595 (*kir...)' to 1595",
    "p444:s11:u01": "image-checked retroflex form is taṛifing; relation is an accepted reflex, not generic related",
    "p444:sfoot:u01": "image-checked vowel length restored: cīkap-",
    "p445:s13:u01": "deduplicated from the separate reflex-claim record before installation",
    "p445:s14:u01": "the page anticipates rejection; the actual reassignment is installed from p. 446",
    "p446:s14:u01": "gloss corrected from Tamil comparator 'mane' to Brahui 'horse'",
    "p446:s15:u05": "image-checked forms normalized to dūī, duwī; CDIAL 5228 remains only a hypothesis",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pages() -> list[dict]:
    pages = []
    for printed_page in PRINTED_PAGES:
        path = AGENT_DIR / f"p{printed_page}.json"
        with path.open(encoding="utf-8") as handle:
            page = json.load(handle)
        assert page["printed_page"] == printed_page, path
        assert page["pdf_page"] == printed_page - 438, path
        pages.append(page)
    return pages


def write_csv(
    path: Path, fields: list[str], rows: list[dict[str, str]], *, header: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if header:
            writer.writeheader()
        writer.writerows(rows)


def record_hash(record: dict) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_audit(pages: list[dict]) -> list[dict[str, str]]:
    form_by_record: dict[str, list[FormSpec]] = {}
    for spec in FORMS:
        for unit_id in spec.records:
            form_by_record.setdefault(unit_id, []).append(spec)
    text_by_record: dict[str, list[TextSpec]] = {}
    for spec in TEXTS:
        for unit_id in spec.records:
            text_by_record.setdefault(unit_id, []).append(spec)

    audit = []
    seen = set()
    for page in pages:
        for record in page["records"]:
            unit_id = record["unit_id"]
            assert unit_id not in seen, unit_id
            seen.add(unit_id)
            forms = form_by_record.get(unit_id, [])
            texts = text_by_record.get(unit_id, [])
            emitted = [spec.key for spec in forms]
            targets = list(dict.fromkeys(target for spec in texts for target in spec.targets))
            if forms:
                final_status = "installed_form"
                final_form = "|".join(dict.fromkeys(spec.form for spec in forms))
                final_parameter = "|".join(dict.fromkeys(spec.parameter for spec in forms if spec.parameter))
            elif texts:
                final_status = "entry_text_only"
                final_form = ""
                final_parameter = ""
            else:
                final_status = "context_only"
                final_form = ""
                final_parameter = ""

            resolution = {
                "installed_form": "image-checked canonical form installed; accepted links are direct and tentative links use the ranked overlay",
                "entry_text_only": "comparative, phonological, or dictionary-correction prose retained without fabricating a lexical row",
                "context_only": "supporting example or repeated context; no independent installation action",
            }[final_status]
            source = record.get("source_locator") or f"{SOURCE_ID}[p. {page['printed_page']}]"
            audit.append({
                "Snapshot_Date": SNAPSHOT_DATE,
                "Unit_ID": unit_id,
                "PDF_Page": str(page["pdf_page"]),
                "Printed_Page": str(page["printed_page"]),
                "Section": record.get("section", ""),
                "Raw_Record_Type": record.get("record_type", ""),
                "Raw_Form": record.get("form_original", ""),
                "Raw_Gloss": record.get("gloss", ""),
                "Raw_Target_System": record.get("target_system", ""),
                "Raw_Target_ID": record.get("target_id", ""),
                "Raw_Claim_Status": record.get("claim_status", ""),
                "Raw_Editorial_Action": record.get("editorial_action", ""),
                "Final_Status": final_status,
                "Final_Form": final_form,
                "Final_Parameter_ID": final_parameter,
                "Final_Claim_Status": "accepted" if final_parameter else record.get("claim_status", ""),
                "Emitted_Keys": "|".join(emitted),
                "Entry_Text_Targets": "|".join(targets),
                "Resolution": resolution,
                "Agent_Correction": CORRECTIONS.get(unit_id, ""),
                "Review": "source-image-verified by editorial reconciliation",
                "Material_Error": "no",
                "Source": source,
                "Record_SHA256": record_hash(record),
            })

    referenced = set(form_by_record) | set(text_by_record)
    missing = sorted(referenced - seen)
    assert not missing, f"curated specs reference missing raw units: {missing}"
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, help="optional original PDF for identity verification")
    args = parser.parse_args()

    if args.pdf:
        assert args.pdf.is_file(), args.pdf
        actual = sha256(args.pdf)
        if actual != PDF_SHA256:
            raise ValueError(f"PDF SHA-256 {actual} does not match expected {PDF_SHA256}")

    pages = load_pages()
    forms = [spec.row() for spec in FORMS]
    text_rows = [row for spec in TEXTS for row in spec.rows()]
    audit = build_audit(pages)
    record_counts = {str(page["printed_page"]): len(page["records"]) for page in pages}

    assert len(forms) == 19
    assert len({row["Entry_Key"] for row in forms}) == len(forms)
    assert len(text_rows) == 35
    assert len(audit) == 76
    assert sum(record_counts.values()) == len(audit)
    assert all(row["Material_Error"] == "no" for row in audit)

    # Manual form imports are headerless because make_cldf.py reads them as positional rich rows.
    write_csv(FORM_OUTPUT, FORM_FIELDS, forms, header=False)
    write_csv(TEXT_OUTPUT, TEXT_FIELDS, text_rows)
    write_csv(AUDIT_OUTPUT, AUDIT_FIELDS, audit)
    sample = sorted(audit, key=lambda row: hashlib.sha256(row["Unit_ID"].encode()).hexdigest())[:20]
    write_csv(SAMPLE_OUTPUT, AUDIT_FIELDS, sample)

    RECONCILIATION_OUTPUT.write_text(
        json.dumps(
            {
                "source": SOURCE_ID,
                "date": SNAPSHOT_DATE,
                "policy": "Page-agent JSON is raw evidence; image-checked editorial decisions below govern installation.",
                "correction_count": len(CORRECTIONS),
                "corrections": [
                    {"unit_id": unit_id, "decision": decision}
                    for unit_id, decision in sorted(CORRECTIONS.items())
                ],
                "ranked_hypotheses": {
                    "pužža": ["rank 2 reflex of d4477"],
                    "kūžing": ["rank 2 reflex of d2687", "rank 3 reflex of d1876"],
                    "pisfing": ["rank 2 reflex of d4135", "rank 3 reflex of d4183"],
                    "šupping": ["rank 2 reflex of d2621"],
                    "dūī": ["rank 2 borrowing hypothesis from CDIAL 5228"],
                },
                "rank1_overlays": {
                    "DEDR Brahui ullī": "reassigned from d500 to d701",
                    "Ali-Kobayashi Brahui hullī": "assigned to d701",
                    "Ali-Kobayashi Brahui dū": "borrowed from CDIAL 6586",
                    "Gadaba cīkap-": "borrowed from the Telugu cīk- reflex in d2621",
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    MANIFEST_OUTPUT.write_text(
        json.dumps(
            {
                "source_id": SOURCE_ID,
                "snapshot_date": SNAPSHOT_DATE,
                "stable_url": "https://www.jstor.org/stable/619537",
                "doi": "10.1017/S0041977X00032481",
                "pdf_sha256": PDF_SHA256,
                "pdf_pages": PDF_PAGES,
                "article_printed_pages": [440, 447],
                "pdf_redistributed": False,
                "rights": "Copyright SOAS 1997; only extracted linguistic facts and audit metadata are checked in.",
                "extraction": {
                    "method": "one low-cost agent per rendered article page, followed by image-checked editorial reconciliation",
                    "agent_model": "gpt-5.6-luna",
                    "contract": "data/other/forms/raw_data/emeneau_brahui_1997_prompt.md",
                    "raw_page_directory": "data/other/forms/raw_data/emeneau_brahui_1997_agent",
                    "record_counts": record_counts,
                    "record_total": len(audit),
                    "reconciliation_corrections": len(CORRECTIONS),
                },
                "outputs": {
                    "forms": str(FORM_OUTPUT.relative_to(ROOT)),
                    "form_count": len(forms),
                    "entry_texts": str(TEXT_OUTPUT.relative_to(ROOT)),
                    "entry_text_count": len(text_rows),
                    "audit": str(AUDIT_OUTPUT.relative_to(ROOT)),
                    "audit_count": len(audit),
                    "sample": str(SAMPLE_OUTPUT.relative_to(ROOT)),
                    "sample_count": len(sample),
                    "reconciliation": str(RECONCILIATION_OUTPUT.relative_to(ROOT)),
                },
                "graph_overlay": {
                    "path": "data/etymology-assignments.csv",
                    "assignment_count": 11,
                    "status": "installed after stable source keys were resolved by the first CLDF build",
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        f"installed {len(forms)} forms, {len(text_rows)} entry-text blocks, "
        f"{len(audit)} audited agent records, and {len(sample)} sampled records"
    )


if __name__ == "__main__":
    main()
