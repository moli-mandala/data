"""Physical-unit census of the complete Nair 2013 Adiya chapter.

This is a page-block inventory, not a normalized transcription or output CSV.
It deliberately counts repeated forms and comparison controls as source evidence.
"""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[6] / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
DATA = HERE.parents[5]

# Page numbers are both the printed folios and one-based PDF pages in this scan.
# A physical block may contain multiple printed forms, slash variants, and prose
# witnesses. Those are enumerated separately in the next first-reading layer.
BLOCKS = [
    (4, "community-names", "documentary", "Adiya/atiya and self-name raavuleeru; locality and census context"),
    (4, "vowel-a-length-table", "comparative table", "3 Malayalam–Adiya pairs: inside, lac, pestle"),
    (4, "vowel-a-to-e-table", "comparative table", "3 Malayalam–Adiya pairs: distance, day-time, sea"),
    (4, "final-vowel-table", "comparative table", "4 Malayalam–Adiya–Kannada triples: jaundice, liking, crow, elephant"),
    (4, "rounded-u-table", "comparative table", "3 Malayalam–Adiya pairs: eye, nose, broom"),
    (4, "l-to-v-table", "comparative table", "3 Malayalam–Adiya pairs: neck, seven, depth"),
    (4, "v-to-b-table", "comparative table", "3 Malayalam–Adiya pairs: stick, bangle, basket"),
    (5, "plural-morphology", "morphology and controls", "Adiya -a; Malayalam -kaḷ and ava/iva comparison claims"),
    (5, "plural-table", "comparative table", "3 Malayalam–Adiya pairs: cats, children, Paniya women"),
    (5, "pronoun-paradigm", "paradigm", "I/II/III and honorific columns; slash alternatives and repeated cells retained"),
    (5, "first-person-variation", "alternative and expression", "naanu/naa and poyuley 'I went'"),
    (5, "first-person-oblique", "morphology and expressions", "en- with two whole clauses, hut/salt and want"),
    (5, "second-person-distribution", "grammatical prose", "nii/niiyu, innu, na, iñcu, plural/honorific niinka and postposition okka"),
    (5, "all-of-you-expression", "whole expression", "niŋkaḷokkaayi mubari"),
    (5, "kin-address-pronouns", "grammatical prose", "niiŋka versus nii by affinity/seniority"),
    (6, "third-person-gender-examples", "expressions and controls", "4 masculine/non-feminine phrases plus 2 feminine pronouns; Malayalam parentheses separately attributed"),
    (6, "elder-kin-demonstratives", "whole expressions", "2 respectful elder-kin clauses with atu; source restricts itu/atu"),
    (6, "younger-kin-demonstratives", "whole expressions", "2 daughter/son clauses; younger-kin contrast"),
    (6, "three-tense-examples", "whole expressions", "3 I see paradigmatic clauses"),
    (6, "personal-termination-examples", "whole expressions", "6 person/number clauses including two questions"),
    (6, "addressee-sex-markers", "morphology and expressions", "-va/-ma and two come-here imperatives"),
    (6, "affinal-address-mi", "morphology and expressions", "-mi and two give imperatives"),
    (7, "affinal-address-i", "morphology and expressions", "-i and two come-here imperatives"),
    (7, "negative-kaani", "morphology and expressions", "-kaaṇi with 4 full clauses; tense read from context"),
    (7, "negative-a", "morphology and expressions", "-a with 3 full clauses; Tamil/Malayalam comparison prose"),
    (7, "archaic-ula", "morphology and expressions", "uḷa with 3 full clauses; old-Malayalam comparison"),
    (7, "kinship-terms", "glossary prose", "all kin terms, slash alternatives, address/reference distinctions and extended senses"),
    (7, "bibliography-heading", "documentary", "bibliography begins at page bottom; entries continue on p8"),
    (8, "bibliography-continued", "documentary", "4 additional cited theses, 1978; no target forms"),
]

rows = []
for page, section, kind, contents in BLOCKS:
    rows.append({
        "unit_key": f"nair2013adiya:p{page}:{section}",
        "printed_page": page,
        "pdf_page": page,
        "kind": kind,
        "source_scope": contents,
        "status": "physical census; individual glyphs and witness slots pending first reading",
    })
assert len(rows) == 29 and len({r["unit_key"] for r in rows}) == 29
out = HERE / "physical-unit-census.jsonl"
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
with (DATA / "cldf/languages.csv").open(newline="") as stream:
    languages = list(csv.DictReader(stream))
assert any(r["ID"] == "Ravula" and r["Glottocode"] == "ravu1237" for r in languages)
ravula_rows = 0
for source_csv in (DATA / "data/other/forms").glob("*.csv"):
    with source_csv.open(newline="") as stream:
        for row in csv.reader(stream):
            ravula_rows += bool(row and row[0] == "Ravula")
assert ravula_rows == 148, ravula_rows
status = {
    "scope": "Nair 2013 Adiya chapter pp4–8, including comparison controls and bibliography",
    "physical_units": len(rows),
    "pages": {str(p): sum(r["printed_page"] == p for r in rows) for p in range(4, 9)},
    "pdf_sha256": sha(PDF),
    "census_sha256": sha(out),
    "existing_jambu_adiya_rows": 0,
    "existing_jambu_adiya_base_language": False,
    "existing_jambu_ravula_base_language": True,
    "existing_jambu_ravula_rows": ravula_rows,
    "source_wide_outside_chapter_mentions": "p9 Aranadan and p33 Kani make feature-level comparison statements but print no additional Adiya lexical form",
    "limits": "Printed form slots and normalization are not yet certified; source glyph review and full proposal are pending.",
    "canonical_changes": False,
}
(HERE / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
print(status)
