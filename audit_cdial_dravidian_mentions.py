"""Reconcile every explicit Dravidian mention in CDIAL with a DEDR target.

This is an editorial audit, not an automatic etymology generator. Existing structured
comparisons are reported as installed; the manually reviewed table below records plausible
new targets and cases where the CDIAL wording does not support a DEDR article-level edge.
"""

import argparse
import csv
import html
import re
from pathlib import Path


ROOT = Path(__file__).parent
DRAVIDIAN = re.compile(r"\bDrav\.|\bDravidian\b", re.I)
TAG = re.compile(r"<[^>]+>")


# entry: (targets, confidence, relation, direction, rationale)
PROPOSALS = {
    "16": (["d23"], "high", "loan", "entry-from-compared", "Ta. akkai 'elder sister' is in DEDR 23."),
    "179": (["d83"], "medium", "loan", "entry-from-compared", "CDIAL sends *aṭṭ- 'obstruct' to the *aḍ- group; DEDR 83 is the matching Dravidian obstruct/stop set."),
    "222": (["d142"], "high", "loan", "entry-from-compared", "Ta. attai and the corresponding aunt/mother-in-law forms are in DEDR 142."),
    "292": (["d327"], "high", "loan", "entry-from-compared", "DEDR 327 has Ta. aṉal, Ma. anal and Ka. analu 'fire, heat'."),
    "2622": (["d1278"], "low", "related", "undetermined", "DEDR 1278 contains Ta./Ma. kari 'charcoal, lampblack' in the *kar- 'black, scorched' family; CDIAL also leaves Munda open."),
    "2652": (["d1299"], "high", "loan", "entry-from-compared", "Ka. kalapu 'miscellaneous mass' and Ma. kalappu 'whole, sum' match 'collection'."),
    "2727": (["d2766"], "high", "loan", "entry-from-compared", "DEDR 2766 ceṇṭu/ceṇḍu is the widespread Dravidian word for a playing ball."),
    "2740": (["d1342"], "low", "related", "undetermined", "DEDR 1342 includes Te. gavva 'cowrie' beside kavva- forms; the internal relationship and direction are uncertain."),
    "2764": (["d256"], "low", "related", "undetermined", "DEDR 256 is a native Dravidian lotus set, but CDIAL supplies no particular Dravidian form and the formal match is weak."),
    "2795": (["d1265"], "medium", "loan", "entry-from-compared", "DEDR 1265 contains Ka. karaku/garaku and Te. karagasamu 'saw, saw teeth', matching CDIAL's krakara-/karakaya- comparison."),
    "2851": (["d1355"], "low", "related", "undetermined", "DEDR 1355 contains Ka. garde/gadde 'field'; CDIAL cites historical galde/gardde/gadde in its Dravidian hole/field comparison."),
    "2918": (["d1297"], "high", "loan", "entry-from-compared", "DEDR 1297 contains Ta./Ka. kal- and Gondi kar- 'learn', exactly CDIAL's comparanda."),
    "2927": (["d1297"], "high", "loan", "entry-from-compared", "The 'practical art' noun is referred to kalayati; DEDR 1297 includes Ta. kalai 'arts and sciences'."),
    "2955": (["d240"], "medium", "loan", "entry-from-compared", "DEDR 240 contains Ta./Ma./Te. alai/ala 'wave'; the formal derivation of kallōla remains non-transparent."),
    "3031": (["d1418"], "medium", "loan", "entry-from-compared", "CDIAL explicitly groups kāntāra with kānana; DEDR 1418 is already compared with CDIAL kānana 3028."),
    "3037": (["d2054"], "medium", "loan", "entry-from-compared", "DEDR 2054 is the kuṭ-/koṭ- 'crooked, bent' set and includes Kolami koṭe 'false'."),
    "3149": (["d1931"], "medium", "loan", "entry-from-compared", "DEDR 1931 supplies the kēsu/kisu 'redness' base required by Mayrhofer's *kīsuka 'red tree'."),
    "3268": (["d1695"], "medium", "related", "undetermined", "DEDR 1695 guṇḍu denotes round/globular objects and beads, a plausible base for 'ring, coil'."),
    "3341": (["d1651"], "medium", "loan", "entry-from-compared", "CDIAL assigns kulāla to the Dravidian pot-word complex; DEDR 1651 is the kuṭam/koḍa pot family."),
    "3354": (["d1651"], "medium", "loan", "entry-from-compared", "CDIAL explicitly groups *kulla 'pot' with the pot words represented by DEDR 1651."),
    "3392": (["d1882"], "high", "loan", "entry-from-compared", "DEDR 1882 has Ta./Ma. kūṭṭam and Te. kūṭamu 'assembly, heap, crowd'."),
    "3393": (["d2064"], "medium", "loan", "entry-from-compared", "DEDR 2064 koṭṭ-/kuṭ- 'strike, dig' includes hoe/spade nouns appropriate to a ploughshare."),
    "3395": (["d1914"], "medium", "related", "undetermined", "DEDR 1914 contains Brahui kūṭī 'hornless', CDIAL's proposed bridge to 'defective, false'."),
    "3465": (["d2004"], "low", "related", "undetermined", "DEDR 2004 contains native names for Colocasia antiquorum, one of CDIAL kēmuka's referents; no close form survives."),
    "3502": (["d1651"], "medium", "loan", "entry-from-compared", "CDIAL sends *kōḍamba 'pot' to the same Dravidian pot complex represented by DEDR 1651."),
    "3954": (["d1519", "d1520"], "low", "related", "undetermined", "CDIAL compares Dravidian words under *gicc-; DEDR 1519 and 1520 independently cite CDIAL 4153 *gicc-."),
    "3977": (["d1148"], "medium", "loan", "entry-from-compared", "DEDR 1148 includes Te. gaḍḍa 'lump; boil, ulcer' and the wider hardening/swelling family."),
    "3997": (["d1148"], "high", "loan", "entry-from-compared", "Te. gaḍḍa 'lump; boil, ulcer' in DEDR 1148 is an exact form-and-meaning comparison."),
    "3999": (["d1337"], "medium", "loan", "entry-from-compared", "DEDR 1337 contains the Ta. kavuḷ/Ma. kaviḷ/Te. gauda 'cheek' family."),
    "4089": (["d1337"], "medium", "loan", "entry-from-compared", "DEDR 1337 is the closest Dravidian cheek family and CDIAL groups galla with gaṇḍa."),
    "4248": (["d2766"], "high", "loan", "entry-from-compared", "DEDR 2766 ceṇṭu/ceṇḍu 'playing ball' closely matches gēnduka/gēṇḍu."),
    "4474": (["d1595"], "high", "loan", "entry-from-compared", "DEDR 1595 has Ka. giṟi/giṟu and Te. giragira 'go round, whirl'."),
    "4479": (["d1946"], "medium", "loan", "entry-from-compared", "DEDR 1946 includes Tulu gaṇṭu 'ankle, knot, joint', matching the ghuṇṭa variant and joint semantics."),
    "4497": (["d1907"], "medium", "loan", "entry-from-compared", "DEDR 1907 has Gondi/Konda gūr- 'roll over', matching CDIAL's proposed Dravidian *ghūr-."),
    "4569": (["d2664"], "high", "loan", "entry-from-compared", "DEDR 2664 Ta. cuṇṭu/coṇṭu and related forms mean 'bill, beak'."),
    "5091": (["d2313"], "high", "loan", "entry-from-compared", "DEDR 2313 Ka./Te. jaḍḍa means 'union, near, connected', matching 'joins; joining, pair'."),
    "5254": (["d2648"], "low", "related", "undetermined", "DEDR 2648 contains Ka. juṅgu, but its present gloss 'dangling tatter/turban end' conflicts with CDIAL's cited 'pubic hair'."),
    "5827": (["d854"], "low", "related", "undetermined", "DEDR 854 eḷ/eḷḷu is the closest native Dravidian sesame family; CDIAL itself prefers a Munda source."),
    "6934": (["d3582"], "medium", "related", "undetermined", "DEDR 3582 naṭ-/naḍ- 'walk, move, dance, skip' is formally exact but only semantically adjacent to 'tremble, totter'."),
    "6936": (["d2909", "d2931"], "medium", "influence", "entry-from-compared", "DEDR 2909 has Ka. naḷḷu 'reed'; DEDR 2931 has Ta. ñeḷ 'be hollow' and Ka. naḷḷu 'depression', CDIAL's two cited contact forms."),
    "7075": (["d3651"], "low", "related", "undetermined", "Partial compound match only: DEDR 3651 contains Ta. nāri 'fibrous covering of a coconut-palm leaf stalk'; CDIAL's second member kēḷi has no DEDR entry."),
    "7696": (["d3082"], "medium", "related", "undetermined", "DEDR 3082 Ta. tappaṭṭam/tappaṭṭai and Te. tappeṭa denote drums and closely match paṭaha by metathesis."),
    "7780": (["d4018"], "high", "loan", "entry-from-compared", "CDIAL cites Kurux padda and cross-refers to pallī; DEDR 4018 is the paḷḷi/palle 'hamlet, village' family."),
    "8181": (["d4142"], "low", "related", "undetermined", "DEDR 4142 contains Gondi pitta- 'bile' and notes the corresponding South Dravidian pitta forms; CDIAL prefers Munda."),
    "8253": (["d4452"], "medium", "loan", "entry-from-compared", "DEDR 4452 contains pottu/pōttu 'hole, hollow, cavity', matching CDIAL's *pōṭṭa comparison."),
    "9124": (["d1688"], "low", "related", "undetermined", "DEDR 1688 is the widespread kuṇṭ-/kuṇṭi 'lame, crippled' family; CDIAL only invokes a general Dravidian/Munda bodily-defect stratum."),
    "9742": (["d4919"], "high", "loan", "entry-from-compared", "DEDR 4919 contains Ta. muṭalai 'ball, globe', the exact form cited in Burrow's *mŏṇḍale proposal."),
    "10059": (["d295", "d63"], "low", "influence", "entry-from-compared", "DEDR 295 is Ta. aḷa/aḷavu 'measure'; DEDR 63 supplies aḍagu 'be contained', the semantic alternation cited by CDIAL."),
    "10150": (["d4932"], "high", "loan", "entry-from-compared", "DEDR 4932 contains Parji muṭka 'blow with fist' and Kurux muṭkā 'fist', CDIAL's exact forms."),
    "10223": (["d4628"], "low", "related", "undetermined", "CDIAL sends musala 'pestle' to maṣati; the latter is already compared with DEDR 4628 Ka. masagu 'rub, whet'."),
    "12115": (["d5479", "d5517"], "medium", "loan", "entry-from-compared", "The old DED 4558 footer is lost, but its cited Kurux bīṛī and Malto béru survive in DEDR 5479 and 5517 respectively."),
    "13267": (["d343"], "medium", "loan", "entry-from-compared", "The old DED 288 footer is lost, but CDIAL's cited Ta. āccā 'Shorea robusta' survives in DEDR 343."),
    "13392": (["d2599"], "high", "loan", "entry-from-compared", "DEDR 2599 contains Ta. cīku 'broom-grass', CDIAL's exact comparandum."),
}


NO_TARGET = {
    "2601": "Only a tentative non-Aryan (Dravidian?) label is supplied; no Dravidian form is cited and no formal DEDR counterpart was found.",
    "2730": "CDIAL cites Te. kandamu 'neck', but that form is absent from the current DEDR; nearby neck entries are not formally comparable.",
    "2757": "CDIAL explicitly prefers a Munda source and supplies no specific Dravidian form or DEDR family.",
    "2898": "CDIAL's Ta. karumā 'smith, smelter' is absent from the current DEDR; DEDR 2133 is semantically relevant but formally unrelated.",
    "2998": "The cited Ka./Te./Ma. kakka/kākke kin terms are absent from the current DEDR.",
    "3061": "The proposed 'black-spear' analysis is a compound etymology with no corresponding whole-word DEDR entry.",
    "3082": "No Gmelina arborea name resembling kārṣmarya/kambhārī was found in the current DEDR.",
    "3173": "CDIAL gives only typological sound alternations and a possible link to 'dwarf'; no specific Dravidian lexical comparandum is named.",
    "3466": "Ma. kayyuṟa 'glove' is absent as a whole word from DEDR and is transparently a hand+cover compound, not an article-level match.",
    "3503": "CDIAL derives the word from Austroasiatic and says it was then borrowed into Dravidian; no suitable retained Dravidian loan entry exists in DEDR.",
    "4053": "The cited Ka. garduge/gaddige 'throne' and Te. gadde 'seat' are absent from DEDR; DEDR 1355 gadde 'field' is a different sense.",
    "4911": "CDIAL asserts Dravidian origin but supplies no Dravidian form; its cross-referenced cēṭa/cēṭṭa group is instead treated as Munda.",
    "4922": "The cited Ta. coṭṭu/cuṟṟu 'steal' forms are absent from DEDR; DEDR 2715 cuṟṟu means 'go around', a different homonym.",
    "5033": "No Dravidian antelope name resembling chikkāra was found; concept-only deer entries are insufficient.",
    "5329": "CDIAL's Ka. jaḍi/jiḍi 'fine continuous rain' forms are absent from the current DEDR.",
    "5528": "CDIAL cites no Dravidian form for *ḍabba 'box'; DEDR box entries such as peṭṭi are conceptually but not formally comparable.",
    "5637": "CDIAL supplies no Dravidian form for taṇḍula; native husked-rice entries are concept matches only.",
    "6173": "No Dravidian pressing verb resembling *dabb was found; superficially similar dapp- forms in DEDR have unrelated senses.",
    "6191": "The possible Dravidian attribution is attached to a broad mountain-name complex without a cited lexical form.",
    "6215": "CDIAL cites no Dravidian form for dala 'party, band'; generic crowd/group entries do not establish a formal comparison.",
    "6250": "CDIAL only allows unspecified Dravidian or Munda influence on an otherwise Indo-Aryan fang/tusk paradigm.",
    "6632": "This merely glosses the ethnonym drāviḍa as 'Dravidian'; it is not a Dravidian-source comparison.",
    "6933": "CDIAL explicitly separates naṭa 'actor' from the Dravidian-derived naṭati entry; no edge should be added.",
    "7692": "CDIAL prefers an Austroasiatic source and names no Dravidian form; apparent Dravidian paṭa forms may themselves be Indo-Aryan loans.",
    "9072": "CDIAL proposes only possible areal phonological influence on an inherited Indo-Iranian ploughshare word, without a specific Dravidian target.",
    "9720": "No Dravidian form resembling mañju 'charming' was found; semantic matches such as PDr *nal 'good, beautiful' are formally arbitrary.",
    "10086": "CDIAL explicitly rejects Bloch's Dravidian derivation; this is a negative citation, not a comparison to install.",
    "12543": "The parenthetical says the Dravidian tax words are borrowed from MIA suṅka; the direction is Dravidian from Indo-Aryan, and those loans are absent from DEDR.",
}


def clean(text: str) -> str:
    return " ".join(html.unescape(TAG.sub("", text)).split())


COMPARISON_COLUMNS = [
    "ID", "Entry_ID", "Compared_Entry_ID", "Relation", "Direction", "Confidence",
    "Source", "Evidence",
]


def comparison_rows():
    for entry_id, (targets, confidence, relation, direction, rationale) in PROPOSALS.items():
        for target in targets:
            yield {
                "ID": f"cdial:{entry_id}:dedr:{target}",
                "Entry_ID": entry_id,
                "Compared_Entry_ID": target,
                "Relation": relation,
                "Direction": direction,
                "Confidence": confidence,
                "Source": f"CDIAL[entry {entry_id}]",
                "Evidence": rationale,
            }


def append_comparisons(path: Path, additions: list[dict[str, str]]) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COMPARISON_COLUMNS:
            raise ValueError(f"unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    seen = {row["ID"] for row in rows}
    new_rows = [row for row in additions if row["ID"] not in seen]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows + new_rows)
    return len(new_rows)


def load_existing() -> dict[str, set[str]]:
    existing: dict[str, set[str]] = {}
    for path in (
        ROOT / "data/cross-family-comparisons.csv",
        ROOT / "data/manual-cross-family-comparisons.csv",
    ):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                left, right = row["Entry_ID"], row["Compared_Entry_ID"]
                if right.startswith("d"):
                    existing.setdefault(left, set()).add(right)
                if left.startswith("d"):
                    existing.setdefault(right, set()).add(left)
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--install", action="store_true",
        help="append reviewed proposals to the manual and compiled comparison sidecars",
    )
    args = parser.parse_args()

    entries = []
    with (ROOT / "data/cdial/params.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 4 and DRAVIDIAN.search(row[3]):
                entries.append((row[0], row[1], clean(row[3])))

    existing = load_existing()

    unresolved = {entry_id for entry_id, _, _ in entries if not existing.get(entry_id)}
    reviewed = set(PROPOSALS) | set(NO_TARGET)
    missing_reviews = unresolved - reviewed
    stale_no_targets = set(NO_TARGET) - unresolved
    mismatched_installed = {
        entry_id for entry_id, (targets, *_rest) in PROPOSALS.items()
        if entry_id not in unresolved and not set(targets) <= existing.get(entry_id, set())
    }
    if missing_reviews or stale_no_targets or mismatched_installed:
        raise ValueError(
            f"manual review coverage drift: missing={sorted(missing_reviews)}, "
            f"stale_no_target={sorted(stale_no_targets)}, "
            f"mismatched_installed={sorted(mismatched_installed)}"
        )

    if args.install:
        additions = list(comparison_rows())
        manual_added = append_comparisons(
            ROOT / "data/manual-cross-family-comparisons.csv", additions
        )
        compiled_added = append_comparisons(ROOT / "cldf/comparisons.csv", additions)
        print(f"installed {manual_added} manual and {compiled_added} compiled comparisons")
        existing = load_existing()

    output = ROOT / "data/cdial-dravidian-mention-audit.csv"
    columns = [
        "CDIAL_ID", "Headword", "Status", "DEDR_IDs", "Confidence", "Relation",
        "Direction", "Rationale", "Evidence",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for entry_id, headword, evidence in entries:
            if existing.get(entry_id):
                data = {
                    "Status": "installed-proposal" if entry_id in PROPOSALS else "existing-match",
                    "DEDR_IDs": " ".join(sorted(existing[entry_id])),
                    "Confidence": "",
                    "Relation": "",
                    "Direction": "",
                    "Rationale": "Already represented by a structured comparison.",
                }
            elif entry_id in PROPOSALS:
                targets, confidence, relation, direction, rationale = PROPOSALS[entry_id]
                data = {
                    "Status": "partial-component-match" if entry_id == "7075" else "proposed-match",
                    "DEDR_IDs": " ".join(targets),
                    "Confidence": confidence,
                    "Relation": relation,
                    "Direction": direction,
                    "Rationale": rationale,
                }
            else:
                data = {
                    "Status": "no-defensible-target",
                    "DEDR_IDs": "",
                    "Confidence": "",
                    "Relation": "",
                    "Direction": "",
                    "Rationale": NO_TARGET[entry_id],
                }
            writer.writerow({"CDIAL_ID": entry_id, "Headword": headword, "Evidence": evidence, **data})

    print(
        f"wrote {len(entries)} rows: "
        f"{sum(bool(existing.get(i)) for i, _, _ in entries)} installed, "
        f"{len(PROPOSALS)} reviewed proposals, {len(NO_TARGET)} without defensible target"
    )


if __name__ == "__main__":
    main()
