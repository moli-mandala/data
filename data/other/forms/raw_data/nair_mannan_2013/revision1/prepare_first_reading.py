"""Exhaustive source-local first reading of Nair's Mannan chapter, pp45–49.

No canonical writes. Source correspondences are comparison claims, not ancestry.
The dotted/underlined glyphs encode source shapes, not reconstructed IPA.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "data/SOURCE_INGESTION_CHECKLIST.md").exists())
PDF = ROOT / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
EXPECTED = "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
units = []


def w(form, gloss="", scope="Mannan target", tags="", note=""):
    return dict(literal_first_reading=form, printed_gloss=gloss, language_scope=scope,
                provisional_tags=tags, note=note)


def add(page, section, n, kind, witnesses=(), claim="", disposition="review_candidate", note=""):
    key = f"nair2013mannan:p{page}:{section}:{n}"
    witnesses = [dict(v, witness_key=f"{key}:w{i}") for i, v in enumerate(witnesses, 1)]
    units.append(dict(source_unit_key=key, printed_page=page, pdf_page=page, kind=kind,
        source_witnesses=witnesses, source_claim=claim, disposition=disposition, note=note,
        review="complete owner first reading; independent literal, metadata and final-output review pending"))


add(45, "identity", 1, "documentary", [w("mannaan", scope="community name")],
    claim="Mannan is said to be confined to Kerala, mainly Thodupuzha, Devikolam, Udambanchola and Peerumede taluks of Idukki. The community claims migration from Madurai; 1991 Census population 42,221.",
    disposition="documentary", note="Parenthetical community name is not an explicitly attributed Mannan-language lexical utterance.")
add(45, "classification", 1, "source commentary", claim="Nair calls Mannan speech closely related to Malayalam and a possible Malayalam dialect, while noting distinctive phonology and lexical items, no personal terminations, and no copula.", disposition="context")
add(45, "phoneme-inventory", 1, "source commentary", [w("l̲", scope="source phoneme symbol"), w("ñ", scope="source phoneme symbol"), w("ŋ", scope="source phoneme symbol"), w("n̲", scope="source phoneme symbol")],
    claim="Vowels are said similar to Malayalam; Mannan is said to lack these consonant phonemes. The drawn l/n lowlines are distinct from dotbelow and are missing from the PDF text layer.", disposition="context")

pair_classes = [
 (45, "l-to-c", "Malayalam and other Southern Dravidian underlined l is said to become c in Mannan.", [("mal̲a","maca","rain"),("pal̲am","pacam","fruit"),("eel̲u","eecu","seven"),("kool̲i","kooci","fowl")]),
 (45, "st-t-to-c", "Malayalam -st- and –t- correspond to c in some Mannan words.", [("meestiri","meecciri","mason"),("kotukŭ","kocukŭ","mosquito")]),
 (45, "medial-to-y", "Word-medial j, v and ṣ of Malayalam are said to become y in many words; do not alter the plain s actually printed in masi.", [("puujaari","puuyaari","temple priest"),("koovil","kooyil","temple"),("masi","mayi","ink")]),
 (45, "rr-to-tt", "Source describes ṛṛ becoming tt.", [("kaaṛṛu","kaattu","wind"),("peṛṛu","pettu","delivered"),("muṛṛam","muttam","courtyard")]),
 (45, "s-h-to-k", "Source describes Malayalam ṣ and h becoming k in Mannan; examples continue on p46.", [("vaṛṣam","varikam","year")]),
 (46, "s-h-to-k", "Continuation of explicit Malayalam/Mannan comparison heading and statement on p45.", [("simham","cimukam","lion"),("sahaayam","cakaayam","help")]),
 (46, "p-k-ph-to-v", "Source describes p, k and ph becoming v.", [("parippu","varippu","grain/seed"),("kapham","kavam","phlegm"),("mutuku","mutuvu","back")]),
 (46, "nasal-clusters", "Source-underlined nn between short vowels becomes ṇṇ; after long vowels becomes ṇ. Preserve source-drawn lowlines on both Malayalam n glyphs.", [("pan̲n̲i","paṇṇi","pig"),("on̲n̲u","oṇṇu","one"),("muun̲n̲u","muuṇu","three")]),
 (46, "final-l", "Source says Malayalam word-final ḷ is invariably dropped in Mannan.", [("makkaḷ","makka","children"),("makaḷ","maka","daughter"),("tinkaḷ","tinka","moon")]),
 (46, "initial-c-s", "Source says initial Malayalam c/s is dropped in many Mannan words.", [("suuci","uuyi","needle"),("ciraku","uraku","wing")]),
 (46, "inserted-vowels", "Source describes consonant clusters replaced by syllables with inserted vowels.", [("pṛaaṇi","puraani","insects"),("pṛiyam","puriyam","affection")]),
]
for page, section, claim, pairs in pair_classes:
    add(page, section, 0, "phonological commentary", claim=claim, disposition="context")
    for i, (ml, target, gloss) in enumerate(pairs, 1):
        add(page, section, i, "comparison table", [w(ml, gloss, "Malayalam comparison"), w(target, gloss)], claim=claim,
            note="Source-correspondence comparison only; no historical etymology, borrowing or variant edge inferred.")

add(46, "phonological-symbols", 1, "source phonological notation", [w("-st-",scope="Malayalam sound sequence"),w("–t-",scope="Malayalam sound sequence"),w("c",scope="Mannan sound symbol"),w("j",scope="Malayalam sound symbol"),w("v",scope="Malayalam sound symbol"),w("ṣ",scope="Malayalam sound symbol"),w("y",scope="Mannan sound symbol"),w("ṛṛ",scope="Malayalam sound sequence"),w("tt",scope="Mannan sound sequence"),w("h",scope="Malayalam sound symbol"),w("k",scope="Mannan sound symbol"),w("p",scope="Malayalam sound symbol"),w("ph",scope="Malayalam sound sequence"),w("n̲n̲",scope="Malayalam sound sequence"),w("ṇṇ",scope="Mannan sound sequence"),w("ṇ",scope="Mannan sound symbol"),w("ḷ",scope="Malayalam sound symbol"),w("c/s",scope="Malayalam sound notation")], disposition="context", claim="Symbols explicitly occur in the p45–46 phonological statements; they are not independently glossed lexical words or identifiable morphology.", note="Page46 is the accounting endpoint; individual symbol evidence spans pp45–46 and is preserved by the source text/image archive.")
units[-1]["printed_pages"] = [45, 46]
units[-1]["count_semantics"] = "context-symbol inventory, not a census of every repeated symbol occurrence"

add(46, "kin-plural", 0, "bound morphology", [w("–laam", tags="affix suffix pl")], claim="Plural suffix occurs after kinship terms. Initial en-dash is printed; preserve it in raw literal reading.")
for i,(form,gloss) in enumerate([("appa-laam","fathers"),("colla-laam","younger brothers"),("peeṭṭa-laam","brothers-in-law"),("acca-laam","elder brothers")],1):
    add(46,"kin-plural",i,"plural noun example",[w(form,gloss,tags="noun pl")])
add(46,"other-plural",0,"bound morphology",[w("kaaṭ-u",tags="affix suffix pl")],claim="Another plural marker attested after a few nouns; source prints kaaṭ-u with no supplied leading hyphen.")
for i,(form,gloss) in enumerate([("kuñci-kaaṭ-u","children"),("teeŋka-kaaṭ-u","coconuts"),("atu-kaaṭ-u","those"),("kal-kaaṭ-u","stones")],1):
    add(47,"other-plural",i,"plural example",[w(form,gloss,tags="pl")],note="Last source gloss has an unmatched opening quote; lexical gloss preserved without editorial quotation delimiter.")
add(47,"plural-comparison",0,"source morphology comparison",[w("-kaḷ",scope="generic Dravidian comparison",tags="affix suffix pl")],claim="Nair says kaaṭ-u is unattested in other Dravidian languages and could be related to the usual Dravidian plural -kaḷ.",note="Tentative relationship; generic Dravidian is not a unique attested language route. No ancestry edge inferred.")
add(47,"case-suffix",0,"bound morphology",[w("–e",tags="affix suffix acc"),w("–e",scope="Malayalam comparison",tags="affix suffix acc"),w("-kku",tags="affix suffix acc dat")],claim="Mannan employs accusative –e found in Malayalam also, but more often -kku marks both accusative and dative.",note="The shared –e is one physical printed suffix with two explicitly named language scopes, not two printed occurrences.")
for i,(form,gloss,tags) in enumerate([("en-akk-u aṭiccu","beat me","multiword-expression acc"),("maratt-ukk-u veṭṭi","cut the tree","multiword-expression acc"),("avan-ukk-u talli","beat him","multiword-expression acc"),("puun-ee-kk-u","to the cat","dat"),("nin-akk-u","to you","pron second-person dat"),("avaḷ-ukk-u","to her","pron third-person sg f dat")],1):
    add(47,"case-suffix",i,"whole case example",[w(form,gloss,tags=tags)],claim="Source explicitly marks accusative and dative example sets; no tense/mood inferred from ambiguous English beat/cut.")
for i,(form,gloss,tags) in enumerate([("een","","pron first-person sg"),("eŋka","","pron first-person pl exclusive"),("naŋka","","pron first-person pl inclusive"),("nii","","pron second-person sg"),("niŋka","","pron second-person pl")],1):
    add(47,"personal-pronouns",i,"pronoun table",[w(form,gloss,tags=tags)],claim="Columns explicitly Sg./Pl.; rows I P./II P.; source labels Excl./Incl. for first-person plurals.")
for i,(form,tags,claim) in enumerate([
 ("avan","pron demonstrative m","Dist Mas"),("avaḷu","pron demonstrative f","Dist Fem. left cell"),("ava","pron demonstrative f","Dist Fem. right cell"),("atu","pron demonstrative n","Dist Neu."),
 ("ivan","pron demonstrative m","Prox Mas"),("ivaḷu","pron demonstrative f","Prox Fem. left cell"),("iva","pron demonstrative f","Prox Fem. right cell"),("itu","pron demonstrative n","Prox Neu.")],1):
    add(47,"demonstratives",i,"demonstrative table",[w(form,tags=tags)],claim=claim,
        note="Column placement resembles preceding Sg./Pl. personal-pronoun table, but the demonstrative subsection does not repeat number headings. Number/person claims need independent metadata review rather than inferred from coordinate alone.")
add(47,"present",0,"bound morphology",[w("–nnu",scope="Malayalam comparison",tags="affix suffix pres"),w("–nu",tags="affix suffix pres")],claim="Malayalam present –nnu becomes –nu in Mannan; no lowline is visible in this printed occurrence, unlike p46 nn controls.")
for i,(form,gloss) in enumerate([("poo-nu","goes"),("kol-nu","kills"),("cuṭ-u-nu","bakes"),("iṭ-inu","puts"),("var-inu","comes")],1):
    add(47,"present",i,"present verb example",[w(form,gloss,tags="verb pres")],note="The bakes gloss lacks a closing quote in the source.")
add(47,"past-negative",0,"bound morphology",[w("aa",tags="affix suffix neg pret"),w("–aat",scope="Malayalam comparison",tags="affix suffix neg")],claim="Source calls plain aa the past negative suffix before verbal participles, unlike Malayalam –aat. No leading hyphen is printed on aa.")
for i,(form,gloss) in enumerate([("var-aa-pooyi","having not come, went"),("kaaṇ-aa-pooyi","having not seen, went")],1):
    add(47,"past-negative",i,"whole negative construction",[w(form,gloss,tags="verb neg pret")])
add(47,"nonpast-negative",0,"bound morphology",[w("-aat",tags="affix suffix neg"),w("–att",tags="affix suffix neg")],claim="Source explicitly calls -aat and –att non-past negative suffixes. Whole example pook-aatt-u has aatt rather than att; preserve the printed difference.")
for i,(form,gloss) in enumerate([("kaaṇ-aat-e","without seeing"),("pook-aatt-u","will not go"),("pook-aat-u","do not go"),("kaaṇ-aat-u","will not see"),("cirri-kk-aat-u","do not laugh")],1):
    add(47,"nonpast-negative",i,"whole negative construction",[w(form,gloss,tags="verb neg")],claim="Examples occur under non-past negative heading; future/imperative subtypes are not mechanically inferred from English.")
for i,(form,gloss) in enumerate([("itu een pookina kuura","this is the house where I am going"),("ninaṭṭu vantatu en makanu","the one who came yesterday is my son")],1):
    add(48,"no-copula",i,"whole sentence",[w(form,gloss,tags="sentential")],claim="Source says Mannan has no copula verb unlike Malayalam.")
for i,(form,gloss) in enumerate([("amma vantu","mother came"),("makanu vantu","son came"),("koocci vantu","fowl came")],1):
    add(48,"no-personal-endings",i,"whole sentence",[w(form,gloss,tags="sentential")],claim="Source says Mannan, like Malayalam, has no personal terminations.",note="koocci is source-specific and differs from the earlier kooci comparison; do not harmonize.")

vocabulary = [
 ("ippiṭe","calling a girl"),("icca","calling a bor"),("iḷaya vaṭṭam","leader of youngsters"),("iiṭu","weight"),
 ("ekaram","branch of a tree"),("ecpiṭi","plenty"),("naccataan","a type of rat"),("attee","an expression of surprise,"),
 ("akappa","winnowing fan"),("aaṇam","curry"),("olli","emaciated person"),("utaṭu","lip, beak"),
 ("uḷa","feather"),("uḷantaari","young man"),("ulantaaricci","young woman"),("paṭṭippuli","fox"),
 ("pani","fog"),("paaḷi","bush"),("tappaṛu","rest"),("tookku","tail"),("tuṛam","bravery"),("cimpu","petal"),
 ("cimp","to carry on head"),("ceeti","news"),("akkan","sediment of oil"),("kunnaṭi","arm pit"),
 ("karinkaṇa","millipede"),("kaalmontu","feet"),("koyyan","parrot"),("kuṇayan","boy"),("kuṇacci","girl"),
 ("kuṇṇi","dragon fly"),("kuuvilaan","cuckoo"),("miṭa","wall"),("meppu","shoulder"),("meeṭṭa","step")]
for i,(form,gloss) in enumerate(vocabulary,1):
    add(48,"vocabulary",i,"lexical list",[w(form,gloss)],claim="Nair describes these as peculiar Mannan lexical items not found in other languages of the area.",
        note="The source typo bor and surprise gloss trailing comma are literal. Quote delimiter omissions are documentary typography, not silently corrected lexical senses." if i in (2,8,25) else "")

kinship = [
 (48,"coḷḷan",["Father’s elder /younger brother’s son (younger to the addressee.)","Mother’s elder /younger brother’s daughter (younger to the addressee)"]),
 (48,"coḷḷi",["Father’s elder/younger brother’s daughter (younger to the addressee)","Mother’s elder/younger sister’s daughter (younger to the addresse)"]),
 (48,"peettan",["fathers elder/younger sister’s son (elder to the addressee)","Mother’s elder /younger brother’s son (elder to the addressee)"]),
 (49,"peetti",["father’s elder/younger sister’s daughter (elder to the addressee)","Mother’s elder /younger brother’s daughter (elder to the addressee)"]),
 (49,"maṭayan",["fathers elder/younger sister’s son (younger to the addressee)","Mother’s elder /younger brother’s son (younger to the addressee)"]),
 (49,"maaḷa",["father’s elder/younger sister’s daughter (younger to the addressee)","Mother’s elder /younger brother’s daughter (younger to the addressee)"]),
 (49,"kaṛuttamma",["Mother’s younger brother’s wife.","Mother-in-law"])]
for i,(page,form,definitions) in enumerate(kinship,1):
    add(page,"kinship",i,"head with two definitions",[w(form,"; ".join(definitions),tags="noun")],claim="Some kinship terms are said unique.",note="Both printed definitions preserved as a joint head gloss; no invented subform or reordered relationship.")
    units[-1]["printed_definitions"] = definitions

names = "ilunki ilunkan acaki acakan ariyan ariyaalu pericci periyan pancan pancaaṇṭi parayan palicci palyan paappu paappan ponnaayi ponnaayan cinavan cakki cakkan kaṇṭaru kaṛuppi kaatiyal kompi kompan kuppi kuppan naacci naayan viiru".split()
add(49,"traditional-names",1,"documentary personal-name list",[w(form,scope="traditional personal name") for form in names],claim="Some people retain these traditional names despite most adopting fashionable names of plains people; no lexical meanings or language attribution is provided.",disposition="documentary")
add(49,"bibliography",1,"documentary",claim="Annie Joseph, Description of the Tribal dialects of Devikolam, Unpublished Ph.D. Thesis, Department of Linguistics, University of Kerala, 1980.",disposition="documentary",note="Auxiliary source reported by Nair, not independently collated here.")
add(50,"outside-phonology",1,"outside chapter source commentary",claim="The following Muduga chapter explicitly compares Mannan: Muduga, like Mannan and some other tribal speeches, is said to lack palatal/velar/dental nasals and the fricative continuant; unlike Mannan, Muduga has voiced stops.",disposition="context",note="Original p50 independently viewed. No new Mannan lexical forms are attested here; all surrounding pronouns, suffixes and paired lexical forms are explicitly Muduga/Malayalam controls belonging to that chapter. Contents p3 is documentary; p44 ends the preceding Malaveda chapter.")


def main():
    assert hashlib.sha256(PDF.read_bytes()).hexdigest() == EXPECTED
    keys = [u["source_unit_key"] for u in units]
    assert len(keys) == len(set(keys))
    out = HERE / "physical-units-first-reading.jsonl"
    out.write_text("".join(json.dumps(u,ensure_ascii=False,sort_keys=True)+"\n" for u in units))
    counts = Counter()
    pages = {}
    for u in units:
        counts[u["disposition"]] += 1
        pages.setdefault(str(u["printed_page"]),Counter())[u["kind"]] += 1
        for witness in u["source_witnesses"]:
            counts["source_witness_records"] += 1
            counts[witness["language_scope"]] += 1
    status = dict(status="complete owner inventory/first reading; independent review pending; no canonical installation",
        source_pdf_sha256=EXPECTED, chapter_pages=list(range(45,50)), audit_groups=len(units),
        outside_commentary_pages=[50],
        counts=dict(sorted(counts.items())), per_page={p:dict(c) for p,c in pages.items()},
        first_reading_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        shared_printed_witness="case-suffix:0 –e has two explicitly named language scopes, one printed occurrence",
        count_semantics="208 source-witness records include a shared –e scope expansion and 22 documentary/context phoneme/symbol records; they are not a count of independently printed lexical tokens",
        source_defined_identity="Mannan, existing mann1246 base; author’s possible Malayalam-dialect claim retained, not used to merge registry identities",
        pending=["complete independent original/metadata review","font lowline and PUA crosscheck","identity/reference/profile proposals","graph review","fresh final-output audit","actual parser and focused tests","root authorization before source-stage install","full CLDF build/full suite/browser QA deferred"])
    (HERE / "first-reading-status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(status,ensure_ascii=False))


if __name__ == "__main__": main()
