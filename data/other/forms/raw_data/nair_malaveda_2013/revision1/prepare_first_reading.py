"""Complete owner first reading of Nair's Malaveda/Veda chapter, pp40–44.

Provisional literal witnesses only: independent original-page review, source
identity, bibliography, graph, metadata and installation gates remain pending.
"""
import hashlib
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOT = json.loads((HERE / "snapshot.json").read_text())
assert SNAPSHOT["source_pdf_sha256"] == "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
blocks = []


def add(page, block, specs, claim="", language="Veda source lect", disposition="target"):
    forms = []
    for index, spec in enumerate(specs, 1):
        fields = spec.split("|")
        assert 3 <= len(fields) <= 4
        form, gloss, tags = fields[:3]
        note = fields[3] if len(fields) == 4 else ""
        assert form and unicodedata.is_normalized("NFC", form)
        forms.append({"witness_key": f"nair2013malaveda:p{page}:{block}:w{index}",
                      "literal_first_reading": form, "printed_gloss": gloss,
                      "provisional_tags": tags, "note": note, "language_scope": language,
                      "disposition": disposition})
    blocks.append({"unit_key": f"nair2013malaveda:p{page}:{block}", "printed_and_pdf_page": page,
                   "source_claim": claim, "witnesses": forms})


add(40, "community-scope", ["Veda||", "Malaveda||", "Vettuvar||", "Ulladans||", "Malappandarams||"],
    "English ethnographic names; Nair explicitly warns the named communities may not be the same tribe. Pathanamthitta 1977 and Kollam 1991 speech must not be silently merged.",
    language="documentary community labels", disposition="documentary")
add(40, "phonological-context", [], "Source describes Veda stops as lacking Malayalam's Indo-Aryan-derived voiced/aspirated series; no segment inventory printed.")
add(40, "first-person-singular", [
    "ñaanu||pron 1sg nom", "en-||pron 1sg obl", "ennaaru|I (Dat)|pron 1sg dat", "enṛu|my|pron 1sg gen poss"],
    "Source states nominative form and oblique base; the two case examples are explicitly glossed.")
add(40, "first-person-plural", [
    "ñaamu||pron 1pl|Source says there is no inclusive-exclusive distinction; do not assign either value.",
    "om||pron 1pl obl", "ommaaru|we (Dat)|pron 1pl dat"], "Source explicitly says the same nominative serves both inclusive and exclusive.")
add(40, "second-person", [
    "iyyu||pron 2sg nom", "iññu||pron 2sg obl", "iññaaru|you (Dat)|pron 2sg dat",
    "iiru||pron 2pl nom", "imm-||pron 2pl obl", "immaaru|you (Dat)|pron 2pl dat"])
add(40, "third-person", [
    "avaru||pron 3sg m", "avaru||pron 3pl m", "atuvu||pron 3sg f", "avaru||pron 3pl f",
    "attu||pron 3sg n", "avvu||pron 3pl n",
    "veeṛu||pron m prox|Source gives a single form for masculine singular and plural; number is not split into invented attestations.",
    "ituvu||pron 3sg f prox", "veeṛu||pron 3pl f prox",
    "ivvu||pron n prox|Source calls this the neuter proximate form without an explicit number."],
    "Nair subsequently says 'proximate forms beginning in a are distant'; the contradictory wording is retained as source uncertainty, not an independently settled deixis analysis.")
add(41, "neuter-base", ["at||pron stem n"], "Source claims feminine singular is derived from a neuter base at and proposes an earlier masculine/nonmasculine division; no external graph parent is asserted.")
add(41, "gender-markers", ["–aan||suffix m", "–an||suffix m", "atti||suffix f", "-aṭṭi||suffix f", "-cci||suffix f"], "Gender markers said to match Malayalam; printed dash styles and absence of a dash on atti are retained.")
add(41, "gender-examples", ["tampuraan|respectable person|noun", "mutian|grand father|noun kinship", "areyatti|fisher woman|noun f", "ñukkatti|old woman|noun f"])
add(41, "accusative-markers", ["–avu||suffix acc", "–a||suffix acc"])
add(41, "accusative-examples", ["tokkaliyavu|dog (Acc.)|noun acc", "imma|you (Acc.)|pron acc"])
add(41, "dative-markers", ["–aaru||suffix dat", "–e||suffix dat"])
add(41, "dative-comparison", ["–e||suffix acc"], "Explicit Malayalam comparison of the Veda dative marker with an accusative marker.", language="Malayalam", disposition="comparison")
add(41, "dative-comparison-tamil", ["–e||suffix acc"], "Explicit Tamil comparison of the Veda dative marker with an accusative marker.", language="Tamil", disposition="comparison")
add(41, "dative-free-variants", ["ennaaru|I (Dat.)|pron 1sg dat", "enne|I Dat.|pron 1sg dat"], "Source explicitly calls these two forms free variants.")
add(41, "genitive-markers", ["-uṭe||suffix gen", "-eṭe||suffix gen", "iṭe||suffix gen", "-aṭe||suffix gen", "–ṭe||suffix gen"])
add(41, "genitive-examples", [
    "avariṭe|their|pron gen", "kiḷani-ite|of the sister|noun gen", "mele-eṭe|of moutain|noun gen",
    "iññ-aṭe|of you|pron gen", "atuv-uṭe|of her|pron gen", "tokkali-ṭe|of the dog|noun gen"],
    "Segmented example spellings and the printed gloss typo moutain are retained.")
add(41, "locative-markers", ["u||suffix loc", "ē||suffix loc", "–ilu||suffix loc", "ī||suffix loc", "–ikkam||suffix loc", "-akkam||suffix loc", "–ukkam||suffix loc"], "Source lists u/ē or –ilu/ī as free variation plus three other suffixes; exact pair scope awaits independent review.")
add(41, "locative-examples", [
    "kiḷani-ilu|in sister|noun loc", "kilannii|in sister|noun loc",
    "peru- eelu|in house|noun loc", "pere|in house|noun loc",
    "kayy-ikkam|in hand|noun loc", "coluŋk-akkam|in head|noun loc", "atuv-ukkam|in her|pron loc"], "Two printed arrows relate segmented and surface locatives; preserve both written witnesses, without silently respelling them.")
add(41, "numeral-ending", ["–avu||suffix num"], "Source describes this as a peculiar ending on some numerals.")
add(41, "numerals", [
    "onṛu|one|num", "reeṭu|two|num", "muuṛu|three|num", "muppatavu|thirty|num", "naalu|four|num", "naalpatavu|forty|num",
    "n̲aanuuṛavu|four hundred|num", "añcu|five|num", "onpatavu|fifty|num uncertain|Printed fifty retained; no correction to a presumed numeral value.",
    "aññuuṛavu|five hundred|num", "aaṛu|six|num", "aṛupatavu|sixty|num", "aṛnuuṛavu|six hundred|num"])
add(42, "numerals-continuation", [
    "aaṛaakkiram|six thousand|num", "eel̲u|seven|num", "el̲upatavu|seventy|num", "el̲unuuṛavu|seven hundred|num",
    "eṭtu|eight|num", "eṇpatavu|eighty|num", "eṇṇuuṛavu|eight hundred|num", "ompatavu|nine|num",
    "toṇṇuuṛavu|ninety|num", "pattavu|ten|num", "irupatu|twenty|num"])
add(42, "past-markers", ["ñc||suffix pret", "-cc-||suffix pret", "-ṭṭ-||suffix pret", "-t-||suffix pret", "-ṭ-||suffix pret"])
add(42, "past-examples", ["ceñceen|I did|verb pret 1sg", "naṭanteen|I walked|verb pret 1sg", "varicceen|I came|verb pret 1sg", "maṛantēn|I forgot|verb pret 1sg"])
add(42, "present-markers", ["–tta||suffix pres", "–ta||suffix pres"])
add(42, "present-examples", ["cal̲ttaveen|I speak|verb pres 1sg", "pookutaveen|I go|verb pres 1sg"])
add(42, "future-markers", ["–pp-||suffix fut", "-pa-||suffix fut", "–a-||suffix fut"])
add(42, "future-examples", ["timpa|will eat|verb fut", "uri-ppoorin|you will not eat|verb fut neg", "cali –pp- oorin|you will not speak|verb fut neg", "viiv-a|will fall|verb fut"], "The source prints segmented and spaced forms; preserve whole examples and negation, not inferred stems.")
add(42, "relative-participle-markers", ["a||suffix relative participle", "-een||suffix relative participle", "–an||suffix relative participle", "–e||suffix relative participle"])
add(42, "relative-participle-examples", ["variccan|come (which)|verb relative participle", "varaatta|did not come (which)|verb relative participle neg", "varitteen|coming (which)|verb relative participle", "ceyaatte|not doing (which)|verb relative participle neg"])
add(42, "negative-participle-markers", ["-aatt||suffix neg participle", "-att-||suffix neg participle"])
add(42, "negative-participle-examples", ["n̲aakuvaata|not laughing|verb neg participle", "n̲aakuvatte|not laughing|verb neg participle"])
add(42, "imperative-singular-markers", ["-u-||suffix impv sg"], "Source also mentions no suffix; the zero-exponent condition is documentary, not an emitted sound form.")
add(42, "imperative-singular-examples", ["vera|come (you)|verb impv sg", "pooku|go|verb impv sg"])
add(42, "imperative-plural-markers", ["-mi-||suffix impv pl", "-miiru||suffix impv pl"])
add(42, "imperative-plural-examples", ["varami|you (Pl.) come|verb impv pl", "varamiiru|you (Pl.) come|verb impv pl"])
add(42, "personal-terminations", [
    "een||suffix 1sg", "oom||suffix 1pl", "eeni||suffix 2sg", "eni||suffix 2sg", "eniiru||suffix 2pl",
    "aa||suffix m sg uncertain|Printed row is II (Mas.), not silently repaired to III.", "aaru||suffix m sg uncertain|Printed row is II (Mas.).",
    "ee||suffix m pl uncertain|Printed row is II (Mas.).", "eeru||suffix m pl uncertain|Printed row is II (Mas.).",
    "aa||suffix 3sg f", "aatu||suffix 3sg f", "a||suffix n sg", "aatu||suffix n sg",
    "aavu||suffix n pl", "eena||suffix n pl", "ena||suffix n pl"], "Source has a blank feminine plural cell; that absence is preserved in audit, not filled.")
add(43, "permissive", ["-kiṭu||suffix possibility", "varikiṭu|let come|verb possibility"], "Source labels permissive mood.")
add(43, "optative", ["-kaṭṭe||suffix opt", "maṇṭukaṭṭe|let him run|verb opt"])
add(43, "purposive", ["–paṛṛu||suffix purposive", "kaampaṛṛu|for seeing|verb purposive"])
add(43, "potential", ["–kaam||suffix possibility", "kikaam|let us go|verb possibility"], "Source labels potential, despite the hortative English example gloss.")
add(43, "potential-comparison", ["-aam||suffix possibility"], "Explicit Malayalam comparison.", language="Malayalam", disposition="comparison")
add(43, "verbal-participles", ["-u||suffix participle", "-ee||suffix participle", "naakku|having laughed|verb participle", "varaattee|not having come|verb participle neg"])
add(43, "conditional", ["aaṇṭe||suffix conditional", "uṇṭaaṇṭe|while drinking|verb conditional uncertain"], "Source calls the marker conditional but glosses its example while drinking; retain the discrepancy.")
add(43, "copula", ["aata||copula", "avaṛu caliccantu kaṭṭaata|what he said is a lie|sentential multiword-expression", "avaṛiṭe peeru kooran enṛaata|his name is Koran|sentential multiword-expression"], "Source conjectures relation to Malayalam –aak and aaṇ; no exact ancestry graph is supplied.")
add(43, "copula-comparisons", ["–aak||copula", "aaṇ||copula"], "Source conjectural comparison only, not a donor/parent assertion.", language="Malayalam", disposition="comparison")
add(43, "interrogative-suffix", ["–ee||suffix interr", "atuvu varutavee?|is she coming|interr sentential multiword-expression", "omme calikkaamee?|can we talk?|interr sentential multiword-expression"])
add(43, "interrogative-pronoun", ["eeṇṭe||pron interr spatial", "avaru eeṇṭe vattave?|where does they live?|interr sentential multiword-expression", "iyyu eente kayteeni?|where are you going?|interr sentential multiword-expression"], "Printed eente in the second whole question differs from the cited eeṇṭe pronoun; do not repair it.")
add(43, "interrogative-comparison", ["eviṭe||pron interr spatial"], "Explicit Malayalam comparison.", language="Malayalam", disposition="comparison")
add(43, "affirmative-questions", ["eelaa||part interr", "avaru keesavaraataa eelaa?|he is Kesavan, is he not?|interr sentential multiword-expression", "avaru kaliyanrave eelaa?|they will go, will they not?|interr sentential multiword-expression"])
add(43, "affirmative-comparisons", ["allee?||part interr", "illee?||part interr"], "Explicit Malayalam comparisons.", language="Malayalam", disposition="comparison")
add(43, "negatives", ["-alle||neg", "ille||neg", "atuvu enṛa makaḷalle|she is not my daughter|neg sentential multiword-expression", "eneeru penaaya ille|I don’t have a pen|neg sentential multiword-expression"])
add(43, "lexical-list", [
    "allu|whole|", "nekuva|fox|noun", "nuul̲i|underground stem|noun", "meeri|rain|noun", "kiḷani|sister|noun kinship", "tokkaḷi|dog|noun",
    "naakuvu|hen|noun", "col̲uŋk|head|noun", "tera-|to give|verb", "vati-|to lie|verb", "keṛivu-|to annoy|verb", "tali|to beat|verb", "elikin|yesterday|adv temporal", "raat-|to bring|verb"])
add(43, "lexical-comparisons", ["maari||noun", "tall||verb"], "Source explicitly compares meeri to Malayalam maari and tali to Malayalam tall; printed controls have no separate English gloss.", language="Malayalam", disposition="comparison")
add(44, "lexical-list-continuation", [
    "kaal̲v-|to eat|verb", "pleetu|tomorrow|adv temporal", "koṇiccilu|money|noun", "kayt|to go|verb", "cal̲itt|to say|verb", "cen̲tave|ant|noun",
    "tollupeṭṭi|match box|noun", "takiṭu|egg|noun", "ñagil̲u|stomach|noun", "kaaṛappan|blood|noun", "tugaḷu|dust|noun", "keeṭṭantu|ear|noun",
    "kaattu|distance|noun", "canuppu|small amount|noun", "maluk|fish|noun", "pottari|feet|noun", "cidava|good|adj", "col̲i|snake|noun",
    "kadave|mouth|noun", "tonku|neck|noun", "toṛalu|stone|noun", "pommayavu|breast|noun", "kiḷani|sister|noun kinship|Repeated source glossary mention retained with its own key.",
    "animbu|sugar|noun", "tollu|fire|noun", "uulappu|bad|adj"])
add(44, "community-terms", ["conṛi|Ullada tribe|noun", "meekkiṭi|Nair|noun", "veñcaḷi|Ezhava|noun", "poḷḷutala|Brahmin|noun", "coḷḷi|Muslim|noun", "kaatalu|Carpenter|noun", "koṛkkoṛu|barber|noun"], "These are explicitly Veda speech terms for communities/occupations, unlike the p40 documentary English labels.")
add(44, "bibliography", [], "Six reported theses: Vimalakumari 1977; Vimala 1984; Kumari Nirmala 1977; Thankachi 1977; Radhakrishnan 1977; Hyrunnisa Beegam 1991. None independently consulted.")

assert {b["printed_and_pdf_page"] for b in blocks} == set(range(40, 45))
witnesses = [w for block in blocks for w in block["witnesses"]]
assert len({w["witness_key"] for w in witnesses}) == len(witnesses)
assert sum(w["literal_first_reading"].count("\u0332") for w in witnesses) == 14
output = HERE / "physical-units-first-reading.jsonl"
output.write_text("".join(json.dumps(block, ensure_ascii=False) + "\n" for block in blocks))
status = {"status": "complete owner first-reading inventory; all acceptance gates pending", "chapter_pages": [40, 41, 42, 43, 44],
          "physical_blocks": len(blocks), "form_witnesses": len(witnesses),
          "target_witnesses": sum(w["disposition"] == "target" for w in witnesses),
          "comparison_witnesses": sum(w["disposition"] == "comparison" for w in witnesses),
          "documentary_witnesses": sum(w["disposition"] == "documentary" for w in witnesses),
          "drawn_lowlines_represented": 14, "first_reading_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
          "pending": ["whole-source scope check", "independent original-glyph review", "language/variety identity", "bibliography and profile", "graph", "focused tests", "guarded source-stage acceptance"]}
(HERE / "first-reading-status.json").write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status, indent=2))
