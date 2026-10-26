"""Complete physical first reading of Nair 2013 Betta Kurumba pp. 12–18.

Each record is a printed form mention, paradigm cell, blank, or documentary
claim. ``form`` is the left/only string as printed; ``surface`` is populated
only when an arrow separately prints a right-hand realization. This is not
an IPA conversion or a claim that both sides are independent lexemes.
"""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[5] / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
PDF_SHA = "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == PDF_SHA
rows = []


def add(p, section, form="", gloss="", surface="", role="target", claim="", note="", label=""):
    num = 1 + sum(r["printed_page"] == p and r["section"] == section for r in rows)
    key = f"nair2013bettakurumba:p{p}:{section}:{num:02d}"
    rows.append({"unit_key": key, "printed_page": p, "pdf_page": p,
                 "section": section, "section_index": num, "source_order": len(rows) + 1,
                 "form": form, "surface": surface, "gloss": gloss, "role": role,
                 "source_claim": claim, "note": note, "grammatical_label": label,
                 "source_pdf_sha256": PDF_SHA})


def forms(p, section, entries, *, claim="", label=""):
    for item in entries:
        add(p, section, *item, claim=claim, label=label)


# p. 12: classification/ethnography and ten separately printed diagnostics.
add(12, "community-context", claim="Alu/Palu/Halu, Betta, Jenu/Ten, Urali and Mullu Kurumba are named communities, not separately printed Betta Kurumba speech forms.", role="documentary")
add(12, "community-context", claim="Kurumar/Kurumba/Kuruba and the Pallava-origin account are ethnographic claims, not assigned language lexemes.", role="documentary")
add(12, "community-context", "Bettu", "people of the hills", role="attribution-held",
    claim="Quoted as meaning 'people of the hills' in 'their language' after discussion of the heterogeneous Kurumba group; the specific lect is not unambiguous.")
add(12, "community-context", claim="Kargudi, Theppakkadu, Gudalur, Mayar, Devar Solai, Pandalur and Mudumalai are community concentration areas, not source elicitation coordinates.", role="documentary")
add(12, "classification", claim="Nair reports historical Kannada-dialect classifications and Emeneau, Zvelebil, Upadhyaya and Jayapal's independent-language analyses; no comparator forms.", role="documentary")
for f,g in [("-ma","accusative"),("-ooda","instrumental"),("-beliya","sociative"),
            ("-ka","dative"),("-lito","ablative"),("-pilito","ablative"),
            ("-puli","locative"),("-inda","causative case"),("-kabeyri","purposive")]:
    add(12,"diagnostic-case-markers",f,g,claim="Jayapal diagnostic list as reported by Nair; source grammar class only.",label=g)
add(12,"diagnostic-plural","-g","plural",claim="Reported Jayapal diagnostic.",label="plural")
add(12,"diagnostic-nine","enbadu","nine",claim="Reported numeral innovation; repeated in the p. 14 numeral paragraph.",label="cardinal")
add(12,"diagnostic-gender",claim="Source reports absence of third-person gender contrast in pronouns and verbal person suffixes.",role="documentary")
add(12,"diagnostic-link","-p-","link morph",claim="Reported Jayapal diagnostic; interior hyphens mark its boundary, not a phonemic claim.",label="bound link morph")
add(12,"diagnostic-nonpast","-ø","non-past",role="zero-exponent",claim="Printed zero exponent, not a pronounceable lexical form.")
add(12,"diagnostic-negative",claim="Source reports distinctive negative morphology without a form in this list; examples occur on p. 16.",role="documentary")
add(12,"diagnostic-infinitive",claim="Source reports infinitive suffixes without listing them here; later pp. 15–16 provide forms.",role="documentary")
for f,g in [("-u","relative participle"),("-un","verbal participle"),("-n","verbal participle"),("-di","verbal participle")]:
    add(12,"diagnostic-participles",f,g,claim="Reported Jayapal diagnostic; explicit suffix form.",label=g)
add(12,"diagnostic-personal-suffixes",claim="Different past/non-past personal suffix sets are reported, with full table on p. 18.",role="documentary")
add(12,"classification",claim="Jayapal's Betta Kurumba–Kodagu subgroup proposal is attributed scholarship, not an attested cognate form.",role="documentary")

# p. 13: nouns, case inventory and a complete sociative sentence.
for f,g,gender in [("aran","a man of Brahmin caste","m"),("ajji","grandmother","f"),
                   ("taṭn","goldsmith","m"),("taṭiti","goldsmith woman","f")]:
    add(13,"gender-nouns",f,g,claim="Printed masculine/feminine noun table; third-person pronouns/verbs are said to lack gender contrast.",label=gender)
for f in ("-r","-g"):
    add(13,"plural-markers",f,"plural",claim="Source says singular unmarked and plural uses -r or -g.",label="plural suffix")
forms(13,"plural-examples",[("abbi-r","mothers","abbiru"),("aaḷ-g","persons","algu"),("piliye-g","tigers","piligu")],
      claim="Left segmented form and arrowed surface are both printed; right-hand form is the lexical surface.",label="plural")
add(13,"case-context",claim="Source states nominative is unmarked and there are ten cases; zero nominative is not a pronounceable form.",role="zero-exponent")
for f,claim in [("-e","Common accusative, compared with other South Dravidian languages."),
                ("-ma","Betta Kurumba accusative after noun stems ending -iye; described as peculiar to this language.")]:
    add(13,"accusative-markers",f,"accusative",claim=claim,label="accusative suffix")
forms(13,"accusative-examples",[("piliya-ma","tiger (Acc.)"),("meriye- ma","young one of animal (Acc.)")],label="noun accusative")
add(13,"instrumental-marker","ooḍa","instrumental",claim="Source calls this an instrumental suffix; the same marker can also express sociative relation.",label="instrumental")
forms(13,"instrumental-examples",[("bumman-ooḍa","by Bomman"),("beriye-ooḍa","by the stick")],label="instrumental")
add(13,"sociative-sentence","naavu adamooḍa niiraaḍni","I took bath with him",claim="Complete printed sentence illustrating sociative use of the instrumental marker.",label="sociative")
for f,claim in [("-na","Dative following third-person singular demonstratives, place nouns and personal names."),
                ("-ka","Dative in all other places.")]:
    add(13,"dative-markers",f,"dative",claim=claim,label="dative suffix")
forms(13,"dative-examples",[("ad-na","him/her/it (Dat.)","aduna"),("uuṭi-na","Ooty (Dat.)"),
                             ("kiri-ka","house (Dat.)","kirka"),("agar-ka","them (Dat.)")],
      claim="The arrowed surface is separately printed only where a right-hand form is supplied.",label="dative")
for f,claim in [("-pilito","Ablative after animate noun stems and personal pronouns."),
                ("-lito","Ablative after all other stems.")]:
    add(13,"ablative-markers",f,"ablative",claim=claim,label="ablative suffix")
forms(13,"ablative-examples",[("bumma-pilito","from Bomman"),("adani-pilito","from him/her/it"),
                               ("mert-lito","from the tree"),("perikaadt-lito","from the forest")],label="ablative")
add(13,"genitive-context",claim="Genitive case is said to have no suffix; two complete possessive expressions follow on p. 14.",role="zero-exponent")

# p. 14: genitive through interrogative paradigms, with comparison controls.
forms(14,"genitive-examples",[("adan kiiri","his/her house"),("pikki kiiriye","bird’s nest")],label="genitive phrase")
add(14,"locative-markers","-l","locative",claim="Source calls it a reflex of a South Dravidian -il; no specific comparison lexeme is supplied.",label="locative suffix")
add(14,"locative-markers","-pili","locative",claim="Source says this suffix lacks a reflex in other Dravidian languages and follows animate nouns/pronouns.",label="locative suffix")
forms(14,"locative-l-examples",[("tooṭṭ-l","in the garden"),("laarit-l","in the lorry")],label="locative")
forms(14,"locative-pili-examples",[("enni-pili","with me/in me"),("bumman –pili","with Bomman/in Bomman")],
      claim="Source's combined with/in gloss is preserved without forcing one case sense.",label="locative/sociative")
add(14,"causative-case-marker","-inda","causative case",claim="Source calls this a causative case suffix; examples mean 'because of'.",label="causative case")
forms(14,"causative-case-examples",[("en-inda","because of me"),("akan –inda","because of elder sister")],label="causative case")
add(14,"purposive-marker","-kabeyri","purposive",claim="Source calls this a peculiar purposive case suffix not attested in other speech.",label="purposive suffix")
forms(14,"purposive-examples",[("mari-kabeyri","for the sake of Mari"),("yaani-kabeyri","for the sake of elephant")],label="purposive")
add(14,"numeral-nine","enbadu","nine",claim="Source contrasts this Betta Kurumba form with onpatu of other South Dravidian languages.",label="cardinal")
add(14,"numeral-nine","onpatu","nine",role="comparison-control",claim="Generic other-South-Dravidian comparator, not a Betta Kurumba attestation.")
for f,g in [("on-","one"),("ond-","one"),("od","two"),("ord","two")]:
    add(14,"numeral-adjectives",f,g,claim="Source lists numeral-adjective alternatives without assigning separate meanings to each.",label="numeral adjective")
add(14,"pronoun-context",claim="Third-person gender-neutral claim is compared with Toda and Brahui without printed comparator forms.",role="comparison-context")
for f,g,lab in [("navu","I","1sg"),("naŋ","we","1pl inclusive"),("en","we","1pl exclusive"),
                ("e-","I oblique base","1sg oblique"),("ni-","you","2sg"),("niŋ-","you","2pl"),
                ("ad","he/she/it","3sg distal"),("agu","they","3pl distal"),
                ("id","he/she/it","3sg proximal"),("igu","they","3pl proximal")]:
    add(14,"personal-pronouns",f,g,claim="Printed singular/plural/person table; the source explicitly labels inclusion/exclusion and proximal/distal where shown.",label=lab)
for f in ("da-","e-"):
    add(14,"interrogative-bases",f,"interrogative base",claim="Source names two bases for interrogative pronouns; do not infer a separate English lexical sense.",label="interrogative base")
forms(14,"interrogative-pronouns",[("daru","who"),("edu","which")],label="interrogative pronoun")

# p. 15: indefinite derivations, verbal suffix classes, transitivity and infinitives.
add(15,"indefinite-marker","-aŋg","indefinite",claim="Source says this suffix attaches to interrogative pronouns.",label="indefinite suffix")
forms(15,"indefinite-examples",[("ed-aŋg","something","ed-aŋgu"),("dar-aŋg","somebody","da-raŋgu")],
      claim="Source prints both segmented input and arrowed surface; not separate meaning variants.",label="indefinite pronoun")
add(15,"tense-context",claim="Source reports a two-tense past/non-past system, not literary Dravidian three-tense analysis.",role="documentary")
for f in ("-t","-d","-nd-","-n-","-c","-i"):
    add(15,"past-markers",f,"past",claim="One of six source-listed past-tense suffixes; each is said to select a verb class.",label="past suffix")
forms(15,"past-examples",[("ede-t-i","took-I"),("ade-t-i","closed-I"),("erka-d-i","swept-I"),
                           ("ta-nd-o","gave-they"),("ett-n-i","tapped-I","etni"),
                           ("ank-n-i","dried-I","ankuni"),("ett-c-a","(one) who tapped","ettica")],
      label="past paradigm")
for f in ("-p","-u"):
    add(15,"nonpast-markers",f,"non-past",claim="Source explicitly lists this non-past suffix.",label="non-past suffix")
add(15,"nonpast-markers","ø","non-past",role="zero-exponent",claim="Source explicitly reports suffixless non-past forms; ø is zero notation, not a lexical word.")
forms(15,"nonpast-examples",[("eḍe-p-iya","will take-I/we"),("eḍe-p-iyo","will take-you (Pl.)"),
                              ("aḍ-u-a","will cook he/she","aduva"),("kiis-u-a","will sharpen he/she","kiisuva"),
                              ("kenga-ø-iyi","will kill-you(sg.)","kengavi(yi)"),("nili-ø-u","one who will play","niliyu")],
      claim="Segmentation and source optional (yi) notation are literal; no unprinted alternate is supplied.",label="non-past paradigm")
add(15,"transitivity-context",claim="Source separates intransitives made transitive by suffix from a second group said only to be causativizable.",role="documentary")
for f in ("-k","-t","-x","-p-"):
    add(15,"transitive-markers",f,"transitivizer",claim="Source lists suffixes deriving transitive forms from intransitive forms.",label="transitive suffix")
forms(15,"transitive-examples",[("amg-k-","press","am-k"),("kal ŋg-k","stir","kal-k"),
                                 ("ad-t","shake","at"),("od-t","drive","ot"),
                                 ("oli-x","hide"),("kay-x","boil"),("a-p","cause to wake up"),
                                 ("uli-p","cause to get down","ulp")],label="transitivized example")
for f,g in (("-l","purposive infinitive"),("-laye","purposive infinitive"),("-ka","effective/simultaneous infinitive")):
    add(15,"infinitive-markers",f,g,claim="Source describes function explicitly; complete examples continue on p. 16.",label="infinitive suffix")
add(15,"infinitive-markers","ø","purposive infinitive",role="zero-exponent",claim="Printed Ø is zero suffix notation, not an independent word.")

# p. 16: infinitives, six negative constructions, and relative participles.
forms(16,"infinitive-examples",[("belki-l-beda","(one) need not speak"),("belki-l-appa","(one) can speak"),
                                 ("od-ø-ku","may run","oduku"),("bar-ø-ku","may come","baruku")],label="infinitive construction")
add(16,"infinitive-examples","","for (the purpose of) speaking",role="blank",
    claim="Printed gloss has no corresponding visible form; do not fill from the parallel example.")
forms(16,"infinitive-examples",[("kel-laye","for (the purpose of) reading/learning"),
                                 ("ar-ka","to climb"),("ad-ka","to dance","aduka")],label="infinitive construction")
for f,g in (("-lpad-","durative/simple negative"),("-a","relative participle after -lpad-")):
    add(16,"durative-negative-markers",f,g,claim="Source says -lpad- supports simple and durative negation; -a follows in durative relative participles.",label="negative morphology")
for f,g in (("baar-lpaḍ-a","(one) who does not come; (one) who never comes"),
            ("tin-lpaḍ-a","(one) who does not eat; (one) who never eats")):
    add(16,"durative-negative-examples",f,g,claim="Two printed English descriptions share this one target form; semicolon only represents gloss scope here.",label="durative/simple negative")
for f,g in (("ldid","past negative"),("-il","past negative"),("-ani","conditional suffix context"),("-di","verbal participle context")):
    add(16,"past-negative-markers",f,g,claim="Source reports host environment for past-negative markers; do not collapse -ani/-di into the negative itself.",label="negative morphology")
forms(16,"past-negative-examples",[("od-ldid-a","(one) who did not run"),("bar-ldid-a","(one) who did not come"),
                                      ("od-il-di","without running"),("bar-il-di","without coming")],label="past negative")
add(16,"nonpast-negative-marker","-lk","non-past negative",claim="Source explicitly gives one non-past negative suffix.",label="negative suffix")
forms(16,"nonpast-negative-examples",[("kenga-lk-an","if (one) will not kill"),("noḍ-lk-an","if (one) will not see")],label="non-past negative conditional")
add(16,"doubt-negative-marker","-lidapad","negative expressing doubt",claim="Source explicitly labels this negative suffix as expressing doubt.",label="negative suffix")
forms(16,"doubt-negative-examples",[("noḍ-lidapad-o","they might not have seen"),
                                     ("tin-lidapad-I","you might not have eaten")],label="negative uncertainty")
add(16,"general-negative-marker","-l","negative",claim="Source says this suffix precedes pronominal suffixes in all other stems.",label="negative suffix")
forms(16,"general-negative-examples",[("ba-nd-la-a","did not come-he/she/it"),("odici-l-u","did not run-you (Pl.)")],label="negative finite")
for f in ("-a","-u"):
    add(16,"relative-participle-markers",f,"relative participle",claim="Source contrasts common -a with rare -u; this is a source frequency statement, not a cross-lexeme variant.",label="relative participle suffix")
forms(16,"relative-participle-examples",[("ba-nd-a","(one) who came"),
                                           ("baar-lpad-a","(one) who does not come / (one) who never comes"),
                                           ("tin-p-u","(one) who will eat","timbu")],label="relative participle")

# p. 17: verbal participles, modal and aspectual auxiliaries.
for f,g in (("-di","verbal participle after negative -il or defective -il"),
            ("-n","verbal participle after past -I"),("-un","verbal participle elsewhere"),
            ("-il","defective verb/negative context")):
    add(17,"verbal-participle-markers",f,g,claim="Source explicitly states the conditioning; its -un claim conflicts with printed ba-nd-um example.",label="verbal participle morphology")
forms(17,"verbal-participle-examples",[("il-di","without being"),("tin-il-di","without eating"),
                                         ("ooḍ-i-n","having run","oydn"),("ba-nd-um","having come"),
                                         ("neg-dun","having jumped")],
      claim="Printed ba-nd-um differs from prose suffix -un; preserve both literally.",label="verbal participle")
add(17,"modal-ku","-ku","possibility/probability/permissive",claim="Source states this auxiliary has several modal readings, not one-to-one glosses.",label="modal auxiliary")
forms(17,"modal-ku-examples",[("ooḍ-ku","(one) may run","ooḍuku"),("nooḍ-ku","(one) may see","nooḍuku")],label="modal possibility")
add(17,"potential-aappa","-aappa","potential",claim="Source says this auxiliary attaches to infinitive verb forms.",label="potential auxiliary")
forms(17,"potential-examples",[("ooḍl-aappa","(one) can run"),("turkul-aappa","(one) can twist")],label="potential")
add(17,"negative-potential","aagalillaa","negative potential",claim="Source compares the form with Kannada but gives no separate Kannada witness.",label="negative potential auxiliary")
forms(17,"negative-potential-examples",[("belkil-aagalillaa","(one) cannot speak"),
                                              ("olpicil-aagalillaa","(one) cannot cause (someone) to hide")],label="negative potential")
add(17,"prohibitive-marker","-beeḍa","prohibitive",claim="Source compares its function with Kannada, without printing a Kannada comparator form.",label="prohibitive auxiliary")
add(17,"prohibitive-example","ooḍl-beeḍa","(one) should not/need not run",label="prohibitive")
add(17,"attempted-marker","-nooḍ","attempted action",claim="Source says this auxiliary attaches to infinitive verb forms.",label="attempted-action auxiliary")
forms(17,"attempted-examples",[("belki-nooḍ-n-I","I tried to speak","belkil-nooḍni"),
                                    ("belki-nooḍ iya","I will try to speak")],label="attempted action")
for f,g in (("uḍ","leave"),("tii","complete")):
    add(17,"definitive-auxiliaries",f,g,claim="Source says their past forms create past definitive and the same auxiliaries occur in non-past definitive.",label="definitive auxiliary")
forms(17,"past-definitive-examples",[("tardun-uḍ-t-l","I have opened","traduṭṭi"),
                                        ("aaṭicin-uḍ-ta","he/she/it had shaken","aaṭicinuṭṭa")],label="past definitive")
add(17,"nonpast-definitive-example","tardun-uḍ-iyaa","I/we will open","tardunuḍiya",label="non-past definitive")
add(17,"completive-marker","ay","to become",claim="Source says this auxiliary follows verbal participle forms in past-tense completive aspect.",label="completive auxiliary")

# p. 18: completion, durativity/reflexivity, complete person-suffix matrix.
forms(18,"completive-examples",[("bandun-aay-taa","(one) has come","bandunaacaa"),
                                 ("ooyḍn-aay-taa","(one) has run","ooyḍṇaacaa")],label="completive")
for f,g in (("-ir","to be; past/future durative"),("ooḍ","to be; present durative")):
    add(18,"durative-auxiliaries",f,g,claim="Source contrasts past/future -ir with present ooḍ.",label="durative auxiliary")
forms(18,"durative-examples",[("bandun-ir-d-i","was coming-I","banduniddi"),
                                ("tindun-ooḍ-a","(is) eating-he/she/it"),
                                ("bandum-ir-p-iya","will be coming-I/we")],label="durative")
add(18,"reflexive-base","tan","reflexive oblique base",claim="Source says tan is inserted between a verbal participle and ir/ooḍ.",label="reflexive")
forms(18,"reflexive-examples",[("ardun-tan-ir-daa","he was beating himself","arduntanidda"),
                                 ("ardun-tan-oo-ḍ-a","he is beating himself")],label="reflexive aspect")
add(18,"person-suffix-context",claim="Source says two person-suffix series follow past vs non-past forms and third person lacks gender distinction.",role="documentary")
for person,tense,number,raw in [
    ("1","past","sg","i/ee"),("1","past","pl","aa"),("1","nonpast","sg","iyaa"),("1","nonpast","pl","iyaa"),
    ("2","past","sg","i"),("2","past","pl","u/aa"),("2","nonpast","sg","iyi"),("2","nonpast","pl","iyo/iyu"),
    ("3","past","sg","aa"),("3","past","pl","oo"),("3","nonpast","sg","aa"),("3","nonpast","pl","iyoo")]:
    add(18,"person-suffix-table",raw,"",claim="One exact printed table cell; slash alternatives remain in the same physical cell at first reading.",
        label=f"{person}{number} {tense} personal suffix")
forms(18,"negative-person-examples",[("bandli","did not come-I"),("bandla","did not come he/she/it")],label="negative person suffix")
add(18,"bibliography",claim="Reports Jayapal, S., Descriptive Grammar of Kurumba, unpublished PhD thesis, Annamalai University, 1978; reported authority only.",role="bibliographic-context")

# The PDF draws some true low lines as short vector rectangles beneath letters.
# They are absent from its selectable text. Preserve the visible typography,
# with a source-keyed exact-old guard and no phonological interpretation.
lowlines = json.loads((HERE / "lowline-corrections.json").read_text())
by_key = {r["unit_key"]: r for r in rows}
assert len(lowlines["changes"]) == 15
for change in lowlines["changes"]:
    row = by_key[change["unit_key"]]
    assert row[change["field"]] == change["old"], change
    row[change["field"]] = change["new"]
    row["note"] += " Printed low line on selected letters is retained as U+0332; its phonetic value is not inferred."
notation = json.loads((HERE / "dash-notation-corrections.json").read_text())
for change in notation["changes"]:
    row = by_key["nair2013bettakurumba:" + change["key"]]
    assert row["form"] == change["old"], change
    row["form"] = change["new"]

assert {r["printed_page"] for r in rows} == set(range(12, 19))
assert len({r["unit_key"] for r in rows}) == len(rows)
assert sum(r["section"] == "person-suffix-table" for r in rows) == 12
assert sum(r["role"] == "blank" for r in rows) == 1
assert sum(r["role"] == "zero-exponent" for r in rows) == 5
(HERE / "physical-units-first-reading.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
status = {"status": "complete-chapter-first-reading-pending-independent-review",
          "source_pdf_sha256": PDF_SHA, "chapter_pages": list(range(12, 19)),
          "physical_units": len(rows),
          "page_counts": {str(p): sum(r["printed_page"] == p for r in rows) for p in range(12, 19)},
          "role_counts": dict(collections.Counter(r["role"] for r in rows)),
          "arrowed_surface_units": sum(bool(r["surface"]) for r in rows),
          "source_stage_installation": "pending"}
(HERE / "first-reading-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(status, ensure_ascii=False))
