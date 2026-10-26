"""Keyed grammar decisions from explicitly reviewed source spans; no emission."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = HERE.parents[6]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def person(number, size):
    return [f"{number}{size}", size, {1:"first-person", 2:"second-person", 3:"third-person"}[number]]


PRONOUN_FEATURES = {
    "I":person(1,"sg"), "me":person(1,"sg")+["obj"], "my":person(1,"sg")+["poss"],
    "we":person(1,"pl"), "our":person(1,"pl")+["poss"],
    "thou":person(2,"sg"), "thee":person(2,"sg")+["obj"], "thy":person(2,"sg")+["poss"],
    "you":person(2,"pl"), "your":person(2,"pl")+["poss"],
    "he":person(3,"sg")+["m"], "him":person(3,"sg")+["m","obj"],
    "his":person(3,"sg")+["m","poss"], "it":person(3,"sg")+["n"],
    "they":person(3,"pl"), "their":person(3,"pl")+["poss"]}


def decision(row):
    shorthand = row["physical_parent_key"].split(":p",1)[1]
    page, section, group = shorthand.split(":")
    group = int(group)
    rank = int(row["witness_key"].rsplit(":w",1)[1])
    form, gloss = row["form"], row["gloss"]
    tags, note, typed, kind = [], "", [], "lexical-form"
    def set_tags(values, rationale, category="lexical-form"):
        nonlocal tags, note, kind
        tags, note, kind = values.split() if isinstance(values,str) else values, rationale, category
    def morphology(values, rationale):
        set_tags(values, rationale, "source-morphological-element")
    if section == "noun-plural":
        if rank in (1,7,8,9,10,12,13):
            morphology("affix suffix pl", "Source explicitly calls this a plural suffix or its directly labelled plural comparator.")
            if rank in (12,13):
                tags += ["uncertain"]
                typed.append(dict(type="source-analysis", detail="Source says apparently ur/r; retain qualified segmentation, not a certain standalone stem."))
        else:
            set_tags("noun pl" + (" dat" if rank==11 else ""), "Nouns/plural examples explicitly introduced by the source; 'to the men' additionally supplies dative meaning.")
    elif section == "accusative-dative":
        if rank <= 2:
            morphology("affix suffix acc dat", "Source explicitly states that dative and accusative share this suffix; tags describe both source functions, not disjoint invented forms.")
        else:
            set_tags("noun acc dat" if rank==3 else "noun dat" + (" pl" if rank==5 else ""), "Source case examples; unqualified 'the son' retains the explicitly syncretic accusative/dative class rather than inventing a unique case.")
    elif section == "genitive-locative":
        if rank <= 4:
            morphology("affix suffix " + ("gen" if rank<=3 else "loc"), "Explicit source genitive/locative suffix; no lexical noun or verb POS.")
        elif rank <= 7:
            set_tags("multiword-expression gen" + (" dat" if rank==7 else ""), "Printed genitive noun phrase; dative 'to the servants' applies to the complete third phrase. Person/gender of contained possessors not transferred to the whole phrase.", "whole-noun-phrase")
        else:
            set_tags("noun loc" if rank<=9 else "noun", "Source locative examples or parenthetical field base; no undocumented morphology generated.")
    elif section == "numerals":
        set_tags("num", "Explicit numeral section, including source-labelled comparison-language numerals.")
        if row["printed_label"] == "masc.":
            tags += ["m"]
        if page == "566" and rank in (3,4):
            tags += ["f","pl"]
            if rank==4:
                tags += ["multiword-expression"]
                kind="whole-noun-phrase"
            note += " Basim source expressly says feminine plural in this numbered-two/daughters example; no gender transferred to other numerals."
    elif section == "personal-pronouns":
        set_tags(["pron","personal"] + PRONOUN_FEATURES[gloss], "Explicit personal-pronoun paradigm and printed English person/number distinction (thou versus you). Possessive/object function retained without guessing accusative versus dative.")
        if row["printed_label"] == "neut.": tags += ["n"]
        if page=="562" and gloss in ("we","you","they"):
            note += " Source qualifies plural forms as seldom used and says singulars substitute; tag denotes this printed paradigm interpretation, not exclusive surface distribution."
    elif section == "other-pronouns":
        if form in ("yēnd","yēr","tān","tāned"):
            set_tags("pron interr", "Explicit who?/what? pronoun interpretation; no inferred case/person.")
        elif form == "adnet":
            set_tags("pron demonstrative gen", "Explicit genitive of preceding neuter demonstrative 'that'; no invented lexical definition.")
            tags += ["n"]
        elif gloss in ("this","that","those things"):
            set_tags("pron demonstrative" + (" n" if row["printed_label"]=="neuter" else "") + (" pl" if gloss=="those things" else ""), "Printed demonstrative interpretation; proximal/distal subclass not asserted beyond source wording.")
        elif gloss == "its":
            set_tags(["pron","poss","n"]+person(3,"sg"), "Printed possessive 'its'; source does not separately name its case.")
        else:
            assert gloss == "his"
            set_tags(["pron","poss","m"]+person(3,"sg"), "Printed possessive 'his'; no inferred reflexivity for tānet.")
    elif section == "present-copula":
        set_tags("verb copula pres", "Complete present verb-substantive forms, not isolated affixes.")
        if page=="562":
            tags += person((1,2,3,3)[rank-1],"sg")
            if rank in (3,4): tags += ["m" if rank==3 else "n"]
            note += " Source says no real plural examples in its materials; optional parentheses remain literal, not new forms."
        elif group == 3:
            tags += ["sg","m","uncertain"]
            typed.append(dict(type="source-numbering", detail="Printed Sing.2.m. duplicates previous 2 and precedes3.n.; withhold both2sg and3sg, no silent correction."))
        else:
            number,size = {1:(1,"sg"),2:(2,"sg"),4:(3,"sg"),5:(1,"pl"),6:(2,"pl"),7:(3,"pl")}[group]
            tags += person(number,size)
            if group==4: tags += ["n"]
    elif section == "present-copula-comparison":
        set_tags(["verb","copula","pres","n"]+person(3,"sg"), "Source explicitly labels third-person singular neuter; no analogy to duplicate2.m. row.")
    elif section == "past-copula":
        if page=="562":
            set_tags(["verb","copula","pret"]+person((group-1)%3+1,"sg" if group<=3 else "pl"), "Printed past-copula table person and singular/plural headings; grammatical labels belong in Tags, lexical Gloss remains blank.")
        elif rank <= 2:
            set_tags(["verb","copula","pret"]+person(1,"sg" if rank==1 else "pl"), "Explicit past copula 'I was'/'we were'.")
        elif rank == 3:
            set_tags("uncertain", "Printed 'he was dead' is alternatively analyzed by the author as perhaps an agency noun 'one who is dead'; no confident verb/noun POS or intrinsic tense tag.")
            typed.append(dict(type="source-analysis",detail="Source offers conflicting finite-past versus agent-noun analyses; preserve both in Notes, do not infer a derivation."))
        elif rank == 4:
            set_tags("uncertain", "Author says anden seems contained for aṇḍen; quoted embedded component, not independently attested lexical stem.", "quoted-embedded-analysis-component")
            typed.append(dict(type="source-analysis",detail="Proposed contained copula versus alternative agency-noun analysis is tentative."))
        else:
            set_tags("verb copula pret", "Explicitly quoted comparison form 'was'; no undocumented person/number inherited from surrounding hypothetical segmentation.")
    elif section == "finite-verbs":
        if page == "563":
            mapping = {
                1:("pret","Source names characteristic past consonant, not a separately demonstrated suffix or lexical verb."),
                2:("pret","Source names characteristic past consonant, not a separately demonstrated suffix or lexical verb."),
                3:(["verb","pres","fut"]+person(1,"sg"),"Present explicitly used also as future; source 'I shall make'."),
                4:(["verb","pret"]+person(1,"sg"),"Explicit past example 'I did'."),
                5:(["verb","pret","m"]+person(3,"sg"),"Explicit past example 'he went'."),
                6:(["verb","copula","pres","n"]+person(3,"sg"),"Explicit third-person neuter 'it is'."),
                7:(["verb","copula","pres","n"]+person(3,"sg"),"Explicit third-person neuter 'it is'."),
                8:(["verb","pret","n"]+person(3,"sg"),"Explicit third-person neuter 'it became'."),
                9:("verb pret third-person n multiword-expression","Source explicitly places swine-ate phrase under third-person neuter; no invented3pl or singular inflection because source discusses plural/singular substitution."),
                10:("affix fut uncertain","Source says another future seems to add d; tentative morphological element, not lexical verb or proven suffix location."),
                11:(["verb","fut"]+person(1,"sg"),"Explicit future 'I shall strike'."),
                12:(["verb","fut","multiword-expression"]+person(3,"pl"),"Explicit future whole subject-verb expression 'they will strike'.")}
            set_tags(*mapping[rank])
            if rank in (1,2,10): kind="source-morphological-element"
            if rank==10: typed.append(dict(type="source-analysis",detail="Source's seems-to-add-d qualification retained."))
        else:
            if rank<=2:
                morphology("affix suffix pres participle" if rank==1 else "affix suffix pret", "Printed present participle ending s or past form ending t; not lexical verb stems, no automatic past-participle claim.")
            else:
                set_tags("verb", "Printed verb 'to beat'; English infinitival citation wording does not prove a target infinitive form.")
    elif section == "imperative":
        if rank == 1:
            morphology("affix suffix pl impv", "Source explicitly adds r in plural imperative; singular no-suffix statement is documentary, no invented zero form.")
        else:
            set_tags("verb impv" + (" 2pl pl second-person" if rank==4 else ""), "Explicit imperative example; only put-ye has source-explicit plural person. Bare give/eat do not acquire guessed2sg.")
    elif section == "negative":
        if page=="563":
            if rank in (1,4):
                morphology("affix neg", "Source adds tōten to base and sometimes inflects it; no lexical verb POS or unsupported suffix category imposed.")
            elif rank==6:
                morphology("affix neg", "Explicit negative e added to base; no standalone lexical stem.")
            else:
                set_tags("verb neg pret", "Printed negative finite example and explicit past English translation; source person scopes kept separately.")
                if rank==2: tags+=person(1,"sg")
                elif rank==5: tags+=person(2,"sg")
                elif rank in (7,8): tags+=person(3,"sg")+["m"]
                elif rank==3:
                    tags += ["sg"]
                    typed.append(dict(type="source-interpretation-scope",detail="One form given with thou/he alternatives; retain shared singular, withhold conflicting person and gender."))
        else:
            if rank==1:
                set_tags("part neg negator", "Independent not added to the infinitive in the printed analysis; not the infinitive itself and no invented suffix attachment.")
            else:
                set_tags("verb neg pret", "Printed negative finite example; source says formed without tense suffixes, so pret denotes its explicit translated past use, not a claimed tense marker.")
                if rank==2: tags+=person(1,"sg")
                elif rank==3: tags+=person(3,"sg")+["m"]
                elif rank==4: tags+=person(3,"sg")+["n"]
                elif rank==5:
                    tags += ["sg"]
                    typed.append(dict(type="source-interpretation-scope",detail="Same negative form receives thou/he alternatives; no forced unique person or gender."))
    elif section == "verbal-nouns":
        if rank in (1,2):
            set_tags("noun uncertain", "Source says base seems used as verbal noun; preserve tentative nominal analysis, no inf/participle inferred from English.")
            typed.append(dict(type="source-analysis",detail="Source verbal-noun interpretation is explicitly qualified by seems."))
        elif rank==3:
            morphology("affix suffix n nominalizer", "Source explicitly names verbal-noun ending containing neuter demonstrative; tags classify ending, not independent pronoun.")
        else:
            set_tags("noun", "Explicit verbal noun dancing; not automatically a finite verb or participle.")
    elif section == "conjunctive-participle":
        if (page=="563" and rank==1) or (page=="567" and rank<=2):
            morphology("affix suffix participle conjunctive-participle", "Source explicitly names conjunctive-participle ending; no lexical verb POS on isolated suffix.")
        else:
            set_tags("participle conjunctive-participle", "Explicit complete conjunctive participle example; no finite TAM/person/number inferred from having-translation.")
    elif section.startswith("pronunciation-"):
        symbolic = (section=="pronunciation-a-u" and rank<=2) or (section=="pronunciation-l-n" and rank in (1,2,5,6)) or (section=="pronunciation-n-nd-d" and rank in (1,2,5,6))
        if symbolic:
            set_tags([], "Quoted letter/sound correspondence symbol is documentary, not lexical/morphological stem or automatic normalization rule.", "documentary-symbol")
        elif form=="avnd":
            set_tags([], "Unstarred but explicitly hypothetical source of amd; documentary proposed derivation, not an attested pronoun or generated ancestor.", "documentary-hypothesis")
        elif gloss in ("I die","I say"):
            set_tags(["verb"]+person(1,"sg"), "Pronunciation example explicitly says I die/I say; no independent tense heading, so present English does not create a confident target pres tag.")
        elif gloss=="he sent":
            set_tags(["verb","pret","m"]+person(3,"sg"), "Source pronunciation examples explicitly gloss he sent; correspondence does not imply shared variant/ancestor edges.")
        elif gloss=="he said to his father":
            set_tags(["verb","pret","multiword-expression","m"]+person(3,"sg"), "Printed whole pronunciation expression; grammatical person/tense from explicit he-said, no father case guessed.")
        elif gloss=="he":
            set_tags(["pron","personal","m"]+person(3,"sg"), "Explicitly glossed he in pronunciation examples; no morpheme segmentation or cross-source harmonization.")
        elif gloss=="to him":
            set_tags(["pron","personal","dat","m"]+person(3,"sg"), "Printed to-him forms; dat describes this source use, no invented variants/edges.")
        else:
            set_tags("pron", "Quoted pronoun comparison within the avan/he argument; no independent gloss/person/gender created for unglossed comparators.")
    elif section=="nouns":
        if rank in (1,2,5,6):
            morphology("affix suffix "+("pl" if rank<=2 else "dat"), "Explicit source plural/dative suffix; no lexical noun or verb POS.")
        else:
            set_tags("noun "+("pl" if rank<=4 else "dat"), "Printed plural/dative noun example; no assumed donor analysis.")
    elif section=="possessive-agreement":
        if rank<=2:
            set_tags(["pron","poss"]+person(rank,"sg"), "Printed possessive my/thy contains neuter demonstrative; do not transfer embedded n gender to whole possessive, which source says can occur in all connections.")
        elif rank==3:
            set_tags("pron demonstrative n", "Explicit isolated neuter demonstrative ad, that; no morphological edge guessed.")
        else:
            set_tags("multiword-expression poss", "Source possessive agreement noun phrase; no possessor person/gender transferred to the whole thy-son expression.", "whole-noun-phrase")
    elif section=="beat-paradigm":
        tense = ("pres","pret","fut")[(group-1)//6]
        cell = (group-1)%6
        set_tags(["verb",tense]+person(cell%3+1,"sg" if cell<3 else "pl"), "Complete printed beat finite paradigm, scoped by tense and singular/plural column plus numbered cell; pure analysis labels in Tags, Gloss remains blank.")
    elif section=="present-suffix-comparison":
        morphology("affix suffix pres", "Explicit present s-suffix and named Telugu tsu/tu comparisons; no lexical verb POS or unprinted words.")
    else:
        raise AssertionError((row["witness_key"],section))
    return list(dict.fromkeys(tags)), note, typed, kind


def main():
    path=PACKAGE/"grammar-review/reviewed-child-witnesses.jsonl"
    inputs=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(inputs)==224 and len({r["witness_key"] for r in inputs})==224
    tree=ast.parse((ROOT/"data/tags.py").read_text())
    allowed=set()
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in ("GRAMMATICAL_TAGS","GENDER_TAGS") for target in node.targets):
            allowed |= ast.literal_eval(node.value)
    proposals=[]
    for row in inputs:
        tags,note,typed,kind=decision(row)
        # Overlay metadata may add tags, but never clears the prior reading's
        # literal typography/quantity uncertainty discovery flag.
        if row["typed_uncertainties"] or typed:
            tags=list(dict.fromkeys([*tags,"uncertain"]))
        assert set(tags)<=allowed,(row["witness_key"],set(tags)-allowed)
        if kind in ("source-morphological-element","documentary-symbol","documentary-hypothesis","quoted-embedded-analysis-component"):
            assert "verb" not in tags and "noun" not in tags,row["witness_key"]
        proposals.append(dict(witness_key=row["witness_key"], physical_parent_key=row["physical_parent_key"], exact_before_guard={field:row[field] for field in ("form","gloss","language_label","printed_label")}, proposed_tags=tags, source_scope_rationale=note, interpretation_class=kind, typed_metadata_uncertainties=typed, original_literal_uncertainties_retained=row["typed_uncertainties"], original_evidence=dict(printed_page=row["printed_page"], image=row["original_image"], image_sha256=row["original_image_sha256"]), source_gloss_unchanged=True, accepted_for_installation=False))
    target=HERE/"independent-grammatical-tag-proposal.jsonl"
    payload="".join(json.dumps(row,ensure_ascii=False)+"\n" for row in proposals)
    if target.exists() and target.read_text()!=payload:
        archive=HERE/"tag-proposal-history"/sha(target)
        archive.mkdir(parents=True,exist_ok=True)
        for name in ("independent-grammatical-tag-proposal.jsonl","independent-grammatical-tag-review.json"):
            original=HERE/name
            saved=archive/name
            if saved.exists(): assert saved.read_bytes()==original.read_bytes()
            else: shutil.copyfile(original,saved)
    target.write_text(payload)
    report=dict(status="complete independent grammar tag proposal; owner/root output reconciliation required",input_path=str(path.relative_to(ROOT)),input_sha256=sha(path),registered_tags_sha256=sha(ROOT/"data/tags.py"),all224keys_exact=True,counts=dict(Counter(row["interpretation_class"] for row in proposals)),tag_usage=dict(Counter(tag for row in proposals for tag in row["proposed_tags"])),literal_uncertain_rows=sum(bool(row["original_literal_uncertainties_retained"]) for row in proposals),uncertain_union_verified=all("uncertain" in row["proposed_tags"] for row in proposals if row["original_literal_uncertainties_retained"] or row["typed_metadata_uncertainties"]),typed_metadata_cases=[row for row in proposals if row["typed_metadata_uncertainties"]],output_sha256=sha(target),explicit_policies=["No new canonical tags or shared code changes.","Uncertain flag is the union of retained original literal uncertainties and proposed metadata uncertainties; never replace away prior typography/quantity flags.","No generic scanning of contextual prose; decisions scoped to reviewed source parent/rank.","Pure person/TAM table labels remain blank lexical Gloss and become Tags.","Isolated morphology never receives lexical verb/noun POS.","Printed2.m. retains masculine singular/present/copula plus typed conflict, withholds2sg and3sg.","Shared thou/he forms keep common singular only, conflicting person/gender interpretations retained as typed alternatives.","Quoted symbols and unstarred hypothetical avnd stay documentary.","Person/number on complete examples describes source glossed use, not an invented suffix analysis.","No sentential inference from physical line segmentation; multiword examples retain only multiword-expression unless separately source-boundary reviewed."],canonical_changes=False)
    (HERE/"independent-grammatical-tag-review.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("status","input_sha256","counts","output_sha256")},indent=2))


if __name__=="__main__":main()
