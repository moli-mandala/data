from research_helpers import Batch
b=Batch(6)
def family(l,meaning,words,parent,evidence,tier='straightforward',senses=None,locator=None):
 allowed=[g for g in dict.fromkeys(r['Gloss'] for r in b.inv[l]) if set(g.split('; ')) <= set(senses or [meaning])]
 b.add(l,meaning,words,parent,evidence,tier,locator=locator,source_glosses=allowed)
for l,words in [('Malvi','bāp'),('Nimadi','bāp'),('Bagheli','bap')]:
 family(l,'father',words,'9209','CDIAL 9209.1 compares Prakrit bappa and Hindi, Gujarati and Marathi bāp ‘father’. This is the bāppa nursery-word family; the separate bābba branch is not required.',locator='9209.1')
family('Bagheli','father','bep','9209','Compare Hindi bāp and Prakrit bappa in CDIAL 9209.1. The source’s e-vowel needs dialectal confirmation; this is a qualified nursery-family identification, not a demonstrated vowel correspondence.',tier='qualified',locator='9209.1')
for l,words in [('Malvi','mā|māi'),('Nimadi','māi|māy'),('Bagheli','ma|mayi')]:
 family(l,'mother',words,'10016','CDIAL 10016 gives Prakrit māyā/māi, Hindi mā/māī/maiyā, Old Marwari mā (oblique māya), and Gujarati mā/māi. These reduced and y-bearing forms support the mātṛ family directly.')
for l,words in [('Malvi','bēn|behn|ben|beṇ|bahən'),('Nimadi','beiṇ'),('Bagheli','behin|bəhen|behina')]:
 family(l,'sister (older/younger)',words,'9349','CDIAL 9349 compares Prakrit bahiṇī, Hindi bahin/bahan, Gujarati bahen/ben and Old Marwari bahaṇa. The standalone forms fit this family; age-marked compounds are excluded.',senses=['older sister','younger sister'])
family('Malvi','brother (older/younger)','bhai','9661','CDIAL 9661 traces Hindi and Marwari bhāī and Gujarati bhāi through Prakrit bhāi/bhāia. The merged older/younger senses refer to the same unmodified kinship word.',senses=['older brother','younger brother'])
family('Bagheli','brother (older/younger)','bhay','9661','Compare Hindi/Marwari bhāī and Prakrit bhāi in CDIAL 9661. Both source senses are compatible with the unmodified term ‘brother’.',senses=['older brother','younger brother'])
for l,g,words in [('Malvi','younger brother','bhayyo'),('Bagheli','older brother','bheiya')]:
 family(l,g,words,'9661','Prakrit bhāia and Hindi bhāī in CDIAL 9661 support the brother stem. The expanded familiar ending is compatible with this family but its precise morphological history is not independently established here.',tier='qualified')
for l,words in [('Malvi','choro|corā|cora|coro|chorā|chora|chori|cori'),('Nimadi','choro|coro|corā|chora|cora|chori|cori')]:
 family(l,'child, son/boy, daughter/girl',words,'5070','CDIAL 5070.1 compares Prakrit chōyara, Nepali choro/chorī, Hindi chorā/chorī and Gujarati chorɔ/chorī. The lost intervocalic velar and masculine/feminine endings match; all grouped source meanings denote children.',senses=['child','son','boy','daughter','girl'],locator='5070.1')
family('Malvi','child, son/boy, daughter/girl','tsoro|sorā|tsori|sori','5070','The masculine/feminine pattern compares Hindi chorā/chorī and Gujarati chorɔ/chorī (CDIAL 5070.1). The source’s ts-/s- variants require confirmation of the local affricate/sibilant correspondence; retained as qualified.',tier='qualified',senses=['child','son','boy','daughter','girl'],locator='5070.1')
for l,words in [('Malvi','bāḷak|balak|barak'),('Nimadi','bāḷak|bāḷāk|baḷak')]:
 family(l,'child/son',words,'9216','CDIAL 9216 explicitly treats -k- forms such as bālak as either an extended stem or a learned loan from Sanskrit bālaka. The bāla family is secure enough to triage, but inherited versus learned transmission (and r for l in barak) remains unresolved.',tier='qualified',senses=['child','son'])
for l,words in [('Malvi','beṭā|beṭi'),('Nimadi','beṭi'),('Bagheli','beṭi|biṭṭiya|biṭiya|beṭeua|beṭeba|biṭeba|betaba')]:
 family(l,'son/daughter, boy/girl',words,'9238-2','CDIAL 9238.2 specifically gives *bēṭṭa: Prakrit biṭṭa/biṭṭī, Hindi beṭā/beṭī/biṭiyā and Maithili beṭuā/biṭiā. This is the child branch, not the entry’s root ‘defective’; Bagheli extended endings remain part of the qualified family comparison.' if l=='Bagheli' else 'CDIAL 9238.2 specifically gives *bēṭṭa, with Hindi beṭā/beṭī and Marwari beṭo/beṭī. This is the child branch, not the root entry *biḍḍa ‘defective’.',tier='qualified' if l=='Bagheli' else 'straightforward',senses=['child','son','daughter','boy','girl'])
for l,words in [('Malvi','laḍəka|laḍəkā|laḍəko|laḍəki|laḍki'),('Bagheli','leḍika|lerika|leḍəka|lerəka|leḍəke|leḍki|leḍəkiya|leḍeki|leḍiki|lerka|leḍəki|lerki|leṛke')]:
 family(l,'child, son/boy, daughter/girl',words,'10924','CDIAL 10924 compares Bhojpuri/Awadhi larikā and Hindi laṛkā, but explicitly allows both *laḍikka and *laḍḍikka for the central forms. Propose the shared expressive family with branch 2 as an unresolved alternative; vowel variation and suffixes do not settle that choice.',tier='qualified',senses=['child','son','boy','daughter','girl'])
for l,words in [('Malvi','rat|rāt'),('Nimadi','rāt'),('Bagheli','raṭ|raṭi')]:
 family(l,'night',words,'10702','CDIAL 10702 gives Prakrit rattī/rāī, Hindi rāt/rātī, Old Marwari rāti and Gujarati/Marathi rāt. '+('Bagheli’s transcribed retroflex ṭ is preserved and needs a local correspondence check.' if l=='Bagheli' else 'Loss of the cluster and final-vowel variation match these regional comparanda.'),tier='qualified' if l=='Bagheli' else 'straightforward')
for l,words in [('Malvi','aj|āj'),('Nimadi','āj'),('Bagheli','aj|aji|aju')]:
 family(l,'today',words,'242','CDIAL 242 gives Prakrit ajja, Apabhramsha ajju, Hindi/Marwari āj and eastern āji alongside Nepali āju. These provide both the palatal outcome of dy and the attested final-vowel variants.')
b.save()
