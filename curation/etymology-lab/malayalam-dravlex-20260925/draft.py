"""Prepare pending-review Malayalam DravLex decisions; no accepted graph writes."""
import csv
import gzip
import json
from pathlib import Path

P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text())
raw=json.loads((P/'source-record-evidence.json').read_text())
assert len(records)==len(raw)==45

# Keys are one-based positions in the exact residual inventory. Parent IDs are
# whole DEDR articles, following the user's clarification of their scope.
proposals={
    1: (['d4711','d3559'],'The survey marattoli “bark” transparently combines Malayalam maram “tree” (DEDR 4711, Tamil compound stem marattu-) with Malayalam toli “skin, bark, peel, rind” (DEDR 3559). Ordered components preserve the compound rather than assigning it wholly to one head.'),
    8: (['d3393'],'DEDR 3393 directly quotes Malayalam tūval “feather, pen, quill”, beside Tamil tūval and Toda tu·fy. The survey tūvalə retains a final schwa in its house transcription.'),
    11: (['d3682'],'DEDR 3682 gives Malayalam niṟayuka “become full, be full” and niṟa/niṟavu “fullness”, beside Tamil niṟai. The survey niraña “full” is a completed/adjectival formation in this family; its exact inflection is not separately printed there.'),
    14: (['d2017'],'DEDR 2017(a) directly gives Malayalam kēḷkka “hear, perceive, listen”, with Tamil kēḷ and Kodagu kë·ḷ-. The survey kelkuga differs in vowel length and verbal ending but preserves the same hearing stem.'),
    16: (['d314'],'DEDR 314 quotes Malayalam aṟiyuka “know, understand”, plus Tamil aṟi, Kannada aṟi and Kodagu ari-. The survey arayuga differs in medial vowel and rhotic notation; its DravLex know-A comparanda identify this article, but the transcription needs review.'),
    17: (['d546'],'DEDR 546 directly quotes Malayalam īral “liver”, beside Tamil īral and Irula i·ralu. The survey has the same form and gloss.'),
    18: (['d1274'],'DEDR 1274 directly quotes Malayalam karaḷ/karuḷ as “lungs and heart, liver, bowels”, beside Kannada karuḷ and Tulu karaḷů. The survey narrows this polysemous organ word to “liver”.'),
    24: (['d1366'],'DEDR 1366 directly quotes Malayalam kar̤uttu “neck” beside Tamil kar̤uttu and Kannada kattu; the survey kaɻutə keeps a distinct rhotic and reduced final consonant notation. The DEDR article also compares Sanskrit kaṇṭha, but the whole Dravidian article is the proposed parent.'),
    28: (['d664'],'DEDR 664(a) directly gives Malayalam uruṇṭa “round” from uruḷuka (uruṇṭ-) “roll”, with Tamil uruṇṭai “ball”. The survey urunḍa has the same stem and sense in its transcription.'),
    29: (['d4031'],'DEDR 4031 directly gives Malayalam paṟayuka “say, speak, tell”, beside Tamil paṟai and Kodagu pare-. The survey parayuga represents the same verbal stem with a different rhotic and ending notation.'),
    30: (['d3978'],'DEDR 3978 quotes Malayalam parippu “peas, pulse; kernel of corn” and Tamil paruppu “kernel”. The survey parippə glosses it broadly as “seed”; the parent supports a pulse/kernel sense, so the broader survey gloss remains qualified.'),
    33: (['d1298'],'DEDR 1298 directly gives Malayalam kal/kallu “stone, rock”, with Tamil kal and Kodagu kallï. The survey kallə is its form with final schwa transcription.'),
    35: (['d3687'],'DEDR 3687 directly gives Malayalam nīntuka “swim”, with Tamil nīntu and Kannada nīntu. The survey neentuga has equivalent long-vowel and suffix spelling.'),
    36: (['d1'],'DEDR 1 directly gives Malayalam a/ā “that, yonder” in the distal demonstrative article. The survey ā is the long base, kept separate from its adə “that” record.'),
    37: (['d1'],'DEDR 1 directly gives Malayalam atu “that thing” within the distal demonstrative family. Survey adə retains a voiced dental and final-schwa transcription; the article covers the whole paradigm.'),
    38: (['d410'],'DEDR 410 directly gives Malayalam itu “this thing” within the proximal demonstrative family. Survey idə retains a voiced dental and final-schwa transcription.'),
    41: (['d1594','d2654'],'The survey cerəcūḍə “warm” is analyzable as Malayalam ceṟu “small, little” (DEDR 1594) plus cūṭu “heat, warmth” (DEDR 2654). The source form is one written word; the two lexical parts are proposed in order, with their rhotic and stop spellings retained.'),
    43: (['d5496'],'DEDR 5496(a) explicitly gives Malayalam veḷuppu “whiteness, brightness” and veḷukka “be white”. The survey veḷuppə has the noun shape but the gloss “white”; this is a family-level colour link rather than an exact word-class equation.'),
    45: (['d4635'],'DEDR 4635 directly gives Malayalam mañña “yellow or turmeric colour”, beside maññaḷ “turmeric, yellow dye” and Tulu mañjala “yellow”. The survey maña differs in palatal-nasal length only.'),
}

holds={
    3:'The raw source leaves Malayalam asti “bone” unflagged, but this is a Sanskrit asthi-type form; no immediate donor attestation or native DEDR cognate has been verified. DravLex bone-E is a singleton.',
    4:'The raw source leaves Malayalam stanam “breast” unflagged, and groups Kannada stʰana and Telugu stʰanamu with it. Those Sanskrit-looking forms suggest contact; an immediate donor and reliable inherited DEDR family are unresolved.',
    20:'Malayalam ottiri “many” has no verified DEDR match. DEDR 410 gives Malayalam ittiri “so much”, but the vowel and quantifier sense differ; do not equate them solely by near spelling.',
}
assert set(proposals).isdisjoint(holds)
for i,(r,x) in enumerate(zip(records,raw),1):
    if i in proposals or i in holds:continue
    assert x['Loan']=='true',(i,r['Form'],x['Loan'])
    holds[i]=f'The DravLex raw row explicitly flags Malayalam {r["Form"]} “{r["Gloss"]}” as a loan. Its exact immediate donor and supporting attestation are not established; a Sanskrit-looking surface or cognacy code alone is insufficient for a borrowed edge.'
assert len(proposals)+len(holds)==45

accepted=[]
for i,(parents,evidence) in proposals.items():
    r=records[i-1];citation=r['Source']+';'+ ';'.join('dedr['+p[1:]+']' for p in parents)
    item=dict(record=r,parent=parents[0],kind='component' if len(parents)>1 else 'reflex',citation=citation,evidence=evidence,comparison='Full DEDR article and exact DravLex row compared; source spelling preserved.',status='pending-review')
    if len(parents)>1:item['components']=parents
    accepted.append(item)
held=[dict(record=records[i-1],candidates=[],reason=reason) for i,reason in sorted(holds.items())]
dec=dict(status='pending-review',accepted=accepted,held=held)
(P/'proposed-decisions.json').write_text(json.dumps(dec,ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
articles={}
needed={p for parents,_ in proposals.values() for p in parents} | {'d410'}
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as stream:
    for row in csv.DictReader(stream):
        if row['Entry_ID'] in needed:articles[row['Entry_ID']]=row
assert needed<=articles.keys()
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
print('proposed',len(accepted),'rows',sum(len(x.get('components',[x['parent']])) for x in accepted),'held',len(held),'unexamined',45-len(accepted)-len(held))
