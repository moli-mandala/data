"""Conservative review-only Gondi/Koya residual etymology decisions."""
import csv
import gzip
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
records = json.loads((P / 'survey-inventory.json').read_text())
assert len(records) == 686
search = []
for start in range(0, 686, 35):
    end = min(start + 35, 686)
    search.extend(json.loads((P / f'discovery-{start+1:03d}-{end:03d}-results.json').read_text()))
assert len(search) == 686

# These are whole DEDR article families, checked against complete primary articles.
# Inflections in the Koya elicitation remain attached to the original source form.
groups = '''
1,10:3103;12,20,284,296:1977;31,324:698;
58:1514;62,70:5381;81:3728;114,632:2074;117,129,634:5087;
127,394,630:2674;130:2359;143:4764;
195:5052;196,481,659:3655;197,484,493,661:910;
240:5151;273,282:5099;275:3935;280:561;281,293:4528;
292:1479;294:4889;333:3903;337:3332;364:705;367:2626;
391,401:2115;402:3538;425:4198;450:3630;451:5320;
480:474;483,502:2485;509:3729;585:2017;
586:5160;604:2976;648:3821;673:1728
'''
parents = {}
for clause in groups.replace('\n', '').split(';'):
    if not clause.strip():
        continue
    indices, number = clause.strip().split(':')
    for value in indices.split(','):
        index = int(value)
        assert index not in parents
        parents[index] = 'd' + number

articles = {}
with gzip.open(DATA / 'data/dedr/entry-texts-audit.csv.gz', 'rt') as fh:
    for article in csv.DictReader(fh):
        if article['Entry_ID'] in set(parents.values()):
            articles[article['Entry_ID']] = {key: article[key] for key in ('Page', 'Citation', 'Raw_Article')}
            articles[article['Entry_ID']]['URL'] = 'https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=' + article['Page']
assert set(articles) == set(parents.values())

def anchor(parent):
    text = articles[parent]['Raw_Article']
    matches = re.findall(r'(?:Go\.|Gondi).{0,240}', text, re.I)
    return (matches[0] if matches else text[:240]).strip()

notes = {
    1:'Koya talakaia retains the tala head stem, with additional material left unanalyzed.',
    10:'Koya talakai retains the tala head stem, with additional material left unanalyzed.',
    34:'The Koya pestle word retains the us- pounding stem; the -pal ending needs separate study.',
    51:'The article explicitly has Gondi Koya eru “pond, tank”; the elicitation says water, a close but broader sense.',
    81:'The article includes Gondi nūka for broken rice; Koya nukka is consistent in form and rice sense.',
    117:'Gondi Koya mēka “goat” is directly in the article; Koya mekka has gemination.',
    129:'Gondi Koya mēka “goat” is directly in the article; Koya mekke differs in final vowel.',
    143:'The article gives Gondi mari “son”; the source also glosses boy.',
    152:'The article gives Koya payal “day”; source pīa is a reduced regional form.',
    172:'The article gives Gondi paḍana “old”; source panda is a regional contraction.',
    195:'The article gives Gondi muḍu “three”; source muddu is the same numeral family.',
    240:'The article gives Gondi benon and Koya benonḍ “who”; the source records benon.',
    273:'The article gives Gondi mēndul “body”; source menjul is a regional transcription.',
    282:'The article gives Gondi mēndul “body”; source mentuli adds a final vowel.',
    292:'The article gives Gondi kāl “leg”; source kalku is a plural form, distinct from the stone family.',
    294:'The article gives Gondi mukam/mokom “face”; source mukʰamu is direct.',
    316:'The article gives Gondi us- “pound” and pestle derivatives; source usmali is a regional pestle term.',
    337:'The article gives Gondi dumu and dumul “dust”; source dʰumˑu fits this family.',
    367:'The article gives Gondi īr “root”; source irku adds plural -ku.',
    425:'The article explicitly gives Gondi pila “girl” and Koya pilla “child”; both source glosses fit.',
    450:'The article gives Gondi nānd- “get wet”; source nanta(du) retains the wet stem.',
    451:'The article gives Gondi vatt- “dry”; source u̯atˑta(du) retains that stem.',
    571:'The article gives Gondi kēnj- “hear”; source kanja is a vowel variant.',
    574:'The article gives Gondi unḍ- “drink”; source unβat retains the initial drink stem but needs morphological review.',
    604:'The article gives Gondi Koya ḍokka “belly”; source dooka fits.',
    648:'The article gives Gondi pahna “green” and related “wet”; source pahna is direct.',
    673:'The article explicitly gives Gondi Koya kud- “sit”; source kud is direct.',
    678:'The article gives Gondi Koya ūṛ- “see”; source ūda is a dialectal reflex.',
}

accepted = []
for index, parent in sorted(parents.items()):
    record = records[index-1]
    evidence = f'DEDR {parent[1:]} full article (p. {articles[parent]["Page"]}): {anchor(parent)} '
    evidence += notes.get(index, f'Source {record["Form"]} “{record["Gloss"]}” continues the Gondi family named in the article; regional transcription or inflection is retained.')
    accepted.append({'record':record,'parent':parent,'kind':'reflex',
                     'citation':record['Source']+';dedr[entry '+parent[1:]+']',
                     'evidence':evidence,
                     'comparison':'Read-only homepage exact-form discovery and complete DEDR article reviewed; whole-article family proposal.',
                     'status':'pending-review','surveyIndex':index})

special = {
    8:'Kalu has merged leg and stone glosses; those belong to distinct DEDR articles (1479 and 1298), so the shared record has no single secure family.',
    14:'Palu has merged teeth and milk glosses; those belong to distinct whole-article families.',
    34:'Ucpal “pestle” resembles a derivative of us- “pound” (DEDR 583), but the c/pal material is not established.',
    51:'Eru is glossed water; DEDR 1980 gives Gondi Koya eru “pond, tank,” but the wider liquid sense needs source confirmation.',
    87:'Niruli “onion” may be a compound with ulli, but the preceding element has not been analyzed.',
    89:'Korra is glossed thorn, while the exact Gondi korra in DEDR 2163 names millet; this is a homograph, not evidence for the thorn sense.',
    90:'Pani “fruit” resembles the ripe/fruit article 4004, but the full article needs a closer form-and-sense comparison.',
    99:'Mada “tree” may reflect mara (DEDR 4711), but d/r transcription here is not independently established.',
    101:'Veku “root” resembles vēr (DEDR 5535), but the medial consonant does not match securely.',
    152:'Pīa “day” may be a reduced form of Gondi payal (DEDR 3805), but the deletion pattern needs verification.',
    172:'Panda “old” may continue Gondi paḍana (DEDR 3999), but the source contraction is not established.',
    123:'Bare is glossed buffalo, but the exact Gondi string in article 2966 belongs to a different context; no buffalo family is secured.',
    126:'Naīu merges dog and oil senses; cannot choose one DEDR family for the combined record.',
    163:'Nadi merges tomorrow, river and yesterday senses; no single etymon covers them.',
    186:'Era is glossed red, but the exact Gondi era in article 2821 refers to a handful; this match is semantic noise.',
    188:'Idu merges five and demonstrative “this” senses; numeral and pronoun families differ.',
    205:'Nai merges nine and oil senses; numeral and oil families differ.',
    257:'Kenga “hear” is suggestive of kēnj- (DEDR 2017), but g/j correspondence is insufficiently checked.',
    316:'Usmali “pestle” resembles the us- “pound” family (DEDR 583), but the derivative is not directly attested in the article.',
    445:'Indike “today” has multiple possible deictic analyses; precise primary article assignment is unresolved.',
    569:'(Iga)va “come” contains the likely va- stem of DEDR 5270, but the parenthesized prefix remains unanalysed.',
    571:'Kanja “hear” resembles Gondi kēnj- (DEDR 2017), but the vowel pattern needs verification.',
    574:'Unβat “drink” contains un- (DEDR 600), but the β-bearing ending is unanalysed.',
    626:'Niraik “ash” resembles nīr, but the full article 3693 is specifically “water”; hold this semantic conflict.',
    635:'Naittars “snake” includes tars (DEDR 2359), but the preceding constituent remains unanalysed.',
    678:'Ūda “see” resembles Gondi Koya ūṛ- (DEDR 2735), but the dental/rhotic correspondence is unverified.',
    683:'Mamma is glossed inclusive “we”; DEDR 5154 gives Gondi mamma but labels the Tr. paradigm exclusive, so inclusion requires source-specific confirmation.',
}
held = []
for index, (record, result) in enumerate(zip(records, search), 1):
    if index in parents: continue
    form, gloss = record['Form'], record['Gloss']
    label = f'{form} “{gloss}”'
    if index in special: reason = special[index]
    elif ';' in gloss or '?' in gloss: reason = f'{label} has merged or qualified source senses; a specific sense must be isolated before assigning a family.'
    elif ' ' in form or '(' in form or len(form) > 15: reason = f'{label} is multi-part or morphologically extended; its ordered components have not been verified.'
    else: reason = f'{label} lacks a verified matching form and sense in a complete DEDR article at this checkpoint.'
    held.append({'record':record,'reason':reason,'status':'research-hold','surveyIndex':index,'searchResultCount':result['count']})
assert len(accepted)+len(held) == 686
assert len({x['record']['ID'] for x in accepted+held}) == 686

def write(name, value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
write('proposed-decisions.json', {'status':'pending-review','accepted':accepted,'held':held})
write('research-holds.json', held)
write('unexamined-inventory.json', [])
write('primary-evidence.json', articles)
print(json.dumps({'proposed':len(accepted),'held':len(held),'articles':len(articles)}))
