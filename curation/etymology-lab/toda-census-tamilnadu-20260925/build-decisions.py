"""Conservative Toda Census article proposals; no accepted sidecar writes."""
import csv
import gzip
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
records = json.loads((P / 'survey-inventory.json').read_text())
assert len(records) == 456

# Indices refer to the fixed exact residual inventory. DEDR IDs are article IDs.
mapping = '''1:1481;7:1514;15:4753;21:1298;25:1481;56:2170;61:4939;62:3393;64:3606;65:5152;71:1883;77:657;81:1159;86:5292;90:1479;97:3633;98:3986;102:215;122:4345;123:1438;158:220;173:4068;186:474;187:5052;189:474;190:3655;191:2826;192:910;193:1025;195:3655;215:3612;216:504;220:4834;221:83;227:1624;249:347;311:55;358:1851;390:5020;402:4766;408:516;414:410;416:5151;422:5160;426:1;429:1;430:1;435:5154;437:5151;438:5151;439:5151;452:833;453:5142;455:4460;456:1182'''
parents = {int(i): 'd' + p for i, p in (clause.split(':') for clause in mapping.split(';'))}
assert len(parents) == len(mapping.split(';'))
articles = {}
with gzip.open(DATA / 'data/dedr/entry-texts-audit.csv.gz', 'rt') as fh:
    for row in csv.DictReader(fh):
        if row['Entry_ID'] in set(parents.values()):
            articles[row['Entry_ID']] = {key: row[key] for key in ('Page', 'Citation', 'Raw_Article')}
            articles[row['Entry_ID']]['URL'] = 'https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=' + row['Page']
assert set(articles) == set(parents.values())

def toda_section(article):
    m = re.search(r'\bTo\. ', article)
    assert m, article[:70]
    tail = article[m.end():]
    end = re.search(r'\s(?:Ka|Ko|Te|Tu|Pa|Ku|Ta|Ma|Kod|Ir|Kur|Kol|Gad|Go|Kon|Mal|Bra)\. ', tail)
    return tail[:end.start()] if end else tail

# These notes pin the comparison to the reflex within the whole article and
# identify extensions/inflections whose surface form is not an exact match.
notes = {
  1:'The Toda subsection has ko·ṯ “wind”; the survey uses the adjacent air sense.',
  7:'The Toda subsection gives tït “fire”; the separate kic fire term is marked as borrowed from Badaga.',
  15:'The Toda subsection has maw “rain”; the survey mār spelling is a close local variant.',
  25:'The Toda subsection gives ko·ṯ “wind”; the survey kōṭṭəc contains the same wind base with extra material.',
  56:'The Toda subsection gives kwaṯy “cat”, matching the source cat response with transcriptional differences.',
  82:'The Toda subsection gives mu·n “face, front”; source konmun appears to include the face element as a compound.',
  86:'The Toda subsection has poḷc- “live, be alive”; the longer survey life noun shares the verbal base.',
  102:'The Toda subsection gives ašky “rice”; its fuller nes̱išky form is also included.',
  122:'The Toda subsection gives pu·f “flower”, a plausible equivalent of source pōv.',
  123:'The Toda subsection has ko·ṛ “cultivated field”; the survey garden sense is a local extension.',
  158:'The Toda subsection gives arsn “saffron, yellow”; survey harjən retains the yellow sense.',
  185:'The Toda subsection gives muda·l “first”; the source expression mudəl koi contains the ordinal first element.',
  186:'The Toda subsection gives e·ḍ “two”; source ᴂḍvɔθ is an ordinal derivative “second”.',
  187:'The Toda subsection gives mu·ḏ “three”; source murɔθ is an ordinal derivative “third”.',
  195:'The Toda subsection explicitly gives nal po Q “forty”, matching source nəlpət.',
  216:'The Toda subsection gives i·ṣf- “drag, pull”; source ēst has the same action.',
  227:'The Toda subsection gives ki·ṟ- “tear”; source kɛ̄rt is compatible though the vowel differs.',
  282:'The Toda subsection gives ïxy- “(calf) sucks”; source īstitə is an extended elicited verb.',
  390:'The Toda subsection gives mudk “in front” and muda·l “first”; source madalk “before” has the temporal/front sense.',
  408:'The Toda subsection gives eṟk “down, the east”, matching source ærkʰ “down”.',
  416:'The interrogative Toda subsection gives egy “what manner, how”; source ɛgɛ̄c is an inflected how form.',
  422:'The first-person Toda subsection gives oblique en-; source en “my” uses that possessive stem.',
  429:'The distal Toda subsection gives plural a Q a·m “those”; source aθam preserves the distal plural.',
  437:'The interrogative Toda subsection explicitly gives etfok, etfïn “when”, matching source etɔk.',
  438:'The interrogative Toda subsection explicitly gives e·l “where”, matching source ᴂl.',
  439:'The interrogative Toda subsection explicitly gives o·ry “who”, matching source oriyo.',
}
accepted = []
for ix, pid in sorted(parents.items()):
    record = records[ix-1]
    section = toda_section(articles[pid]['Raw_Article']).strip()
    brief = section[:240].rsplit(' ', 1)[0] + ('…' if len(section) > 240 else '')
    evidence = f'DEDR {pid[1:]} Toda: {brief} '
    evidence += notes.get(ix, f'The survey {record["Form"]} “{record["Gloss"]}” corresponds to this Toda form and sense.')
    accepted.append({'record':record,'parent':pid,'kind':'reflex','citation':record['Source']+';dedr[entry '+pid[1:]+']',
                     'evidence':evidence,'status':'pending-review','surveyIndex':ix})

special_holds = {
  4:'Source kɔarv “darkness” might involve DEDR 1395 ka·r dark/dirt or the darkness form nested in DEDR 1278; this source transcription does not select one safely.',
  5:'Source bumi “earth” has a widespread Indo-Aryan/Dravidian regional form; immediate donor and Toda inheritance path need checking.',
  19:'Source ponni “snow” resembles Toda pony “dew” in DEDR 4035, but the snow/dew elicitation and phonological identity remain uncertain.',
  75:'Source tuv “wing” resembles Toda tu·fy “feather, bird’s tail” in DEDR 3393, but the exact anatomical sense is not established.',
  87:'Source məṭmīr “hair” contains Toda mi·r hair (DEDR 4707), but the first component remains unresolved.',
  131:'Source kic means “potato”; DEDR 1514 Toda kic is “fire” and explicitly borrowed from Badaga. Do not link the homograph.',
  151:'Source paccai “green” is Tamil-like; DEDR 3821 has Toda poč “green”, but the source’s immediate contact donor remains unclear.',
  164:'Source mōyi “ink” resembles Toda moy “ashes/dark” in DEDR 5101; ink may be a contact innovation and the donor is unresolved.',
  82:'Source konmun “face” contains Toda mu·n face (DEDR 5031), but the kon- component remains unresolved.',
  185:'Source mudəl koi “first” has Toda muda·l first (DEDR 5020) as its first element, but the final koi is unanalysed.',
  282:'Source īstitə “suck” may relate to Toda ïxy- (calf) sucks (DEDR 504), but its form also resembles the article’s distinct i·ṣf- “pull”; hold.',
  188:'Source uḍ “one” does not match the recorded Toda one forms in DEDR 990; the exact reading needs verification.',
  194:'Source vəd “twenty” does not isolate the Toda ïfo Q “twenty” form of DEDR 3918 securely.',
  224:'Source pərjit “squeeze” differs from Toda pï Q x- “squeeze” in DEDR 4165; the form relation is unresolved.',
  225:'Source divət “spit” differs from Toda tüf ïn- “spit” in DEDR 3323; hold the plausible sense-only match.',
  232:'Source kest “cut” has several possible Toda cut/pluck families, with no secure same-form choice.',
  260:'Source oṭci use “run” is a complex elicited form; the Toda run stem in DEDR 1041 is not securely isolated.',
  267:'Source titnəṭət “walk” might contain a Toda naṛ- walk element (DEDR 3582), but the full analysis is unresolved.',
  324:'Source nīrɔṭɯt “bathe” contains nīr water (DEDR 3690), but the bathing verb is not accounted for.',
  401:'Source koɔrv “today” has no verified direct Toda today reflex in the searched article cache.',
  411:'Source amokʰ “he/she” may be a distal pronoun, but its number/gender formation is not secure under DEDR 1.',
  431:'Source adam “they” resembles distal plural a Q a·m (DEDR 1), but the elicited form also occurs for “these”; retain the ambiguity.',
  433:'Source adam is glossed “these” yet resembles the separately elicited “they”; the proximal/distal distinction is unresolved.',
}
held = []
for ix, record in enumerate(records, 1):
    if ix in parents: continue
    label = f'{record["Form"]} “{record["Gloss"]}”'
    if ix in special_holds: reason = special_holds[ix]
    elif len(record['Form'].split()) > 1 or len(record['Form']) > 12:
        reason = f'{label} is multiword or morphologically extended; no full component-by-component article analysis has been verified.'
    elif ix in set(range(103,122)) | set(range(124,146)) | set(range(165,184)):
        reason = f'{label} is a plant, material, institution or contact-culture response; a Toda inheritance route or immediate donor is not established.'
    elif ix >= 407 and ix <= 448:
        reason = f'{label} is an inflected pronoun, deictic or function word; the exact paradigm and article-level derivation remain unresolved.'
    else:
        reason = f'{label} has no verified Toda form-and-sense correspondence in the full DEDR articles examined.'
    held.append({'record':record,'reason':reason,'status':'research-hold','surveyIndex':ix})
assert len(accepted)+len(held)==456
owners={x['record']['ID']:'proposed' for x in accepted}
owners.update({x['record']['ID']:'held' for x in held})
assert len(owners)==456
def write(name, obj): (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
write('proposed-decisions.json', {'status':'pending-review','accepted':accepted,'held':held})
write('research-holds.json',held)
write('unexamined-inventory.json',[])
write('primary-evidence.json',articles)
write('record-coverage-001.json',{'source':'census2023tamilnadu','language':'Toda','residualSourceRecords':456,
                                  'proposedRecords':len(accepted),'heldRecords':len(held),'unexaminedRecords':0,
                                  'uniqueOwnership':True,'recordOwnership':owners})
print(json.dumps({'proposed':len(accepted),'held':len(held),'articles':len(articles)}))
