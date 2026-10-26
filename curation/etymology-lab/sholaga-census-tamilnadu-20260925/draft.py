"""Research-only Sholaga Census whole-article proposals and source-wide holds."""
import csv, gzip, json, re
from pathlib import Path

P = Path(__file__).resolve().parent
D = P.parents[2]
records = json.loads((P/'survey-inventory.json').read_text())
assert len(records) == 439
search = []
for start in range(0, 439, 35):
    end = min(start+35, 439)
    search += json.loads((P/f'discovery-{start+1:03}-{end:03}-results.json').read_text())
assert len(search) == 439

# Only whole DEDR families confirmed from the full primary article are assigned.
# Inflectional endings in the Census response stay in the source record.
groups = '''
8:1438;12:5503;13:5474;27:131;28:3085;44:23;45:3015;
55:2901;56,69:2591;59:3606;63:1711;67:4307;68:5155;
83:1366;85:3633;86:3986;87,365:3584;90:4096;
109:5485;122:3163;124:3753;142:1249;144:1787;
153:5496;157:3793;181:5496;183,185:474;184:5052;186:3655;187:2826;
202:3570;203:4797;204:1299;209:625;212:4834;225:3549;
240:5270;244:3071;245:4020;248:2053;249:4572;
260:3582;263:5093;264:2119;267:3583;270:5401;
272:1124;273:76;285:2426;293:4834;297:793;303:55;
304:2017;310:4934;313:4960;322:1443;325:1029;
349:3772;358:8;373:5020;
393:410;396,402,403:5160;398:410;404,416:3647;405:2559;
407,408,411,412,413:1;417,418,419,420,421,422:5151;
423:3684;424:3688;429:698;437:4879
'''
parents = {}
for clause in groups.replace('\n','').split(';'):
    if not clause.strip(): continue
    indices, number = clause.strip().split(':')
    for value in indices.split(','):
        index = int(value)
        assert index not in parents, index
        parents[index] = 'd'+number

articles = {}
with gzip.open(D/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
    for article in csv.DictReader(fh):
        if article['Entry_ID'] in set(parents.values()):
            articles[article['Entry_ID']] = {key:article[key] for key in ('Page','Citation','Raw_Article')}
            articles[article['Entry_ID']]['URL'] = 'https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+article['Page']
assert set(articles) == set(parents.values())

def anchor(parent):
    article = articles[parent]['Raw_Article']
    start = article.find('Ka. ')
    if start < 0: start = article.find('Ta. ')
    assert start >= 0, parent
    return article[start:start+210].rsplit(' ',1)[0]+'…'

notes = {
    8:'The article gives Tamil kāṭu and Kannada kāḍu “forest”; the Sholaga kāṭu keeps that woodland sense.',
    12:'Kannada beḷḷa and Tulu boḷḷa “flood” are in this article; Sholaga boḷḷjə̄ preserves that flood stem with a source ending.',
    13:'The article explicitly gives Kannada beṭṭa “mountain”; the Sholaga bɛṭṭā has the same landform sense.',
    27:'The article gives Tamil aṇṇaṉ/Kannada aṇṇa “elder brother”; the Census gloss joins elder and younger, so only its elder sense is supported.',
    28:'The article gives Kannada tamma “younger brother”; the Census gloss joins elder and younger, so only its younger sense is supported.',
    44:'Kannada akka is “elder sister”; the merged elder/younger Census gloss is narrowed to its elder sense.',
    45:'Kannada taṅgi is “younger sister”; the merged Census gloss is narrowed to its younger sense.',
    55:'The article lists Kannada ēḍi “crab”; the Census ēdi retains its crab sense.',
    56:'The article includes Kannada ṟekke and Kodagu rekke “wing,” with related feather reflexes; the Census uses rekka for feather.',
    59:'Kannada nari is specifically “jackal, fox”; the Census nari selects fox.',
    63:'Tamil kutirai and Kannada kudire mean “horse”; Census kutira is the horse stem.',
    67:'The tiger article has Tamil/Kannada puli and explicitly mentions a Badaga huli form; Sholaga huḷi keeps that southern tiger base.',
    68:'The article lists Kannada āve “tortoise”; the Census āvə is directly comparable.',
    69:'The article lists Kannada ṟekke, Kodagu rekke and Telugu rekka for “wing”; Census rekka has the wing sense.',
    83:'The article explicitly gives Kannada kattu “neck, throat”; Census kattu is exact.',
    85:'The article gives Kannada nālage/nālige “tongue”; Census nāḷaga retains the tongue stem.',
    86:'The article gives Tamil/Kannada pal “tooth”; Census haḷḷu is the regional p/h form of this tooth base.',
    87:'The article gives Kannada naḍu “middle, centre, waist”; Census naḍu has the waist sense.',
    90:'The article gives Tamil/Kannada pāl “milk”; Census haḷu is the regional p/h form of the milk noun.',
    109:'DEDR 5485 explicitly gives Kannada bidir(u) “bamboo”; Census bidīru is the same bamboo noun.',
    122:'Tamil tāmarai and Kannada tāmare “lotus” are in this article; Census tamara has that plant sense.',
    124:'The article gives Tamil nel and Kannada nellu “paddy”; Census paddā has a less exact form and remains a provisional family comparison.',
    142:'The article gives Tamil kai and Kannada kai “bitter”; Census kaiyɛ preserves the taste stem.',
    144:'The article gives Tamil kuruṭu and Kannada kuruḍu “blind”; Census kuruṭi is the corresponding adjective stem.',
    153:'The article contains Tamil veḷḷai and Kannada biḷi “white”; Census viḷiyatu keeps the white stem with a source ending.',
    157:'The article explicitly gives Kannada nōvu “pain”; Census nōvu is exact.',
    181:'The white/bright article also gives Tamil veḷḷi “silver”; Census beḷḷi selects the metal sense.',
    183:'The two article includes Tamil iraṇṭāvatu “second” and Kannada eraḍaneya; Census iraṭāvatu is an ordinal response ending in jāgā “place.”',
    184:'The three article includes Tamil mūṉṟām and Kannada mūraneya “third”; Census murāvatu retains the three stem with ordinal morphology and jāgā “place.”',
    185:'The article gives Tamil iraṇṭu and Kannada eraḍu “two”; Census eraṭu is direct.',
    186:'The article explicitly gives Kannada nāku “four”; Census nākku retains this numeral form.',
    187:'The article gives Tamil aintu and Kannada aidu “five”; Census aitu is direct.',
    202:'The article gives Tamil/Kannada nakku “lick”; Census nakkatu is the elicited verb form.',
    203:'The article gives Kannada māḍu “do, make”; Census māṭitivi preserves the making stem with an elicited ending.',
    204:'The article gives Tamil kala and Kannada kalaku “mix”; Census kaḷakkatu keeps the mixing verb.',
    209:'The article gives Tamil/Malayalam untu “push”; Census untu is direct.',
    212:'The article explicitly includes Tamil māṟu “sell, exchange”; Census mār keeps that exchange stem.',
    225:'The article includes Tamil tōṇṭu and Kannada tōḍu “dig”; Census tōṭatu keeps the digging root with an elicited ending.',
    240:'The article gives Kannada bar/bār “come”; Census barātu is its elicited verb form.',
    244:'The article gives Tamil/Kannada tappu “escape, miss”; Census tappicatu is an extended escape verb.',
    245:'The article gives Tamil paṟa and Kannada hāru “fly”; Census āratu is the regional flying form.',
    248:'The article gives Kannada koḍu “give”; Census koḍātu is its elicited verb form.',
    249:'The article gives Kannada hōgu “go”; Census hōgatu is its elicited verb form.',
    260:'The article gives Kannada naḍe “walk”; Census naḍaiyatu is an elicited walking form.',
    263:'The article gives Kannada mēyu/mēsu “graze”; Census mēcātu is a comparable graze form.',
    264:'The article gives Kannada koy/kuy “reap”; Census kuyiyatu is an elicited harvest verb.',
    267:'The article gives Kannada naḍu and nāṭu “plant”; the source response nāttā naṭṭatu repeats the planting stem.',
    270:'The article gives Kannada bittu and bittane “sow/seed”; Census bidtana keeps the sowing stem.',
    272:'The article gives Kannada kaḍi “bite”; Census kaṛiy keeps that bite stem.',
    273:'The article gives Tamil aṭu and Kannada aḍu “cook”; Census aṭike is the cooking form.',
    285:'The article gives Kannada sāy “die”; Census āyātu is a less exact local form retained for review.',
    293:'The article gives Tamil māṟu “change”; Census māttāku has an extended change form.',
    297:'The article gives Kannada eṇṇu “count”; Census ēṇiatu is an elicited counting form.',
    303:'The article gives Kannada añju “fear”; Census añcatu is the same fear verb.',
    304:'The article gives Kannada kēḷu “hear”; Census kēɻatu preserves the hearing stem.',
    310:'The article gives Kannada muṭṭu “touch”; Census muṭṭatu is the elicited touch verb.',
    313:'The article gives Kannada muttu “kiss”; Census muttāgatu retains the kiss stem.',
    322:'The article gives Tamil kāṭṭu and Kannada kāṭu “show”; Census tōrijitivi differs in its lexical stem, so this candidate requires review.',
    325:'The article gives Kannada ōkāḷa “vomiting”; Census kakkāḷu appears to preserve this retching base with a changed onset.',
    349:'The article gives Kannada nēra “straightness”; Census nērā is direct.',
    358:'The article gives Kannada agala “width”; Census agala is exact.',
    365:'The article gives Kannada naḍu “middle”; Census mattiyata “middle” is morphologically different and requires review.',
    373:'The article gives Kannada mun/munnu “before”; Census munte retains the temporal stem.',
    393:'The article gives Kannada illi “here” under the proximal i- demonstrative; Census iḷḷi is a regional locative.',
    396:'The article gives Kannada nān “I”; Census nānu is exact.',
    402:'The article gives Kannada nān and oblique nan- “I”; Census nanaga adds a dative ending to the oblique.',
    403:'The article gives Kannada nān and oblique nan- “I”; Census nannatu adds possessive morphology.',
    398:'The article gives Kannada itu “this”; Census itu is the proximal demonstrative.',
    404:'The article gives Kannada nāvu with oblique nam- “we”; Census nāmu is the inclusive plural stem despite its “our” gloss.',
    405:'The article gives Tamil illai and Kannada illa “not”; Census iḷḷa is direct.',
    407:'The article gives Kannada atu/adu “that”; Census atu is distal.',
    408:'The article gives Kannada avar “those persons”; Census avarigā retains this distal plural plus dative marking.',
    411:'The article gives Kannada alli “there”; Census aḷḷi is a distal locative.',
    412:'The article gives Kannada avar/avargaḷ “they”; Census avarugā retains the distal plural stem.',
    413:'The article gives Kannada avar “they”; Census avaratu is a possessive form of that distal plural.',
    416:'The article gives Kannada nāvu with oblique nam- “we”; Census nāmu keeps that inclusive-plural stem.',
    417:'The article gives Kannada ēn “what”; Census ēnā retains the interrogative stem.',
    418:'The article gives Kannada ēvāga “when”; Census ēvaga preserves that temporal interrogative.',
    419:'The article gives Kannada elli “where”; Census eḷḷi retains the locative interrogative.',
    420:'The article gives Kannada yār/ār “who”; Census ārā is the human interrogative.',
    421:'The article gives Kannada yār “who”; Census yāra is a genitive interrogative.',
    422:'The article gives Kannada yār “who”; Census yārava is a case-marked interrogative retained for review.',
    423:'The article gives Kannada nīn with oblique nin- “you” singular; Census ninnaēri keeps the oblique stem.',
    424:'The article gives Kannada nīm/nīvu with oblique nim- “you” plural; Census nīmatu is its possessive response.',
    429:'The article gives Kannada oḷage “in, inside”; Census olɛkkā keeps the interior locative family.',
    437:'The article gives Kannada mīse “moustache”; Census mīca is the same hair noun in source notation.',
}

# Remove superficially similar candidates whose entire response does not fit
# the article. They remain explicit research holds below.
for index in (124,183,184,273,285,322,325,365,404,422):
    parents.pop(index)

accepted = []
for index,parent in sorted(parents.items()):
    record = records[index-1]
    evidence = f'DEDR {parent[1:]} full article: {anchor(parent)} '+notes[index]
    accepted.append({'record':record,'parent':parent,'kind':'reflex',
                     'citation':record['Source']+';dedr[entry '+parent[1:]+']',
                     'evidence':evidence,'comparison':'Homepage exact form and complete DEDR article checked; whole-article family proposal.',
                     'status':'pending-review','surveyIndex':index})

special = {
    1:'Kāɻi “air” resembles Kannada gāḷi (DEDR 1499), but the Sholaga k/g transcription and immediate donor remain unresolved.',
    4:'Būmi “earth” is a widespread Sanskrit-type noun; the immediate donor is unverified.',
    6:'Tivu “fire” resembles Tamil tī, but the Sholaga vowel and phoneme notation require checking before linking.',
    19:'Ottu is glossed “sun” here, but the primary article family for this exact regional form is uncertain.',
    23:'Kaikōcu “baby” is a compound-like response; neither first element nor its constituent family is secure.',
    34:'The maternal-uncle response contains māma (DEDR 4813), but its preceding kin-chain remains unanalysed.',
    35:'The maternal-uncle’s-daughter response is a long kin-chain; the source order and referents need verification.',
    36:'The same long kin-chain is glossed “nephew”; it cannot be linked before the referent contrast is resolved.',
    37:'The same long kin-chain is glossed “niece”; it cannot be linked before the referent contrast is resolved.',
    51:'Bekkina kotti “cat” is multi-part and the second element is not established in the DEDR cat family.',
    60:'The Sholaga o ːnṭu “goat” is orthographically uncertain; no sex-specific ram/goat article is selected.',
    92:'Uruɻai keḷau is glossed for both uncooked and cooked rice; the two senses have different article families.',
    107:'Aṭekɛmena “kitchen” looks morphologically extended; its hearth and house components are not secure.',
    113:'Manacikāyi “chilly” may contain a fruit term, but the first element and crop loan route are unresolved.',
    117:'Karikappu haṇṇu is glossed “garden” though haṇṇu normally names fruit; the source gloss needs inspection.',
    124:'Paddā “paddy” is not close enough to the nel/nellu paddy article (DEDR 3753) to justify that family.',
    183:'Iraṭāvatu jāgā “second” includes the two/second stem of DEDR 474, but jāgā “place” needs a separately verified donor.',
    184:'Murāvatu jāgā “third” includes the three/third stem of DEDR 5052, but jāgā “place” needs a separately verified donor.',
    125:'Araḷi mara is glossed “plough” although mara denotes a tree; source gloss/form alignment needs inspection.',
    161:'Ishukuḷu “school” appears to be a contact term; identify the immediate donor before linking.',
    163:'Huḍugā is glossed both “student” and “teacher”; the role contrast is unresolved.',
    166:'Toṭṭa tuppākki is glossed “enemy” although tuppākki is a gun word; source gloss alignment needs inspection.',
    170:'Enḍu is glossed “divorce” and again “hell”; source gloss alignment is uncertain.',
    173:'Enḍu is also glossed “hell”; this conflicts with its divorce elicitation.',
    188:'Ontāgi kaṭṭu “to bind” contains Kannada kaṭṭu (DEDR 1147) but its first component requires analysis.',
    192:'Ettuṭɨbərādɨ “carry” contains several verbal elements; no single article is sufficient.',
    199:'Oḍativi “hit” may be from aṭi (DEDR 77), but vowel and morphology are not secure.',
    200:'Iḍitivi “hold” has multiple possible southern roots; the exact holding stem needs verification.',
    208:'Eḷaiyatu “pull” resembles the ir̤u/er̤e pull family (DEDR 504), but its vowel and lateral need verification.',
    231:'Sāyisatu “kill” resembles causative of Kannada sāy “die” (DEDR 2426), but the causative analysis needs confirmation.',
    235:'Beṭṭa ēratu is glossed “sing” but apparently contains beṭṭa “mountain” and ēr “ascend”; this is a source conflict.',
    236:'Bantusēritu “arrive” contains Kannada bant- “come” (DEDR 5270) and another verb; source-order components need review.',
    237:'Beṭṭa ēratu is also glossed “ascend”; its mountain and climb components must be assigned separately.',
    246:'Īatu is glossed “get” and elsewhere “buy” or “earn”; its root and valency remain unclear.',
    266:'Haḷu “to milk” is formally the milk noun (DEDR 4096), but the elicited verbal derivation is not documented.',
    273:'Aṭike “cook” is a provisional response; article 76 gives aṭu/aḍu but the k-extension needs checking.',
    285:'Āyātu “die” lacks the initial s/c of Kannada sāy/Tamil cā (DEDR 2426).',
    300:'Māṭatu “think” resembles Kannada māḍu “do” (DEDR 4797), but the semantic identification is unsupported.',
    322:'Tōrijitivi “show” differs from the kāṇ/kāṭṭu show family (DEDR 1443); its own stem needs primary evidence.',
    325:'Kakkāḷu “vomit” resembles the ōkāḷa retching family (DEDR 1029), but the source onset differs.',
    329:'Entāgiḷa “bad” is a multi-part response with no secure negative or adjective article analysis.',
    365:'Mattiyata “middle” does not match naḍu (DEDR 3584); a different regional middle stem needs research.',
    370:'Baḷākai is glossed “south” but resembles a “right hand” expression; direction/hand alignment needs verification.',
    397:'Atu is glossed “it/this,” yet its distal a- stem fits “that”; the deictic contradiction remains on hold.',
    404:'Nāmu appears both as “our” and “we”; its possessive morphology is unclear, though the plural stem belongs with DEDR 3647.',
    399:'Atakkā inherits the unresolved “to it/to this” distal/proximal contradiction.',
    414:'Avaikaḷe is glossed “these” but its av- stem is distal in the primary demonstrative article.',
    422:'Yārava is glossed “whom” but may encode a different case of “who”; hold until its ending is analyzed.',
    434:'Urukki “mouse” has no checked correspondence to the eli/ili mouse family (DEDR 833).',
    439:'Mukganōṭ a kaṇṇāṭi “mirror” contains kaṇṇāṭi (DEDR 1182), but the first phrase is unanalysed.',
}
modern = ('bus','school','student','teacher','bullet','divorce','brass','money','sari','shoe','potato','tomato','ink','mason')
held=[]
for index,(record,result) in enumerate(zip(records,search),1):
    if index in parents: continue
    form,gloss=record['Form'],record['Gloss']
    label=f'{form} “{gloss}”'
    if index in special: reason=special[index]
    elif any(word in gloss.lower() for word in modern): reason=f'{label} is cultural or likely contact vocabulary; its immediate donor has not been verified.'
    elif ' ' in form or len(form)>14: reason=f'{label} is multi-part or morphologically extended; a full source-order component analysis is lacking.'
    elif '/' in gloss or '?' in gloss: reason=f'{label} has a merged or qualified source gloss; select the precise sense before assigning an article.'
    else: reason=f'{label} has no verified matching form and sense in the full DEDR articles examined.'
    held.append({'record':record,'reason':reason,'status':'research-hold','surveyIndex':index,
                 'searchResultCount':result['count']})
assert len(accepted)+len(held)==439
assert len({x['record']['ID'] for x in accepted+held})==439

def write(name, obj): (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
write('proposed-decisions.json',{'status':'pending-review','accepted':accepted,'held':held})
write('research-holds.json',held)
write('unexamined-inventory.json',[])
write('primary-evidence.json',articles)
print(json.dumps({'proposed':len(accepted),'rows':len(accepted),'held':len(held),'articles':len(articles)}))
