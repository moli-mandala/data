"""Draft conservative Kudiya Joseph G1/K1 article analyses; no accepted writes."""
import csv,gzip,json,re
from pathlib import Path
P=Path(__file__).resolve().parent; ROOT=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==223
mapping='''8:3633;17,18:5393;23:1204;26,27:3726;35:3690;42,43:5101;44,45:4570;46:5535;47,48:4004;49,50:4782;55,56:215;63:4635;74:2674;75:2248;80,81:2115;82,83:4085;85:4410;88:4395;93:3015;95,96,97:4616;101:2552;113,114:3999;115:3610;119,120:1594;121:5517;122:2654;124:3045;125,126,131:5276;134:2049;137,138:5496;139:1931;140:474;141:5052;142:2826;143:910;146,147,148:5151;156,157:410;158,159:1;164,165:5548;167:946;175:1124;178:600;179:1654;183:707;185:1524;187:1835;188,189:2053;193:2426;195:2132;196,197:4020;198:1041;199:4087;200,201:4572;203:4031;205:2017;207:1443;208,209:5160;210,211:3684;212,213,221:3688;215,217,223:1;218,220:5154'''
parents={}
for clause in mapping.split(';'):
 indices,num=clause.split(':')
 for ix in indices.split(','):parents[int(ix)]='d'+num
components={51:['d5373','d4004'],52:['d5373','d4004'],64:['d5496','d705'],65:['d5496','d705'],66:['d3690','d705'],67:['d3690','d705'],145:['d3918','d990'],160:['d410','d844'],161:['d1','d844']}
assert not (set(parents)&set(components))
article_ids=set(parents.values())|{p for parts in components.values() for p in parts}
articles={}
with gzip.open(ROOT/'data/dedr/entry-texts-audit.csv.gz','rt') as f:
 for r in csv.DictReader(f):
  if r['Entry_ID'] in article_ids:
   articles[r['Entry_ID']]={k:r[k] for k in ['Page','Citation','Raw_Article']}
   articles[r['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+r['Page']
assert article_ids==set(articles),sorted(article_ids-set(articles))
# Article-level evidence focuses on the exact reflex/comparandum where recoverable.
special={
8:'DEDR 3633 has Tamil nākku and Kannada nālige tongue; K1 nak retains the shorter tongue stem.',
17:'DEDR 5393 includes Tamil vīṭu house, Kannada bīḍu house and Kodagu buḍ-; G1 buḍɨ fits the southwestern b-/u- form.',
18:'DEDR 5393 includes Tamil vīṭu house, Kannada bīḍu house and Kodagu buḍ-; K1 buḍi is the matching dialect pair.',
42:'DEDR 5101 has Kannada masi soot and Toda moy ashes; G1 massi uses the ash sense of this black-residue family.',
43:'DEDR 5101 has Kannada masi soot and Toda moy ashes; K1 massi repeats the source-defined ash record.',
47:'DEDR 4004 has Tamil/Malayalam paḻam ripe fruit and Tamil paṇṭam fruit; G1 paźuntu preserves the fruit stem.',
48:'DEDR 4004 has Tamil paṇṭam fruit and Malayalam paḻam; K1 padam is compatible with the paṇṭam fruit variant.',
49:'DEDR 4782 directly records Kodagu māŋge mango; G1 maŋge has the same form and sense.',
50:'DEDR 4782 records Kodagu māŋge and Malayalam māṅṅa mango; K1 maŋga is the paired fruit word.',
63:'DEDR 4635 records Tamil mañcaḷ, Kannada arisina and Malayalam maññaḷ turmeric; K1 maja is a shortened local form.',
95:'DEDR 4616 includes Malayalam mōn son and Tamil makaṉ; G1 man is a local son form of this family.',
96:'DEDR 4616 explicitly gives Malayalam mōḷ daughter and Tamil makaḷ; G1 māḷ matches the Malayalam-type daughter form.',
97:'DEDR 4616 includes Tamil/Kannada makaḷ daughter; K1 magaḷ is a direct daughter form.',
113:'DEDR 3999 lists Tamil paḻaiya, Kannada paḷe and Malayalam paḻaya old; G1 paṛ̆ayate fits the old adjective.',
114:'DEDR 3999 lists Kannada paḷe and Tamil paḻaiya old; K1 padayya is the paired old-things adjective.',
121:'DEDR 5517 gives Kannada bisi and Tulu bisi heat, with Tamil vē hot; G1 bisi is an exact Kannada/Tulu-type form.',
125:'DEDR 5276 includes Tamil vala-k-kai right hand and valatu right side; G1 bala continues the right/strong side family.',
126:'DEDR 5276 includes Tamil valatu right side and Tulu balate right; K1 balate directly fits the latter.',
131:'DEDR 5276 explicitly lists Tamil valiya “strong, big”; K1 valiya preserves the same adjective and big sense.',
134:'DEDR 2049 has Kodagu koḍi “top” and Tamil koṭi banner/summit family; G1 koḍi “above” uses the spatial top sense.',
167:'DEDR 946 includes Tamil oṭi and Kannada oḍi “break”, with Malayalam oṭiyuka; G1 oḍanji “broken” retains the break root.',
178:'DEDR 600 includes Kannada umbu “eat” and Tamil uṇ “eat or drink”; G1 umbate uses the drink sense of this food-intake family.',
183:'DEDR 707 gives Tamil/Malayalam uṟakkam “sleep”; K1 orakkam has the same sleep noun under the elicited progressive gloss.',
185:'DEDR 1524 directly lists Malayalam kiṭakka “lie, rest”; K1 kiḍakunnu retains this verb with Malayalam-style progressive morphology.',
187:'DEDR 1835 lists Kannada kuḷḷu and Tulu kuḷḷuni “sit”; K1 kuḷu is the same sit stem.',
195:'DEDR 2132 includes Kannada kol (kond-) and Kodagu koll- (kond-) “kill”; K1 kod̃źa appears to contain the nasal past stem, with exact phonology for review.',
199:'DEDR 4087 lists Malayalam pāyuka “run, leap, flee”; K1 payinjɨ is a plausible inflected form of this running stem.',
203:'DEDR 4031 directly lists Malayalam paṟayuka “say, speak”; K1 paray preserves the Malayalam-type speech verb.',
205:'DEDR 2017 lists Tamil kēḷ (kēṭṭ-) and Malayalam kēḷkka “hear”; K1 keṭu retains the past hear stem.',
207:'DEDR 1443 gives Tamil kāṇ (kaṇṭ-) and Malayalam kāṇuka “see”; K1 kanḍu is the expected past see stem.',
218:'DEDR 5154 explicitly lists Kodagu eŋga “we” and Tamil nāṅkaḷ exclusive we; G1 eŋge resembles the eŋga pronoun.',
220:'DEDR 5154 explicitly lists Kodagu eŋga “we”; G1 eŋga is the same plural form despite its survey label “we (two)”.',
}
for ix in [51,52]:special[ix]='DEDR 5373 lists Kodagu baḷe/bāḷe plantain and Tamil vāḻai; DEDR 4004 lists Malayalam paḻam fruit. The survey expression has plantain + fruit in source order.'
for ix in [64,65]:special[ix]='DEDR 705 explicitly cites Tulu boḷḷ-uḷḷi and Tamil veḷḷ-uḷḷi garlic; DEDR 5496 supplies the white first component. Preserve white + onion/garlic in source order.'
for ix in [66,67]:special[ix]='The onion expression appears to combine nīr “water” (DEDR 3690) and uḷḷi “onion” (DEDR 705); the literal water + onion analysis remains tentative.'
special[145]='K1 patinonnuɨ is eleven: the ten base of DEDR 3918 followed by one of DEDR 990. The source-defined G1 pannandu is held because its second element is two.'
special[160]='DEDR 410 proximal idu plus DEDR 844 ellām “all” yields K1 iduellam “these”; preserve the two components.'
special[161]='DEDR 1 distal adu plus DEDR 844 ellām “all” yields K1 aduellam “those”; preserve the two components.'
def excerpt(pid,gloss):
 s=articles[pid]['Raw_Article']; s=re.sub(r'^\d+(?:\s*\([a-z]\))?\s*','',s)
 # Select a short portion matching the elicited sense when present; fallback to article opening.
 key=re.sub(r'[^A-Za-z]','',gloss.split(';',1)[0].split(' ',1)[0])
 m=re.search(re.escape(key),s,re.I) if len(key)>3 else None
 if m and m.start()>160:
  lo=max(0,m.start()-75);hi=min(len(s),m.end()+75)
  return ('…' if lo else '')+s[lo:hi].strip()+('…' if hi<len(s) else '')
 return (s[:195].rsplit(' ',1)[0]+'…') if len(s)>195 else s
accepted=[]
for ix in sorted(set(parents)|set(components)):
 rec=records[ix-1];parts=components.get(ix,[parents.get(ix)])
 evidence=special.get(ix)
 if not evidence:
  pid=parts[0]
  evidence=f"The full DEDR {pid[1:]} article records {excerpt(pid,rec['Gloss'])}. The {rec['Tags'].split(':')[-1]} form continues this article's {rec['Gloss']} sense."
 accepted.append({'record':rec,'parent':parts[0],**({'components':parts} if len(parts)>1 else {}),'kind':'component' if len(parts)>1 else 'reflex','citation':rec['Source']+';'+';'.join(f'dedr[entry {x[1:]}]' for x in parts),'evidence':evidence,'status':'pending-review','surveyIndex':ix})
contact={2,14,15,16,24,25,29,30,31,32,53,54,59,60,61,62,68,69,70,71,72,73,77,78,86,87,98,99,110,111,112,129,132}
complex={4,5,36,37,38,39,40,41,57,58,62,72,73,79,105,106,107,108,133,135,144,149,150,151,152,153,154,155,162,163,169,170,171,172,176,177,180,181,182,184,186,192,194,202}
mismatch={7,116,144,149,204}
reasons={}
for ix in range(1,224):
 if ix in parents or ix in components:continue
 rec=records[ix-1];label=f"{rec['Form']} ‘{rec['Gloss']}’"
 if ix in mismatch: reason=f"{label} has a source form/gloss conflict with the tempting article-level match; withhold pending source clarification."
 elif ix in contact:reason=f"{label} is regional/crop/contact vocabulary; a secure immediate donor and inheritance path have not been established."
 elif ix in complex:reason=f"{label} is multiword or morphologically complex, or has an unresolved elicited sense; its full components cannot yet be linked securely."
 else:reason=f"{label} has no verified DEDR article-level form-and-sense comparison in the current search and primary review."
 reasons[ix]=reason
# Targeted research notes for likely false friends and donor issues.
reasons.update({
1:'taḍi “body” resembles the taṭi “thick” family but DEDR 3020 does not substantiate the body noun.',
3:'tata “head” cannot be reduced securely to the talai family in DEDR 3103 without explaining the consonants.',
7:'ciri “mouth” resembles the ciri “laugh” family (DEDR 1562), not a mouth noun; hold the semantic mismatch.',
12:'cand “elbow” has no secure joint/elbow comparandum; exact cand matches in global search mean “moon” elsewhere.',
13:'cend “elbow” is the paired form, without a verified same-sense article.',
24:'koḍali “axe” occurs in DEDR appendix 32 with a CDIAL kuṭhāra comparison; the immediate donor remains unresolved.',
25:'Second koḍali “axe” dialect record; DEDR appendix 32 signals possible contact ancestry without an immediate donor.',
33:'kotti “star” has exact-form matches for crest, cat and pickaxe, but no verified star family.',
34:'Second kotti “star” dialect record; exact-form comparanda concern other meanings.',
40:'aru is elicited as path, six and who in one G1 record; these belong to distinct lexical families and need disambiguation.',
41:'aru is elicited as path and six in one K1 record; do not collapse the ambiguous form into DEDR 2485 six.',
59:'bane “eggplant” is a regional crop term with an unverified immediate donor.',
60:'vanaŋa “eggplant” is the paired regional crop term; donor path unverified.',
84:'olanji “spider” resembles Kodagu olañji “fly” (DEDR 1002), a different insect; hold.',
102:'dav “night” has no verified night article; global exact matches are unrelated.',
116:'paḷe is glossed “bad”, whereas exact Kodagu/Kannada paḷe in DEDR 3999 means “old”; do not equate the meanings.',
128:'kada “far” has exact matches with edge/side senses, but a distance relation is not established.',
133:'kanam kurayatɨ “light” appears to be “weight less” and cannot be linked as one inherited stem.',
144:'G1 pannandu is glossed “eleven” but appears to be ten + two (twelve); K1 patinonnuɨ provides a separate eleven response.',
149:'aḍi “where” is homographic with a source “below” form; an interrogative derivation is not secure.',
171:'keppa “many” has exact matches meaning fairness or deafness, not a secure many sense.',
173:'koźɨ “eat” has no matching eat article; do not force it under the lookalike kuḍi “drink”.',
204:'cepa “he hears” has no secure hear stem; linking it to the K1 kēṭ- form would ignore the consonant mismatch.',
206:'kamba “he saw” has no verified path to DEDR 1443 kāṇ (kaṇṭ-) without explaining -mb-.',
214:'agə “he” is a distal-looking G1 pronoun, but its exact relation to DEDR 1 av- is not established.',
216:'dige “she” lacks a secure comparison in the proximal or distal demonstrative articles.',
219:'aŋga “we (two)” may resemble DEDR 5154 eŋga, but person/number and initial-vowel contrasts remain unsettled.',
222:'agge “they” resembles the G1 distal pronoun agə, but the exact article-level morphology is unresolved.'})
assert set(parents)|set(components)|set(reasons)==set(range(1,224))
held=[{'record':records[ix-1],'reason':reasons[ix],'status':'research-hold','surveyIndex':ix} for ix in sorted(reasons)]
(P/'proposed-decisions.json').write_text(json.dumps({'status':'pending-review','accepted':accepted,'held':held},ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
(P/'unexamined-inventory.json').write_text('[]\n')
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
owners={x['record']['ID']:'proposed' for x in accepted};owners.update({x['record']['ID']:'held' for x in held});assert len(owners)==223
(P/'record-coverage-001.json').write_text(json.dumps({'source':'joseph2024kudiya','language':'Kudiya','residualSourceRecords':223,'proposedRecords':len(accepted),'heldRecords':len(held),'unexaminedRecords':0,'uniqueOwnership':True,'recordOwnership':owners},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'proposed':len(accepted),'rows':sum(len(x.get('components',[x['parent']])) for x in accepted),'holds':len(held),'articles':len(articles)}))
