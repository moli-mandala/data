"""Lindgren Tamil article-level etymology draft; pending review only."""
import csv,gzip,json,re
from pathlib import Path
P=Path(__file__).resolve().parent
R=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text()); assert len(records)==244
mapping='''1,2:3684;3,5,9,11,13,15,17:410;4,7,10,12,14,16,18,214:1;23,201:2041;24:844;25:5423;26:864;27,123:2023;28:2453;29:4955;30:5028;32:1832;33:77;34:274;35:5528;36:4205;37:5259;38:5314;39:4411;40,41,100:4020;42:1124;43:1249;44:1395;47:741;48:5450;49:839;50:3939;54,55:811;58:2149;59,60,109:1594;61:916;62:1293;64:5270;65:2342;66:282;67:2426;68:3549;69:1957;70:3650;71:3975;73:1448;74,188:4666;75:4312;77:4939;78:784;79:1159;80:5430;81:3519;83:2146;84:156;85,240,241:2591;86:1851;87:5258;89:2929;90:3266;91:5440;92:4885;93:2826;94:529;97:4841;98:4345;101,102:1479;104:1438;105:3655;106:4004;107:3682;108,242:4395;110,111:2053;112:4572;113:2781;115:3610;116,117,159:4410;118:4300;119:3821;120:228;121:1652;122:4707;125:3103;126:2017;129:410;131:946;132:2115;133:5393;235,236,237:5151;135:1173;136:5160;137:698;140:1204;142:1562;143:497;144:3559;147:3692;148:4449;150:3987;152:4719;154:183;155:364;156:4742;157:5352;158:295;161:1366;162:4275;163:2552;164:1025;165:5024;167:23;168:990;170:2471;171:399;172:672;173:4039;174:5513;175:504;176:4753;177:1931;179:215;180:5159;181:5535;182:1254;184:665;185:1041;187:2674;189:2855;190:4091;191:5401;192:910;193:3679;194:1670;195:3564;196:2485;197:2594;198:4240;199,200:4085;202:4715;203:3323;204:3675;206:1298;207:4121;208:709;210:530;212:851;213:3918;215:3020;216:3302;217:5078;218:5052;219:859;220:1147;222:3774;223:3633;224:3986;225:4711;226:474;229:1369;230:3690;231:3045;232:5154;233:3647;234:4581;238:8;239:4776;243:3758;244:3688'''
parents={}
for clause in mapping.split(';'):
 indexes,num=clause.split(':')
 for ix in indexes.split(','): parents[int(ix)]='d'+num
components={21:['d1','d295'],22:['d410','d295'],31:['d4711','d3876'],134:['d5151','d295'],139:['d4990','d1479'],146:['d1298','d546']}
holds={
6:'ivar is the animate/honorific proximal form in DEDR 410; the source labels it inanimate plural.',
8:'avar is the animate/honorific distal form in DEDR 1; the source labels it inanimate plural.',
19:'vetalai betel leaf is shared crop vocabulary without a verified immediate donor or DEDR article.',
20:'man̪t̪aman̪a slough has no secure matching lexical family.',
45:'irattam blood is Sanskrit rakta-type contact vocabulary; immediate donor not established.',
46:'aṭi means strike or beat in DEDR 77; the unusual elicited sense blow with mouth is not supported.',
51:'mārpakam breast seems based on mārpu in DEDR 4818, but the whole derived formation and suffix need confirmation.',
52:'mūccuviṭu breathe combines mūccu breath (DEDR 4886) with viṭu release; whole multiword analysis remains unresolved.',
53:'pālam bridge has an unresolved contact history and immediate donor.',
56:'eṭuttuccellu carry combines eṭu take and cel go, but the elicited whole verb needs analysis beyond a single inherited root.',
57:'Second independent eṭuttuccellu carry record; the whole compound verb remains to be analysed.',
63:'cīmaimiḷakāy cold is literally a crop term, not an established temperature adjective; elicitation issue likely.',
72:'mattaḷam drum has an unresolved donor path.',
76:'cāppiṭu eat is an established verb but its relation to older eat stems and compound structure is uncertain.',
82:'t̪ōram far resembles regional dūra vocabulary; immediate donor unverified.',
88:'viraln̪akam fingernail is an analyzable compound, but nail n̪akam has unresolved borrowing ancestry.',
95:'sad̪ai flesh has unresolved spelling/parent and may be contact vocabulary.',
96:'catai flesh has unresolved spelling/parent and may be contact vocabulary.',
99:'ī fly is not a secure match for DEDR 4020 paṟa; no separate matching article established.',
103:'pātam foot is Sanskrit pāda-type vocabulary; immediate donor unverified.',
114:'yeru in DEDR 916 means ascend/climb, whereas source gloss says descend; hold the contradictory record.',
124:'kaṭiṇāmaṇa hard has kaṭiṇa-type contact morphology with immediate donor unresolved.',
127:'itayam heart resembles Sanskrit hṛdaya; immediate donor unresolved.',
128:'palamāṇa heavy is not securely segmented or matched to an inherited heavy adjective.',
130:'oɻittal hide may be causative/derived, but no exact parent article or sense verified.',
138:'vaittiru keep contains a put/keep verb plus auxiliary, not a single inherited stem.',
141:'arin̪tiruttal know contains an auxiliary after an apparent know verb; whole formation unanalysed.',
145:'mōdi lid lacks a secure DEDR article or immediate donor.',
149:'manitan man may be a manuṣya-type loan; immediate donor remains unverified.',
151:'kalyāṇam marriage is Sanskrit-type contact vocabulary; immediate donor remains unverified.',
153:'araivaikkal mortar appears compound-like but source segmentation and parent of second element are unclear.',
160:'toppuḷ navel resembles DEDR 4460 pokkuḷ but initial t- versus p- needs verification.',
166:'vayatāṇa old is a complex age expression rather than an isolated inherited adjective.',
169:'pātai path is contact vocabulary with no verified immediate donor.',
178:'uravinarakaḷ relatives has a plural and relational suffix; full formation is not analysed.',
183:'karaṭumūraṭāna rough is a complex adjective, with reduplicative/compound formation not resolved.',
186:'tiyākam sacrifice is Sanskrit-type contact vocabulary; immediate donor unresolved.',
205:'n̪aṭchattiram star is nakṣatra-type contact vocabulary; immediate donor unresolved.',
209:'cūriyaṇ sun is sūrya-type contact vocabulary; immediate donor unresolved.',
211:'vāl tail appears to have a DEDR appendix comparison without a usable persistent etymon node.',
221:'kālam time is shared contact vocabulary; immediate donor unresolved.',
227:'vān̪tiyēṭu vomit contains a take/do element beyond the noun; whole expression needs analysis.',
228:'en̪ttiri wake up is morphologically complex and lacks a verified parent article.'}
all_ix=set(range(1,245)); assert not (set(parents)&set(components)); assert not ((set(parents)|set(components))&set(holds)); assert set(parents)|set(components)|set(holds)==all_ix, sorted(all_ix-(set(parents)|set(components)|set(holds)))
article_ids=set(parents.values())|{p for ps in components.values() for p in ps}
articles={}
with gzip.open(R/'data/dedr/entry-texts-audit.csv.gz','rt') as f:
 for row in csv.DictReader(f):
  if row['Entry_ID'] in article_ids:
   articles[row['Entry_ID']]={k:row[k] for k in ('Page','Citation','Raw_Article')}
   articles[row['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+row['Page']
assert article_ids==set(articles),sorted(article_ids-set(articles))
# Evidence cites an attested Tamil segment of the full article; article units include more than their display head.
def ta_excerpt(pid, gloss=None):
 s=articles[pid]['Raw_Article']; s=re.split(r'\s(?:Ma|Ir|Ko|To|Ka|Koḍ|Tu|Te|Kol|Nk|Pa|Ga|Go|Konḍa|Kui|Kuwi|Br)\.\s',s,maxsplit=1)[0]
 s=re.sub(r'^\d+(?:\s*\([a-z]\))?\s*Ta\.\s*','',s)
 if gloss:
  semantic=gloss.split(' (',1)[0].split(' or ',1)[0].split(' ')[0]
  if semantic not in {'third-person','second-person','we','i','you','what','who','where'}:
   m=re.search(r'\b'+re.escape(semantic)+r'\b',s,re.I)
   if m and m.start()>160:
    start=max(0,m.start()-75);end=min(len(s),m.end()+75)
    return ('…' if start else '')+s[start:end].strip()+('…' if end<len(s) else '')
 return (s[:220].rsplit(' ',1)[0]+'…') if len(s)>220 else s
accepted=[]
for ix in sorted(set(parents)|set(components)):
 rec=records[ix-1]
 if ix in components:
  ps=components[ix]; kind='component'; desc={'21':'distal demonstrative + measure/extent','22':'proximal demonstrative + measure/extent','31':'tree + bark','134':'interrogative + measure/extent','139':'knee/joint + leg','146':'stone + liver'}[str(ix)]
  evidence=f"The survey expression is {desc}. DEDR {ps[0][1:]} records {ta_excerpt(ps[0])}; DEDR {ps[1][1:]} records {ta_excerpt(ps[1])}."
 else:
  ps=[parents[ix]];kind='reflex'
  evidence=f"The full DEDR {ps[0][1:]} article includes Tamil: {ta_excerpt(ps[0],rec['Gloss'])}. This source form matches the article's Tamil stem and the survey's {rec['Gloss']} sense."
  if ix==97: evidence+=' DEDR marks its Tamil *mita* float comparison with a question mark; the proposed link remains tentative.'
  if ix==48: evidence+=' DEDR specifies wind as the blowing agent, so the source’s mouth-agent gloss is a semantic qualification to review.'
  if ix==234: evidence+=' The DEDR sense is put/fasten; source wear is a contextual extension.'
  if ix==202: evidence+=' The article explicitly gives maru-makaṉ as son-in-law.'
  if ix==208: evidence+=' It explicitly gives uṟiñcu “sip, suck up.”'
 accepted.append({'record':rec,'parent':ps[0],**({'components':ps} if len(ps)>1 else {}),'kind':kind,'citation':rec['Source']+';'+';'.join(f'dedr[entry {p[1:]}]' for p in ps),'evidence':evidence,'status':'pending-review','surveyIndex':ix})
held=[{'record':records[ix-1],'reason':reason,'status':'research-hold','surveyIndex':ix} for ix,reason in sorted(holds.items())]
(P/'proposed-decisions.json').write_text(json.dumps({'status':'pending-review','accepted':accepted,'held':held},ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
(P/'unexamined-inventory.json').write_text('[]\n')
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
owners={x['record']['ID']:'proposed' for x in accepted};owners.update({x['record']['ID']:'held' for x in held});assert len(owners)==244
(P/'source-coverage-001.json').write_text(json.dumps({'source':'lindgren2023dravidian','language':'Tamil','compiledSourceRecords':244,'proposedRecords':len(accepted),'heldRecords':len(held),'unexaminedRecords':0,'uniqueOwnership':True,'recordOwnership':owners},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'proposed':len(accepted),'rows':sum(len(x.get('components',[x['parent']])) for x in accepted),'holds':len(held),'articles':len(articles)}))
