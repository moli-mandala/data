"""Draft Irula Census article links from primary DEDR comparanda; never save links."""
import csv,gzip,json,re
from pathlib import Path
P=Path(__file__).resolve().parent; D=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==438
groups='''
6:3266;8:2891;12:5503;17:4035;18:1298;21:3690;28:131;34:4813;44,45:746;47:23;
53:4355;55:2818;57:2901;58:3650;59:2591;60:2248;65:2115;66:1711;69:4307;70:5155;71:2591;
78:3736;79:1448;80:1159;91:5352;93:5024;95:3633;96:3986;97:3584;100:2674;
103:3976;114:5028;115:382;118:3408;119:4004;120:3549;122:5504;123:3163;124:3753;125:1304;127:530;
138:1249;139,140:1787;144:1834;147:4322;149:5496;150:4635;151:3825;153:3793;161:4032;165:1372;174:486;
178:4950;179,182:474;180:5052;181:990;183:2826;184:910;185:784;186:1025;
188:741;190:2654;194:1957;198:77;199:4148;201:3570;203:1299;207:504;208:625;209:5549;211:4834;214:3398;216:3323;218,223:1624;224:3549;226:1850;230:2132;234:4065;235:5270;240:347;241:504;243:3071;244:4020;245,318:5336;247:2053;248:4572;252:1041;256:3687;260:3582;263:5093;264,268:2119;267:3583;270:5401;272:1124;273:76;279,305:2017;281:2855;283:5263;286:2426;293:3675;294:4834;298:793;306:1562;311:3480;314:4960;323:1443;326:1477;330:5320;332:3610;343:1670;348:3739;355:2654;357:3949;361:4922;364:3584;366:5218;369:3449;372:5020;383:4205;
392:410;395,416,417,418,419:5151;396,402:5160;403,415:3647;404:2559;406,409,410,411:1;422,424:3688;429:698;438:1182
'''
parents={}
for clause in groups.replace('\n','').split(';'):
 if not clause.strip():continue
 ids,number=clause.strip().split(':')
 for ix in ids.split(','):
  assert int(ix) not in parents,ix
  parents[int(ix)]='d'+number

articles={}
with gzip.open(D/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
 for article in csv.DictReader(fh):
  if article['Entry_ID'] in set(parents.values()):
   articles[article['Entry_ID']]={k:article[k] for k in ('Page','Citation','Raw_Article')}
   articles[article['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+article['Page']
assert set(articles)==set(parents.values()),set(parents.values())-set(articles)

def opening(pid):
 text=articles[pid]['Raw_Article'];m=re.search(r'\b(?:Ta|Ma|Ka)\. ',text)
 assert m,pid
 tail=text[m.start():]
 return (tail[:185].rsplit(' ',1)[0]+'…') if len(tail)>185 else tail

notes={
 8:'The source forest response has the Tamil cōlai “grove” sense, not a bare woodland root.',
 17:'Tamil paṉi includes dew, cold and snow; the survey selects the snow sense.',
 28:'The article gives Tamil aṇṇaṉ “elder brother”; the survey gloss conflates elder and younger, so the proposed stem applies only to the elder sense.',
 53:'The article has Tamil pūṉai “cat”; the source pūnɛ is the same regional animal term.',
  65:'Tamil kompu is “horn”; the survey kɔimpɯ retains the animal-horn sense.',
  70:'The article has Tamil āmai “tortoise”; the survey ɛmɛ is a vowel-shifted local response.',
  114:'Tamil mūṅkil is “bamboo”; the shorter survey muŋga preserves its nasal-velar base.',
  120:'DEDR 3549 also includes Tamil tōṭṭam “garden” beside the digging verb; the survey tōṭə names the plot.',
  150:'DEDR 4635 gives Tamil mañcaḷ “turmeric” and its yellow colour; survey məiñcə selects the colour.',
  178:'DEDR 4950 directly gives Tamil mutal “first”, the source mɯtəlɯ.',
  179:'DEDR 474 gives Tamil iraṇṭāvatu “second”; source rəiṇḍāvətɯ is its local ordinal.',
  180:'DEDR 5052 gives Tamil mūṉṟām “third”; source mūinvətɯ retains its three base with ordinal morphology.',
  181:'The whole DEDR 990 article includes Tamil oṉṟu “one” beyond its displayed oru head; source ɔinnɯ is a plausible local continuation.',
  184:'DEDR 910 gives Tamil ēḻu “seven”; source ēɻɯ matches.',
  190:'DEDR 2654 gives Tamil cuṭu “burn, heat”; the survey verb retains its stem and elicited -gətɯ ending.',
  194:'DEDR 1957 gives Tamil cey “do, make”; source sēkətɯ preserves the do base with its elicited ending.',
  203:'DEDR 1299 gives Tamil kala “mix”; source kələkɯətɯ is an extended mix verb.',
  207:'DEDR 504 gives Tamil iẓu/ir̤u “pull, drag”; source iɻɯkə preserves that verb.',
  218:'DEDR 1624 gives Tamil kīṟu “tear, cut”; the source kīkətɯ uses the short kī base.',
  223:'The same DEDR 1624 Tamil kīṟu article explicitly includes “cut”; source kikətɯ is its short base.',
  225:'DEDR 5527 gives Tamil vēṭṭai “hunt”; source vēṭəypɔkətɯ appears to add “go” to this hunt base.',
  235:'DEDR 5270 gives Tamil varu with past vant- “come”; source vəintɯṭɛ is a past arrival form.',
  240:'DEDR 347 has Tamil āṭu “dance”; source əṭəāḍɯgətɯ retains the dance stem amid extra material.',
  245:'DEDR 5336 gives Tamil vāṅku “receive, take, get”; source vəiŋgɯətɯ preserves the nasal-velar get stem.',
  304:'DEDR 55 gives Tamil añcu “fear”; source əiñgɯgətɯ is a less exact elicited fear verb for review.',
  383:'DEDR 4205 includes Tamil piṟaku “afterwards” within the posterior-time family; source pərəgɛ fits that adverb.',
  392:'The proximal demonstrative article includes Tamil iṅku “here”; source idikɛ is a local proximal locative.',
  395:'The interrogative article includes Tamil eṉṉa/eppaṭi how; source ɛinnɛ retains the e(n)- interrogative base.',
  403:'DEDR 3647 gives Tamil nām with oblique nam- “we/our”; source nəmɯ is the inclusive-plural stem.',
  404:'DEDR 2559 explicitly gives Tamil illai “not”; source illɛ is the local negative.',
  415:'DEDR 3647 gives Tamil nām “we” with nam- oblique; source nāmɯ is the matching inclusive response.',
  422:'DEDR 3688 gives Tamil nīm “you” plural and Kota nim-; source nimɯ matches the plural base.',
  424:'The same plural-you article has oblique nim-/num-; source nimɔḍɛ is a possessive extension.',
  429:'DEDR 698 is the interior/locative article: Tamil uḷ “inside”, Malayalam uḷḷu “inside”, and Kota uḷ “in”; source ɯiḷḷɛ retains the locative.',
  438:'DEDR 1182 explicitly gives Tamil kaṇṇāṭi “mirror”; source kəiṇṇāḍi matches it.',
}
accepted=[]
for ix,pid in sorted(parents.items()):
 rec=records[ix-1]
 evidence=f'DEDR {pid[1:]} full article: {opening(pid)} '
 evidence+=notes.get(ix,f'The Irula survey {rec["Form"]} “{rec["Gloss"]}” retains this article’s corresponding form and sense; contact transmission remains open to review.')
 accepted.append({'record':rec,'parent':pid,'kind':'reflex','citation':rec['Source']+';dedr[entry '+pid[1:]+']',
                  'evidence':evidence,'status':'pending-review','surveyIndex':ix})

special={
 1:'gāɻi “air” resembles Kannada gāḷi and Tulu gāli in DEDR 1499, but no Tamil or Malayalam reflex is recorded there; the immediate donor into this survey lect remains unresolved.',
 2:'būḍī “ashes” may be a southern ash/dust term, but no same-form article and immediate donor has been verified.',
 4:'būmi “earth” is a widespread Sanskrit-type form; an immediate donor cannot be chosen from the survey.',
 23:'lɛsənəgāli “wind” contains the gāli wind term of DEDR 1499, but the first element and donor path are unresolved.',
 25:'pəiccəpɯiḷḷɛ “baby” contains piḷḷai “child” (DEDR 4198), but the first component is unanalysed.',
 62:'kəḍəy “goat” resembles DEDR 1123 kaṭāy, which specifically denotes a male goat/ram; the source does not fix sex.',
 68:'pəikkikūḍɯ “nest” appears to be bird + nest, but the first component has no securely assigned article.',
 101:'ərisi is assigned both uncooked and cooked rice in one elicitation gloss; DEDR 215 arici is specifically uncooked rice, so clarify the source sense first.',
 145:'rɯsi is glossed “hot” here but “taste” in another source record; its Indo-Aryan-type taste root cannot resolve the hot sense.',
 155:'rɯsi “taste” resembles widespread Sanskrit rasa and regional ruci; the immediate donor is not established.',
  177:'vɛḷi “silver” resembles Tamil veḷḷi, while DEDR 5496 is a white-colour family; the metal noun’s donor and article require separate verification.',
  189:'mɯḷikədɯ “break” resembles Tamil muṟi in DEDR 5008 but the survey lateral and morphology leave more than one possible root.',
  212:'ḍɛikkətɯ “shut” has no secure relation to DEDR 5034 mūṭu “shut”, despite the matching gloss.',
  225:'vēṭəypɔkətɯ “hunt” appears to combine vēṭṭai “hunting” (DEDR 5527) with pō “go” (DEDR 4572), but the full component boundary is not verified.',
  213:'mūiñgətɯ “sink” resembles Tamil mūḻku (DEDR 4993) but the nasal transcription and exact verb are unresolved.',
  257:'ḍūri “swing” is not a secure form match to the Tamil āṭu swing/dance family in DEDR 347; hold this sense-only comparison.',
  269:'vāḷɯ is a “saw” noun in DEDR 5376, while the source elicits “to saw”; verbal use needs confirmation.',
  299:'kāṭɯkɯdɯ “know” may use kāṇ/kaṇṭ- “see” (DEDR 1443) with a semantic extension; the compound morphology remains unresolved.',
  304:'əiñgɯgətɯ “fear” may relate to Tamil añcu (DEDR 55), but the survey nasal-velar sequence does not securely identify that root.',
  300:'pəḍikətɯ “learn” and its “read” counterpart look like a regional paṭi “study/read” form; no secure DEDR root or donor was established.',
  307:'pəḍikətɯ “read” repeats the regional paṭi form; do not force it under the unrelated DEDR 3853 paṭu.',
  397:'ətɯ is glossed “it/this” but appears distal a- rather than proximal i-; hold the deictic contradiction.',
  398:'ətɯkɯ “to it/to this” inherits the unresolved proximal/distal contradiction of ətɯ.',
  413:'əvəkɛ is glossed “these” but has a distal av- form; hold the demonstrative conflict.',
  430:'ɔli “wave” is too short to assign to DEDR 1003 olku “wave” without separating it from sound-related oli homographs.',
  434:'pɛinḍɛli “mouse” contains eli “rat/mouse” (DEDR 833), but its first component remains unexplained.',
}
held=[]
for ix,rec in enumerate(records,1):
 if ix in parents:continue
 label=f'{rec["Form"]} “{rec["Gloss"]}”'
 if ix in special:reason=special[ix]
 elif len(rec['Form'].split())>1 or len(rec['Form'])>14:
  reason=f'{label} is multiword or morphologically extended; a complete source-order component analysis has not been verified.'
 elif 102<=ix<=137 or 156<=ix<=177:
  reason=f'{label} is crop, occupational, material or institutional vocabulary; an immediate donor and inherited path have not been established.'
 elif 389<=ix<=437:
  reason=f'{label} is an inflected pronoun, deictic or function word; its exact paradigm and article derivation need further review.'
 else:reason=f'{label} has no verified form-and-sense correspondence in the full DEDR articles examined.'
 held.append({'record':rec,'reason':reason,'status':'research-hold','surveyIndex':ix})
assert len(accepted)+len(held)==438
owners={x['record']['ID']:'proposed' for x in accepted};owners.update({x['record']['ID']:'held' for x in held})
assert len(owners)==438
def write(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
write('proposed-decisions.json',{'status':'pending-review','accepted':accepted,'held':held})
write('research-holds.json',held);write('unexamined-inventory.json',[])
write('primary-evidence.json',articles)
write('record-coverage-001.json',{'source':'census2023tamilnadu','language':'Irula','residualSourceRecords':438,'proposedRecords':len(accepted),'heldRecords':len(held),'unexaminedRecords':0,'uniqueOwnership':True,'recordOwnership':owners})
print(json.dumps({'proposed':len(accepted),'held':len(held),'articles':len(articles)}))
