import json
from pathlib import Path
P=Path(__file__).resolve().parent
hs=json.loads((P/'pass117-decisions.json').read_text())['held'];a=[]
groups=[
(['1697-2','10666','1031'],['ūḍɛrīyɔ hɛ','ūḍra hɛ','ūḍrɔ hɛ','uḍryɔcʰɛ'],'The complete progressive is fly + rah-/r- remain/progressive + hai/che copula. Turner *rahati 10666 supplies the progressive stem; ākṣeti 1031 explicitly unites the hai and che copula series. Contractions and ordinary inflection remain as printed.'),
(['1697-2','10666'],['ūḍɛrīyɔ','ūḍɨrɔ'],'The full response contains fly plus rah-/r- as the progressive auxiliary, with no overt copula. The retained r followed by the regional perfective/progressive ending is analysed consistently with the overt riyɔ hai responses.'),
(['1697-2','1031'],['ūḍḍɛ hɛ','uḍḍɛ hɛ','uḍɛ hɛ','uḍaṭʰ he','uḍeṭa he','ūdte acʰe','ūdsī ace','ūḍsī ace'],'The complete response contains the inflected fly stem and a present copula. Turner ākṣeti explicitly gives Hindi hai, Gujarati che and Oriya achi; the western he and eastern ache/ace spellings match those copula families. Ordinary participial/finite endings remain on the fly stem.'),
(['1697-2','10895'],['uḍay lagn','uḍay lago','uḍyaṇalagi','uḍeṭ lagɛ'],'This complete inchoative expression consists of fly (infinitive/participle) plus lag-/begin. Turner lagyati explicitly includes western Pahari lagṇū, Nepali lāgnu and Hindi lagnā in the begin sense. No extra remain or copula is added to the recorded expression.'),
(['1697-2','10895','1031'],['ūḍīaṇɨ lāgrī hɛ'],'This complete expression is infinitive fly + lag-/begin with regional feminine/participial marking + hai copula. As in the documented regional drink-begin construction, lagrī is kept with lag- and not split into an unsupported additional rah auxiliary.'),
(['1697-2','4008'],['ūṛigala','ūḍūgalī','ūḍīgī','ūḍīgay','ūḍiṭgɔyo','ūḍigoinū'],'The complete expression has fly followed by the perfective gone auxiliary. Turner gata includes western gaya/gyau/giu and the extended Oriya gali/gelā series. Local linking and agreement endings are retained with their stems.'),
(['1697-2','10452'],['ūḍijā','ūḍ, ūḍijā','ūḍiṭjā'],'The complete imperative combines fly with jā go, from Turner yāti. Where a bare imperative is paired with the expanded imperative, the go component belongs to the expanded answer only.'),
(['1697-2','10452','4008'],['ūḍjā, ūḍgayɔ','uḍjā, uḍgyo','uḍjā, uḍgɛyo','uḍjā, uḍī gɛyo','ūḍī dza, ūḍī gyo','ūḍijā, ūḍigayo','uḍijā, uḍigoyo'],'The source pairs imperative fly + jā/dza go with past fly + gayo/gyo gone. Ordered edges represent the shared fly family followed by the two distinct go stems in response order; they are alternatives across the comma, not a three-verb phrase. Turner yāti and gata explicitly document their regional imperative and past series.')]
for components,forms,evidence in groups:
 for h in hs:
  r=h['record']
  if r['Form'] in forms:a.append(dict(record=r,components=components,citation=';'.join('CDIAL['+x+']' for x in components),evidence='User adjudication fixes the fly lexical parent as uḍḍayate 1697-2, with *uddayati 1697-5 retained as the alternative. '+evidence))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass119-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n');print(len(a),'records',sum(len(x['components']) for x in a),'rows')
