import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
(['8209','10666','1031'],['pīyerīyɔ hɛ','pīre he','pīrīyɔ hɛ','pīryɔ cɛ','pīrɔ hɛ','pīrɔhe','pīrīyɔce','pīvɛ ryɔ hɛ','piryā hɛ','pī ṛehe he'],'The entire drinking progressive is drink + remain/progressive + present copula. Turner pibati supplies pī-/pīv-, *rahati the rah-/r- auxiliary, and ākṣeti the hai/che copula family. Fused spellings are retained and all three lexical constituents are represented.'),
(['8209','10666'],['pīryɔ','pīrahale'],'The full form is drink + remain/progressive, pī- + rah-/r-. Ordinary regional tense/person endings on rah- are kept with that verb; no independent copula is added where none is identifiable.'),
(['8209','1031'],['pīye he','pīve hɛ','pīyeṭ he','pīyaṭ he'],'The whole phrase consists of the drink verb, with its ordinary regional inflection, and the present copula hai/he. Turner pibati and ākṣeti provide those two families; the inflection is not treated as a separate derivational etymon.'),
(['8209','10895','1031'],['pīṇ legrī hɛ'],'The complete inchoative expression is infinitive drink + lag-/begin + present copula, with the feminine/participial marking on lag-. Turner pibati, lagyati and ākṣeti supply its lexical members. The r in lagrī is not split into an additional remain auxiliary without independent evidence.'),
(['13902','10666','1031'],['sorā hɛ','soryɔcɛ','sorɔhɛ','sorehahe'],'The complete sleeping progressive is so-/sleep + rah-/r- progressive + present copula. Turner svapati, *rahati and ākṣeti provide the three members; ordinary inflection and local vowel contractions are retained.'),
(['13902','10666'],['hūyɛrīyɔ'],'The full expression combines western hū-/sū- sleep with rah-/r- as the progressive auxiliary. Turner svapati and *rahati support the two lexical families. No extra copula is assumed in this response.'),
(['13479','1031'],['hūtɔ hɛ','sūtyɔ hɛ','sūtɔ hɛ','sūtɔ cʰɛ','sutyɔ he','sutyɔhɛ','sutɔ hɛ'],'The whole phrase is the sūt-/hūt- asleep stem plus the present copula. Turner supta explicitly supplies Old Marwari sūto and Old Gujarati sūtaü, separately from the svapati present stem; ākṣeti supplies hai/che. The western h forms are compared with that asleep family, without rewriting the source phonetics.'),
(['13479','10666'],['suti raɛ','sɔti rɔ','suti raɔ','sutirava','sut rahalak'],'The whole sleeping expression is the regional sut-/sot- verb developed from supta plus rah-/r- progressive. Turner explicitly gives Jaunsari sutṇu and the neighbouring sūt-/sut- verbs under supta; *rahati supplies the auxiliary and its ordinary endings.'),
(['13479','10895'],['suti laɠa'],'The full response is sut-/sleep + lag-/begin. Turner gives the Jaunsari sleep verb under supta, and lagyati supplies the inchoative auxiliary.'),
(['13902','10452'],['sojaū','soyjā','soyjāo','sojāv','soyījāo','soyjao','soyjāv','sojāo','sūija','hūijā','hūyjao','sojao','sojā'],'The complete imperative is sleep + go, a light-verb expression. Turner svapati gives so-/sū- sleep and yāti supplies jā-. Both members are linked, including normal imperative/person endings; no past gone stem is substituted for imperative jā-.'),
(['13902','4008'],['sūigyo','hūygiyo','sūigiyo','hūygyo'],'The full completed-action expression is sleep + gone, so-/sū-/hū- plus gay-/gy-. Turner svapati and gata provide the two members; the past auxiliary is kept distinct from the imperative jā- branch.'),
(['13479','4008'],['sutigelo','sʷit gʸɛle'],'The full phrase contains the sut- sleep verb, from supta, and the regional gel-/gyel- gone auxiliary from gata. Both members are represented; the survey spelling is preserved.'),
(['13902','1031'],['sɔlaīcʰe','sɔīla acʰe','soīla ace','sūā acʰe'],'The full eastern expression contains the so-/su- sleep verb, with ordinary regional past/finite inflection, plus the ach-/ache present copula. Turner svapati and ākṣeti support these lexical members. The l-past suffix is not detached as a second lexical verb.'),
(['3865','1031'],['kʰaīsī ace','kaīsī ace'],'The full drinking response uses the regional eat/consume verb khā- with a present copula ach-/ace. Turner khādati and ākṣeti supply those two members. The elicited drinking sense is preserved rather than silently relabelled eat.')]
a=[]
for i,(components,forms,evidence) in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and (r['Gloss'].startswith('sleep') or r['Gloss'].startswith('drink')) and r['Form'] in forms:
   a.append(dict(record=r,components=components,citation=';'.join(f'CDIAL[{x}]' for x in components),evidence=evidence,family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass079-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n')
(P/'pass079-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',[(g[0],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
