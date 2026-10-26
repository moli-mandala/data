import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='12548-2',gloss=['dry'],languages=['hal','RathwiBareli','Bhilali','Bhili','PalyaBareli','PauriBareli','mewari_basad','Nimadi','Noiri','Vasavi'],forms=['suklo','hūklo','sūklo','sūkalɔ','sūkhlo','xūkalū','sūkhḷā','hūkalū','hūkelū','hukhalu','hūkala','hūkḷa','hūkalo','hūkaḷa','sūkʰel','sūkailā','hukālo','hukālõ','hukalo','hukāila','hukla','huklo','hukala'],citation='CDIAL[12548,extension ll]',evidence='Turner explicitly distinguishes an ll extension of śuṣka and cites Oriya sukhilā dried/withered alongside Torwali šugil. These western and central survey sūkh-l/hūk-l adjectives preserve both the dry stem and the l expansion, with ordinary final vowels and western s-to-h outcomes. Compound gone/dried expressions, separate copulas and prefix-bearing forms are excluded; this is a comparative extension-family link, not a detailed reconstruction of every local ending.'),
 dict(parent='12548',gloss=['dry'],languages=['kaithal','jaun','kul','poth'],forms=['sokkʰā','çuko','çuka','sukyɔ','śukhiyẽ','śokʰə','śəkə','sūkʸā'],citation='CDIAL[12548]',evidence='Turner lists Punjabi sukkā, Jaunsari sūkhō and the Kullu-area śukkhɔ/śūkho forms under the plain śuṣka dry adjective. These unexpanded survey forms retain that regional stem with ordinary agreement vowels and source sibilant notation. l-bearing expanded adjectives and whole verbal phrases are assigned separately or held.'),
 dict(parent='10299',gloss=['sweet'],languages=['Hajong'],forms=['mitʰu'],citation='CDIAL[10299]',evidence='The sweet branch of mr̥ṣṭa/miṣṭa explicitly gives Assamese miṭhā and Bengali miṭha; the addendum records Assamese phonetic mitha/mita. Hajong mithu therefore fits the eastern sweet series with its source dental spelling and final u, without inventing a separate root from the dental notation.'),
 dict(parent='5806',gloss=['bitter'],languages=['Hajong'],forms=['titu'],citation='CDIAL[5806]',evidence='Turner gives Prakrit titta, Assamese titā and Bengali tita/titā bitter under tikta. Hajong titu preserves this dental t-t bitter stem with regional final u; it is not assigned to the separate retroflex bitter family which Turner discusses as an alternative for Torwali.'),
 dict(parent='10729',gloss=['empty'],languages=['Dotyali'],forms=['rito'],citation='CDIAL[10729]',evidence='Turner explicitly lists Kumaoni rīto and Nepali ritto empty under rikta, and discusses emphatic doubling in Himalayan reflexes. Dotyali rito matches this whole regional empty adjective; the attestation’s short vowel and singleton spelling are preserved.'),
 dict(parent='5850',gloss=['empty'],languages=['Dang'],forms=['cʰucʰa','cuce'],citation='CDIAL[5850,1]',evidence='The first tucchya branch includes Prakrit chuccha, Bhojpuri chũch, Awadhi chũch/chūchā and Hindi chūchā empty. These Dang chucha/cuce forms fit that affricate series and final-vowel variation; the distinct northwestern tuśya branch is not used.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and r['Language_ID'] in g['languages']:a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
for r in rs:
 if r['ID'] in done:continue
 reason=None
 if r['Gloss']=='wet':
  if r['Form'] in ['sinna','sinnā','sina','ˈsunnə','ˈsinə','śunnə','sinə','śunni','sino']:
   reason='Turner explicitly cross-refers competing snih (13798) and svinna (13934) analyses for the regional sinnā wet family; these surface forms do not settle the choice.'
  elif r['Form'] in ['sijyā','sijā','sijʸā']:
   reason='Both sicyate (13393) and svidyati (13933) are offered by Turner for regional sijj- wet forms; the Pothwari survey spelling does not resolve the historical root.'
  elif r['Form'] in ['oda','oḍ','oḍī','oḍa','oḍʰ','oḍʰe','oḍʰī']:
   reason='Turner 2554 gives *odra/*audra but explicitly allows all these wet forms to come instead from odman, especially because of the Hindi/Oriya nouns. Hold the competing parent analysis.'
 if reason:h.append(dict(record=r,reason=reason))
(P/'pass025-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass025-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
