import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='6251',gloss=['right'],languages=['Dang','Kathoriya','Sunha','Chitwan','Hajong','KochilaTharu','MagahiNepal','Majhi','B','kul'],forms=['dāhīnā','dahīna','dahīn','dāhīn','dahīnī','dan','dahina','dahinā','dāihine','ḍan','ɖan','dahuṇə','doĩnə','dõĩnə','daĩṇə','dā̃ṇə','dʰəĩṇə','ḍāhɛn','dāhina'],citation='CDIAL[6251]',evidence='Turner distinguishes dākṣiṇa from dakṣiṇa and gives Prakrit dāhiṇa/ḍāhiṇa, Hindi da(h)inā, Maithili dahīn/dahinā, Nepali dāinu, Kumaoni daiṇo and Assamese/Bengali ḍāhin. These survey right-hand forms match the h-bearing or contracted dāin-/ḍān- series, with ordinary agreement vowels and source length notation. The distinct northern affricate series is assigned to dakṣiṇa instead.'),
 dict(parent='6119',gloss=['right'],languages=['Kal','Phal','Gaw','Bshk','Mai','Chil','Gowro'],forms=['draʦ̣vī','draʦ̣','drạʦ̣','dɛcīnī','dācīné','daśtīm','lāśīm','lāʦ̣ūm','lāʦ̣īm','lāśīn','daʦ̣īū̃','dacīū̃','dacīū','dācūī','daʦ̣ū̃'],citation='CDIAL[6119]',evidence='The dakṣiṇa article specifically gives Palula dechiṇi, Kohistani Shina daṣuṇu, Kalasha drä̃cũri with intrusive r, and Bshk lāchim remodeled with paścima. These survey northern affricate/sibilant right forms fit those named regional series; Bshk l-initial forms preserve the documented analogical mixture. This is a qualified regional-family assignment and does not erase source transcription differences or claim every shortened ending is regular.'),
 dict(parent='11533',gloss=['left'],languages=['kul'],forms=['bõuə','bʰõvã','bʰaĩyə','bãhu','bõvã'],citation='CDIAL[11533]',evidence='Turner gives western Pahari bā̃o/bauā̃, Kumaoni bāyõ and Hindi bā̃wā̃/bāyā̃ under vāma left. These Kullu bã-/bõ- left forms agree with that nasalized, weakened-medial series. The spelling of initial aspiration is retained without asserting it to be an independently established sound law; m-retaining forms elsewhere are excluded from this inherited series.'),
 dict(parent='5539-5',gloss=['left'],languages=['Bhatri','hal','AdivasiOriya'],forms=['ḍebṛi','ḍeba','ḍebri'],citation='CDIAL[5539,5]',evidence='The fifth *ḍēvva branch explicitly gives Oriya ḍebarā/ḍebiri and Bengali ḍebrā left-handed, alongside Nepali ḍebre left. These retroflex-ḍ eastern survey ḍeb-/ḍebr- forms match that branch, with the local feminine/adjectival final i. Dental-d forms are kept on Turner’s separate eighth branch.'),
 dict(parent='5539-8',gloss=['left'],languages=['Hajong','Bhatri','AdivasiOriya','Bote'],forms=['debra','debreā','debṛi','debrī'],citation='CDIAL[5539,8]',evidence='Turner separates dental *dēvva, with Nepali debre and Hindi debrā left, from retroflex *ḍēvva. These survey dental-d debr- forms fit the eighth branch; vowel length and the final adjectival vowel vary locally. Forms with initial l/n, a different stem vowel, or an additional word are not silently collapsed into this set.'),
 dict(parent='10896-5',gloss=['small','short'],languages=['Phal','poth','Goj'],forms=['lōko','loko','lokā','loka'],citation='CDIAL[10896,extension kk]',evidence='The kk expansion of laghu explicitly includes Palula lohōko small/young and Punjabi lauhkā small/young, as well as Domaaki lōka and Bshk lōk. These short/small survey lok- adjectives fit that regional light/small family. They are not assigned to the bare laghu node or the separate metathesized halkā series.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and r['Language_ID'] in g['languages']:a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
for r in rs:
 if r['ID'] not in done and r['Gloss']=='path' and r['Form'] in ['baṭ','bato','bat','vāṭh','vat','vāṭe','bātʰ','batʰ','bətʰ','ˈbotʰə','bʰot']:
  h.append(dict(record=r,reason='CDIAL 11366 explicitly allows some feminine path words to derive equally well from vartis (11363), rather than vartman. The survey does not provide the gender/morphological evidence needed to settle that alternative; hold these bare bāṭ/vāṭ-family forms.'))
(P/'pass029-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass029-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
