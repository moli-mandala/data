import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='12760-2',gloss=['tomorrow'],forms=['śuvi','ˈśui','ˈśuvi','śui'],citation='CDIAL[12760,2]',evidence='Turner specifically gives Western Pahari Padari śūī tomorrow under the śuvas branch, beside Pali suve and Prakrit suvo/suve. Kullui śui/śuvi matches this regional family, with variation in the glide; the unsyllabified śvas first branch is not selected.'),
 dict(parent='3104-2',gloss=['tomorrow'],forms=['kāⁱl','kāilh','kalike','kalīke','kalīkū','kalke','kāle','kal'],languages=['KochilaTharu','MagahiNepal','AdivasiOriya','hal','Goj','dog'],citation='CDIAL[3104,2]',evidence='Turner’s kalya dawn branch explicitly includes Punjabi/Lahnda kallh, eastern kāil/kālh, Assamese/Oriya kāli and Hindi kal for yesterday/tomorrow. These simple survey time words fit that family, including local locative/dative endings -ke/-ku; the survey tomorrow sense is preserved. Responses containing coming or next are excluded.'),
 dict(parent='3104',gloss=['tomorrow'],forms=['kal'],languages=['kul'],citation='CDIAL[3104,1 and addenda]',evidence='The first kālya branch explicitly gives Western Pahari Bhadrawahi/Bhalesi kāla tomorrow, with Kotgarhi kallε and Kotkhai kala in the addenda. The Kullui kal response is placed with this directly documented neighbouring Western Pahari series; the modern reduced vowel does not by itself distinguish the prehistoric long/short alternatives.'),
 dict(parent='11813',gloss=['tomorrow'],forms=['bihāno','vahaṇe','vahāṇe','vahṇe','vāhaṇẽ','vāhṇe','vāhṇā','vaṇẽ','vāhṇey','vaṇe','vāṇe','vāhāṇe'],citation='CDIAL[11813]',evidence='Turner gives eastern bihān/biyāna and Gujarati vahāṇũ/vāṇũ dawn, including morning adverbials. The survey tomorrow responses preserve this regional dawn/next-morning family. Turner also allows derivation via Middle Indo-Aryan vibhāyana brightening; this family placement leaves that internal formation alternative open and does not silently claim a unique reconstructed intermediate.'),
 dict(parent='11813',gloss=['morning'],forms=['binu','bʰinhyā','bʰinhik'],citation='CDIAL[11813]',evidence='Turner records Nepali biyāna, eastern bihān and Western Pahari bhɛṇi daybreak under the vibhāna family. The Danuwar/Tharu survey bin-/bhin- morning forms match this reduced regional stem with local endings. The dictionary’s alternative formation through vibhāyana is retained as unresolved within the family.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
for r in rs:
 if r['ID'] not in done and r['Gloss']=='mother' and r['Form'] in ['dāī','dāyī','dāi','ḍayī','ḍāi','ḍaⁱ','ḍāⁱ','daⁱ']:
  h.append(dict(record=r,reason='The mother form dāi is not safely settled by comparison with Hindi dhāī nurse (6774) or contracted dāi under the elderly-relative family dādda (6261). Both semantic extension and the loss/absence of medial d need regional evidence. Existing graph placement of another Tharu dāi is a discovery aid, not independent proof.'))
(P/'pass037-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass037-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
