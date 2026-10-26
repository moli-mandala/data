import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};priorheld={x['record']['ID'] for f in P.glob('pass*-decisions.json') for x in json.loads(f.read_text()).get('held',[])};a=[];held=[]
for r in rs:
 if r['ID'] in done:continue
 f,g,l=r['Form'],r['Gloss'],r['Language_ID'];x=None;reason=None
 if l=='Kho' and g=='stone' and f in ['boht','boṛt']:
  x=('11348','CDIAL[11348];bashir2023[s.v. boht]','Turner *varta round stone explicitly includes Khowar bort. Bashir documents boht with regional bohrt and bóxt variants and cites T11348. The survey boht/boṛt forms fit this family; the unexplained boṣt readings are excluded.')
 if g=='pestle' and f in ['baṭṭa','baṭṭā','bat t ā','bat t a']:
  x=('11348','CDIAL[11348]','Turner *varta round stone explicitly includes Bihari baṭṭā stone roller for spices, grindstone, beside Punjabi baṭṭā stone. The central/northern survey baṭṭa/baṭṭā pestles match this grinding-stone noun; source-separated t t transcriptions are retained unchanged. This is not vartaka 11349, whose cited reflexes mean clod, pill or egg.')
 if g=='stone' and f=='baṭṭo' and l=='mewari_basad':
  x=('11348','CDIAL[11348]','Turner *varta round stone includes Punjabi baṭṭā stone and western Pahari baṭṭ small round stone. Mewari baṭṭo stone preserves the diagnostic baṭṭ- stem with its ordinary western masculine vowel.')
 if l=='Kho' and g=='door':reason='Bashir s.v. duváht (variants duvahrt, duváxt) explicitly gives Old Persian (Morgenstierne 1936) versus possible Sanskrit (1973). Neither immediate donor nor inherited compound is resolved; resemblance to dvāra/vartman is insufficient.'
 if l=='Kal' and g=='man' and f in ['muc','mũc','c']:reason='Trail & Cooper p. 202 explicitly cite martya T9888 for moc man, but the full Turner entry marks Kalasha as possibly borrowed from Kati. Immediate ancestry remains unresolved; isolated c is also an apparently truncated source reading.'
 if l=='Kho' and g=='stone' and f=='boṣt':reason='Bashir documents boht/bohrt/bóxt stone under T11348, but not boṣt. A ṣ for h transcription issue cannot be silently treated as a sound change.'
 if l in ['Kal','Kho','Phal'] and g=='skin' and f in ['post','pʰost','pʰust','pūsto']:reason='Iranian skin family: Trail & Cooper s.v. post explicitly identify Persian borrowing, but the existing Persian parent is only an unlinked generic loan-source category, not an attested donor lexeme. Do not use that category or substitute remote *pōstikā.'
 if x:a.append(dict(record=r,parent=x[0],kind='reflex',citation=x[1],evidence=x[2]))
 elif reason and r['ID'] not in priorheld:held.append(dict(record=r,reason=reason))
(P/'pass115-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=1)+'\n');print('accepted',len(a),'held',len(held))
