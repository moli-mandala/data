import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
forms={r['ID']:r for r in csv.DictReader(open('cldf/forms.csv'))}
assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'}
assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('f_napwmsybm5wuo','12548',['f_4ipze6tm4oqrs','f_akrefevdo26lw'],'śuṣka','CDIAL12548 explicitly gives European Romani šuko dry, thin, beside Pali sukkha and Prakrit sukkha/sukka. Zargari śukho belongs to this inherited family; aspiration is retained in the source spelling.','3.15.2','factitive -ar, perfect -ard'),
('f_kg2m4a365nzf2','4780',['f_6hmstiu7nssfg','f_p7liplb7s7ct6'],'cikka','CDIAL4780 explicitly gives European Romani čik mud, dirt, clay, ashes. Link Zargari cik to this Indo-Aryan cikka family. Turner discusses competing Dravidian/Munda origins of the older family; this does not establish a direct Zargari borrowing from either, and no such donor claim is made.','3.15.2','factitive -ar, perfect -ard'),
('f_y2lqcnagnxmgi','9271',['f_k45442wodvboa','f_4iashqwqfjrto'],'*buḍḍha²','CDIAL9271 explicitly lists European Romani phuro old under *buḍḍha². Retain Turner’s alternatives for the deeper origin and influence from MIA vuḍḍha; do not flatten this onto Sanskrit vṛddha.','3.16.1','mediopassive -iv, perfect -il'),
('f_ke6ydubncm3qu','3690',['f_ym2t2ofhfgfsm','f_ofvyvvwwpidnk'],'kṣīṇa','CDIAL3690 explicitly gives European Romani khino tired beside Pali khīṇa exhausted and Prakrit khīṇa tired/weak. CDIAL3884 explicitly assigns Romani khino to kṣīṇa rather than khinna because it has kh-, not x-. The Zargari aspirated initial agrees with that distinction.','3.16.1','mediopassive -iv, perfect -il')]
a=[];table=[]
for n,(base,parent,children,head,evidence,section,morph) in enumerate(groups,1):
 r=forms[base]; assert base not in assigned
 rec={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']}
 citation=r['Source']+f';CDIAL[{parent} and addenda]'+(';CDIAL[3884]' if parent=='3690' else '')
 a.append(dict(record=rec,parent=parent,kind='reflex',citation=citation,evidence=evidence))
 for child in children:
  r=forms[child];assert child not in assigned
  rec={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']}
  a.append(dict(record=rec,parent=base,kind='derived',citation=r['Source']+f';rezaibaghbidi2003zargari[p. 141, section {section}]',evidence=f'The primary grammar p. 141, section {section} explicitly derives this form from {forms[base]["Form"]} ({forms[base]["Gloss"]}) using {morph}. Both ordinary stems of the derived verb point to the existing same-source base {base}; its inherited ancestry is established separately in this pass.'))
 display=', '.join(forms[x]['Form'] for x in children)
 table.append(f'| {n} | **{forms[base]["Form"]}** ‘{forms[base]["Gloss"]}’; **{display}** ‘{forms[children[0]]["Gloss"]}’ | Base: **{head.replace("*",chr(92)+"*")}**, CDIAL **{parent}**; verbs **derived** from the Zargari base | {evidence} The grammar p.141 §{section} explicitly specifies {morph}. |')
assert len(a)==12
(P/'pass032-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=2)+'\n')
locs={x['record']['Source'] for x in a}
aud=[r for r in csv.DictReader(open('data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv')) if r['Source'] in locs]
(P/'pass032-source-audit.json').write_text(json.dumps(aud,ensure_ascii=False,indent=2)+'\n')
(P/'pass032-review.md').write_text('# Zargari adjective and noun bases with explicit derivatives\n\n12 records / 12 rows: 4 reflex, 8 derived. Prepared under standing autonomous authorization.\n\n| # | Zargari | Proposed etymology | Evidence |\n|---|---|---|---|\n'+'\n'.join(table)+'\n\nEvidence: homepage search011–012; full CDIAL articles and addenda in zargari-primary009–010.json; existing primary grammar glyph decoding in zargari-primary-pages.json, printed p.141. Source audit retained in pass032-source-audit.json. No source reparsing or new nodes. The additional cik-ālo adjective is excluded from this batch pending its own morphology review.\n\nValidation: pending.\n')
print(len(a))
