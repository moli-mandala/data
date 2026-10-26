import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'pattanaik-koul2003varli[' in r['Source']];linked={r['Form_ID'] for r in read_assignments()};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('ear',['kān'],'2830','karṇa','Turner explicitly gives Gujarati/Marathi kān and Konkani kānu under karṇa, through MIA kaṇṇa. Both lects have the same long-vowel nasal-final stem.'),
('eye',['ḍōḷo','ḍōḷɛ','ḍola'],'6582','dōla','Turner gives Prakrit ḍōla eye and Marathi ḍoḷā under dōla; this specific eye sense identifies the family. The source l/ḷ and vowel differences are retained; these are not assigned to akṣi simply from their gloss.'),
('hair',['kē̃s','kẽs'],'3471','keśa','Turner explicitly lists Marathi kes/kẽs alongside Gujarati keś under keśa. The nasalized source vowels have a direct western comparator.'),
('nose',['nāk'],'6909','*nakka','Turner explicitly lists Gujarati/Marathi nāk under *nakka and Prakrit ṇakka; the cluster simplifies with vowel lengthening. Retain the reconstructed head and do not use the lexicographers’ nakra Sanskritization.'),
('tongue',['jībh','jīb'],'5228','jihvā','Turner section1 gives Gujarati/Marathi jībh and Konkani jīb(h)a under jihvā via Prakrit jibbhā. The separately recorded aspirated and unaspirated Davar responses remain distinct records.'),
('tooth',['dāt','dā̃t'],'6152','danta','Turner explicitly gives Gujarati/Marathi dãt under danta, with MIA daṁta. The survey records nasalization in Dungar but not Davar; retain this difference rather than silently supplying a nasal mark.'),
('salt',['mīṭh','mīṭ'],'10299','mṛṣṭa','Turner explicitly identifies Marathi mīṭh salt and Konkani mīṭa salt under mṛṣṭa/miṣṭa, whose wider senses include savoury/sweet. The salt sense is directly attested in the regional comparison and is not inferred from sweetness alone.'),
('milk',['dudh'],'6391','dugdha','Turner explicitly gives Gujarati/Marathi dūdh milk with Pali/Prakrit duddha under dugdha. Retain the source short vowel spelling; the d-dh sequence and milk sense support this family.')]
a=[];held=[];lines=['# Varli survey: body parts, salt and milk','','16 records proposed; two Davar eye readings held across eight families; standing autonomous authorization.','','| # | Varli survey | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(g,fs,parent,head,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['ID'] not in linked and r['Gloss']==g and r['Form'] in fs]
 for r in hits:a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=f'CDIAL[{parent} and addenda];'+r['Source'],evidence=ev+' Existing Davar/Bhili and Dungar/Varli source classifications are preserved. These comparisons support an inherited regional family; no specific recent donor is established.'))
 lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({len(hits)} records) | **{head.replace("*",chr(92)+"*")}**, CDIAL **{parent}** | {ev} |')
assert len(a)==18,len(a)
for x in list(a):
 if x['record']['Form'] in ['ḍōḷo','ḍōḷɛ']:
  a.remove(x);held.append(dict(record=x['record'],parent=x['parent'],reason='Source audit flags undefined capital L in Do:Lo/Do:LE. Installed rendering uses retroflex lateral; do not resolve that transcription interpretation through an etymology assignment. Dola-family comparison plausible but held pending source-symbol clarification.'))
audit=[json.loads(l) for l in open('data/other/forms/raw_data/20260911-dadra-varli-audit.jsonl')];glosses={g[0] for g in groups};audit=[r for r in audit if r['gloss'] in glosses and r['status']=='ingested'];assert len(audit)==16,len(audit)
assert all(r['review']=='manual-page-image' for r in audit)
assert sum(bool(f.get('uncertainty')) for r in audit for f in r['forms'])==2
(P/'pass038-source-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
(P/'pass038-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
lines+=['','Eight elicitation items across two lects yield18 existing records because two Davar responses have separate alternatives. All16 source-audit rows have manual-page-image review. Two Davar eye alternatives flag undefined capital L and are held; the other16 records have no such uncertainty. The eye table row includes these two held comparisons; only Dungar ḍola is accepted. No source transcription, identity or dialect mapping changed. Earlier holds remain unresolved.','Research: homepage varli-search007.json and full dictionary articles/addenda varli-primary006.json. No new nodes or rebuild.','Validation: pending.']
(P/'pass038-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted; audit rows',len(audit))
