"""Enumerate exact same-representation component witnesses for manual review."""
import csv
import json
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'graph-reconciliation-r2'
claims=list(map(json.loads,(R/'claim-review-queue.jsonl').open()))
audit=list(map(json.loads,(P/'whole-proposal-r1/whole-proposed-audit.jsonl').open()))
rows={r[10]:r for r in csv.reader((P/'whole-proposal-r1/whole-proposed.csv').open())}
# Exact lexical components named by the source; no decomposition from surface form.
pdf={
 'nečango':['na'],'čeli pete':['peti'],'čuri karen':['karen'],
 'dandi genen':['genen'],'dang tʰiyen':['tʰiyen'],'derāz kadden':['kadden'],
 'fitak kadden':['kadden'],'gap čʰiyen':['čʰiyen'],'gāy':['gāyen'],
 'gere čʰiyen':['gere','čʰiyen'],'hat kadden':['kadden'],'hate genen':['genen'],
 'hesāb karen':['karen'],'hite čʰiyen':['čʰiyen'],'hite xaten':['xaten'],
 'hoven':['ho'],'hozen':['ho'],'jodā karen':['karen'],'karčax čʰiyen':['čʰiyen'],
 'kesam xāyen':['xāyen'],'komak karen':['karen'],'lāk karen':['karen'],
 'majbur karen':['karen'],'majbur tʰiyen':['tʰiyen'],'mate karen':['karen'],
 'nekter noko':['noko'],'nekter vazo':['vazo'],'nevalm':['valm'],
 'pal karen':['karen'],'pay karen':['karen'],'poni mate pʰiyen':['mate','pʰiyen'],
 'šājjo karen':['karen'],'sorāx genen':['genen'],'sut kadden':['kadden'],
 'ters karen':['karen'],'tof karen':['karen'],'tof lāren':['lāren'],
 'vāt vanen':['vanen'],'vāv čʰiyen':['čʰiyen'],'vov vov karen':['karen'],
 'xān diyen':['diyen'],'xān xaten':['xaten'],
}
repo={
 'berenoko':['noko'],'movāy':['māv ~ mov'],'nevalem(te)':['ne','valem(te)'],
 'nevalm':['ne','valm'],'sāvāyi':['sāo'],'dand digenen':['dand','digenen'],
 'to anen':['to','anen'],'carx diyen':['carx','diyen'],'hate genen':['hat','genen'],
 'vāt venen':['vāt','venen'],'bafer pʰazen':['bafer','pʰazen'],
 'vyāvāy':['vyāw'],'dahri tale':['dahri','tale'],'vazoro':['vazo'],
}
results=[]
for c in claims:
 ispdf=c['provenance']=='printed author dictionary'
 expected=(pdf if ispdf else repo).get(c['source_form'])
 if expected is None:continue
 choices=[]
 for form in expected:
  matches=[]
  for u in audit:
   if (u.get('representation')=='published PDF')!=ispdf:continue
   r=u.get('source_record',{})
   if r.get('form')!=form:continue
   for key in u['emitted_entry_keys']:
    matches.append(dict(entry_key=key,form=rows[key][2],gloss=rows[key][3],
                        source_unit_key=u['source_unit_key'],citation=rows[key][7]))
  choices.append(dict(named_component=form,candidates=matches))
 results.append(dict(source_unit_key=c['source_unit_key'],source_form=c['source_form'],
  emitted_entry_keys=c['emitted_entry_keys'],literal_claim=c['literal_claim'],
  components=choices,status='pending sense and qualification review; no edges assigned'))
(R/'local-component-review-queue.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print('Named component claims queued:',len(results))
