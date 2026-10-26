"""Prepare exact keyed r2 scope corrections from the preserved metadata derivative."""
import csv
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
by={r[10]:r for r in csv.reader((HERE/'proposed.csv').open())}
assert all(not r[11] for r in by.values() if ':v2' in r[10]), 'revision already applied'
census=json.loads((HERE/'same-cell-slash-graph-census.json').read_text())
changes=[]
def change(key,field,new):
    column={'Notes':6,'Tags':14,'Variant_Of_Key':11}[field]
    old=by[key][column]
    if old!=new: changes.append({'entry_key':key,'field':field,'old':old,'new':new})
for g in census['groups']:
    child,parent=g['keys'][1],g['keys'][0]
    g['decision']='Root-approved same-unit source-listed alternate realizations; second-to-first pointer, without free-interchangeability claim.'
    section=g['unit_key'].split(':')[2]
    rationale={
      'plural-markers':'Source groups both realizations as a plural marker.',
      'plural-human-marker-repeat':'Source repeats this pair with av-/iv- stems and human substantives.',
      'plural-prevocalic':'Both source-listed realizations occur in the before-vowel group; no link to the separate before-pause group.',
      'plural-prepause':'Both source-listed realizations occur in the before-pause group; no link to the separate before-vowel group.',
      'feminine-markers':'Both realizations occupy one feminine-marker source slot.',
      'remote-demonstratives':'Both realizations share the remote neuter-plural pronoun cell.',
      'proximate-demonstratives':'Both realizations share the proximate neuter-plural pronoun cell; feminine singular iva is a separate attestation.',
      'personal-ending-table':'Both realizations share the third-person human source cell; the table supplies no number heading.',
      'zvelebil-temporal-comparison':'Both realizations share one reported temporal formula; distinct formulas remain separate.'
    }[section]
    g['rationale']=rationale
    change(child,'Variant_Of_Key',parent)
    for key in g['keys']:
        note=by[key][6]
        if section=='zvelebil-temporal-comparison':
            stem='past' if g['unit_key'].endswith(':03') else 'non-past'
            change(key,'Tags','suffix temporal')
            note+='; Nair reports Zvelebil (1982:204) conditioning this formula on a '+stem+' stem; this describes stem conditioning, not intrinsic suffix tense.'
        note+='; Source-listed same-slot alternatives. '+rationale
        change(key,'Notes',note)
for i,kind in [('01','weak'),('02','strong')]:
    for section in ['optative-markers','optative-examples']:
        key='nair2013irula:p26:'+section+':'+i
        note=by[key][6]
        note+=('; ' if note else '')+'Nair reports Zvelebil (1982:202): this optative realization is used with '+kind+'-class verbs; the class describes conjugation conditioning.'
        change(key,'Notes',note)
        if section=='optative-examples':change(key,'Tags',by[key][14]+' '+kind)
census['status']='All 11 same-cell slash groups approved with per-group source conditioning; no cross-group edges.'
(HERE/'same-cell-slash-graph-census.json').write_text(json.dumps(census,ensure_ascii=False,indent=2)+'\n')
assert sum(c['field']=='Variant_Of_Key' for c in changes)==11
assert sum(c['field']=='Tags' for c in changes)==8
(HERE/'scope-revisions-r2.json').write_text(json.dumps({'status':'root-approved scope refinements after independent review','changes':changes,'qualification':'Six temporal suffix tense tags reverted; conditioning retained in Notes. Two optative examples inherit explicitly adjacent weak/strong conjugation classes.'},ensure_ascii=False,indent=2)+'\n')
print('prepared',len(changes),'keyed field corrections')
