"""Census explicit quoted target forms and annotation examples outside primary head columns."""
import csv,json,re
from pathlib import Path
P=Path(__file__).resolve().parent
rows=list(csv.reader((P/'upstream/lexicon.csv').open()))[1:];out=[]
for n,r in enumerate(rows,2):
 key=f'kholosi:github:18b43f982efa:lexicon.csv:row{n}'
 if r[11].startswith('Anonby ('):
  m=re.fullmatch(r'Anonby \((.*)\)(\?)?',r[11]);assert m,r[11]
  form=m.group(1);out.append(dict(source_unit_key=key+':quoted-anonby',parent_key=key,form=form,gloss=r[5],source_authority='Anonby, quoted by author spreadsheet',raw_source=r[11],source_scope='Kholosi alternate attestation; not foreign-language control',tags=['uncertain'] if m.group(2) else [],uncertainty_type='source questions identification of truncated quoted form' if m.group(2) else '',relation_status='no automatic variant edge; cited witness need not be same lexical/morphological form',review_status='pending independent review'))
 if n==85:
  assert r[13]=='"in front of" = ji jolo mā'
  out.append(dict(source_unit_key=key+':note-expression',parent_key=key,form='ji jolo mā',gloss='in front of',tags=['multiword-expression'],raw_source=r[13],source_scope='explicit translated whole phrase in source note',review_status='pending independent review'))
(P/'supplemental-annotation-occurrences.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
print('quoted',sum('quoted-anonby' in x['source_unit_key'] for x in out),'noteexpressions',sum('note-expression' in x['source_unit_key'] for x in out))
