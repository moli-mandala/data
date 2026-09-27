"""Stage the complete visually reviewed Bailey Padari source; never install implicitly."""
from pathlib import Path
import csv,json,unicodedata,re
from collections import Counter
P=Path(__file__).resolve().parent
DATA=P.parents[4]
SOURCE='bailey1908padari'
def nfc(s): return unicodedata.normalize('NFC',s)
def generate():
 units=[json.loads(s) for s in (P/'full-transcription.jsonl').read_text().splitlines()]
 assert len(units)==753 and len({x['entry_key'] for x in units})==753
 rows=[];audit=[]
 for u in units:
  assert u['review']=='visual-second-pass'
  assert all(f.startswith('source-glyph-uncertain:') for f in u['flags'])
  tags=[{'past':'pret','ptcp':'participle','adp':'prep','imp':'impv'}.get(t,t) for t in u['tags']];notes=[u['notes'].strip()] if u['notes'].strip() else []
  person=u.get('source_person','')
  if not person:
   m=re.search(r':pronoun:([123]):(sg|pl):',u['entry_key'])
   if m:person=m[1]
  number=next((v for v in ('sg','pl') if v in tags),'')
  if person in ('1','2','3') and number:tags.append(person+number)
  if u.get('source_asterisk'):notes.append('Source asterisk marks resemblance to a corresponding Pangwali word; no etymological relationship is asserted here.')
  if u['section']=='glossary' and tags:notes.append('Part of speech follows the English lexical head; the source glossary supplies no explicit POS label.')
  locator=f"Part {u['part']}, p. {u['printed_page']}, {u['section']}, "
  locator+=(f"{u['column']} column, " if u['column'] else '')+f"item {u['item']}"
  emitted=[]
  for j,form in enumerate(u['forms'],1):
   key=u['entry_key']+(f':variant:{j}' if j>1 else '')
   local_tags=list(tags)
   uncertain=bool(u['flags']) and not (j>1 and all('first-alternative' in f for f in u['flags']))
   if uncertain:local_tags.append('uncertain')
   if ' ' in form and 'suffix' not in local_tags:local_tags.append('multiword-expression')
   row=['Padri','',nfc(form),u['gloss'],'','',' '.join(notes),f'{SOURCE}[{locator}]','','',key,u['entry_key'] if j>1 else '','','',' '.join(dict.fromkeys(local_tags))]
   rows.append(row);emitted.append(key)
  audit.append({**u,'source_unit_key':u['entry_key'],'status':'ingested' if emitted else 'source-blank','entry_keys':emitted,'citation_locator':locator,'source_form_layer':'literal Bailey Roman notation; not inferred IPA','grammar_policy':'explicit paradigms and source headings; contextual glossary POS identified in Notes','uncertainty':u['flags']})
 assert len({r[10] for r in rows})==len(rows)
 legacy=list(csv.reader((DATA/'data/other/forms/20260925-bailey-padari.csv').open()))
 assert {r[10] for r in legacy}.issubset({r[10] for r in rows})
 return rows,audit

def main():
 rows,audit=generate()
 with (P/'proposal.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
 (P/'proposal-audit.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in audit))
 # Exact literal profile; normalization never turns historical transcription into IPA.
 graphemes=set()
 for r in rows:
  clusters=[]
  for c in r[2]:
   if unicodedata.combining(c) and clusters:clusters[-1]+=c
   else:clusters.append(c)
  graphemes.update(clusters)
 (P/'proposal-profile.txt').write_text('Grapheme\tIPA\n'+''.join(f'{c}\t'+('#' if c==' ' else c)+'\n' for c in sorted(graphemes)))
 print(len(rows),'rows;',len(audit),'units;',dict(Counter(x['status'] for x in audit)))
if __name__=='__main__':main()
