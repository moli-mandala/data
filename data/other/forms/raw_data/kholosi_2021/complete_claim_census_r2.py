"""Every source etymology receives a durable graph-scope disposition."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'graph-reconciliation-r2'
claims=list(map(json.loads,(R/'claim-review-queue.jsonl').open()))
numeric={r['source_unit_key']:r for r in map(json.loads,(R/'numeric-claim-decisions.jsonl').open())}
components={r['source_unit_key']:r for r in json.loads((R/'local-component-decisions.json').read_text())}
out=[]
for c in claims:
 r=dict(c);key=c['source_unit_key'];text=c['literal_claim'].lower().strip()
 if key in numeric:
  r['numeric_decision']=numeric[key]
  r['graph_disposition']=numeric[key]['disposition']
 elif key in components:
  r['component_decision']=components[key]
  r['graph_disposition']='explicit local components reviewed; unmatched pieces retained'
 elif text in ('unknown.','unknown','?'):
  r['graph_disposition']='author explicitly leaves ancestry unknown'
 elif text.rstrip('.') in ('sanskrit','skt','native'):
  r['graph_disposition']='language/origin label only; no identified ancestral lexeme or target'
 elif 'gārhya' in text:
  r['graph_disposition']='qualified headword claim unmatched in target inventory'
  r['reason']='No exact GĀRHYA head exists in the tracked CDIAL target inventory; reconstructed MIA and comparative Domari/Romani forms do not establish a unique target ID.'
 elif text.startswith('cf.') or text.startswith('lit.'):
  r['graph_disposition']='comparison or literal compositional explanation, not an explicit ancestry assignment'
 elif text.startswith('-'):
  r['graph_disposition']='explicit morphology handled by occurrence-linked morphology witnesses'
 else:
  r['graph_disposition']='attributed external donor/comparison or unresolved ancestry prose; no keyed donor witness established'
  r['reason']='Preserve exact named language, lexeme, qualifiers and source locator. No donor source key is supplied by this dictionary; target-language homographs must not become borrowed-from nodes. No unrelated-language entry is synthesized.'
 r['review_status']='scope disposition recorded; independent output review pending'
 out.append(r)
assert len(out)==830 and len({r['claim_key'] for r in out})==830
(R/'complete-source-claim-dispositions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
print('Complete claim accounting:',len(out))
