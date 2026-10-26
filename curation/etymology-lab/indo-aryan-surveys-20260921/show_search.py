import json,sys
from pathlib import Path
p=Path(__file__).resolve().parent
for name in sys.argv[1:]:
 for q in json.loads((p/(name+'.json')).read_text()):
  print(q['params'])
  for x in q['rows']:
   print(x['id'],x['language_id'],x['word'],repr(x['gloss']),'=>',x.get('origin_lemma_id'))
