"""Generate discovery leads from earlier pending reviews; not etymological evidence."""
import difflib
import json
import re
import unicodedata
from pathlib import Path

p = Path(__file__).resolve().parent
lab = p.parent
registry = json.loads((lab / 'dravidian-surveys-20260921/goal-pass-registry.json').read_text())
targets = json.loads((p / 'survey-inventory.json').read_text())

def norm(value):
    value = unicodedata.normalize('NFKD', value.casefold())
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return re.sub('[^a-zɨəɛɔŋñḍṭṇḷḻṟ]', '', value)

def gloss_tokens(value):
    return set(re.findall('[a-z]+', value.casefold())) - {'the', 'a', 'to', 'of', 'for', 'and', 'person', 'things'}

prior = []
for name in registry:
    file = lab / name / 'proposed-decisions.json'
    if not file.exists():
        continue
    for item in json.loads(file.read_text()).get('accepted', []):
        rec = item['record']
        prior.append((norm(rec['Form']), gloss_tokens(rec['Gloss']), rec['Gloss'], rec['Language_ID'], rec['Form'], item['parent']))

out = []
for target in targets:
    form = norm(target['Form'])
    gloss = gloss_tokens(target['Gloss'])
    leads = []
    for prior_form, prior_gloss, prior_gloss_raw, language, prior_raw, parent in prior:
        if not form or not prior_form or not gloss.intersection(prior_gloss):
            continue
        sim = difflib.SequenceMatcher(None, form, prior_form).ratio()
        if sim >= 0.58:
            leads.append({'formScore': round(sim, 3), 'parent': parent, 'language': language, 'form': prior_raw, 'gloss': prior_gloss_raw})
    leads.sort(key=lambda x: x['formScore'], reverse=True)
    out.append({'ID': target['ID'], 'Form': target['Form'], 'Gloss': target['Gloss'], 'leads': leads[:12]})
(p / 'prior-proposal-discovery-leads.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print({'priorProposals': len(prior), 'targets': len(out), 'withLeads': sum(bool(x['leads']) for x in out)})
