"""Independent source-local routing and exact canonical-occurrence reuse review."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = next(p for p in P.parents if (p / 'data/SOURCE_INGESTION_CHECKLIST.md').exists())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
base = [json.loads(x) for x in (P/'outside-comparands-first-reading.jsonl').read_text().splitlines()]
supplement = json.loads((P/'outside-comparands-scope-supplement-20260927.json').read_text())
rows = base + supplement['units']
languages = {r['ID']: r for r in csv.DictReader((ROOT/'data/cldf/languages.csv').open())}
dialects = {r['ID']: r for r in csv.DictReader((ROOT/'data/cldf/dialects.csv').open())}
print('LABELS', dict(Counter(r['source_language_label'] for r in rows)))
mapping = {'Juang':'ju', 'Tamil':'Tamil', 'Tamil (contextual)':'Tamil', 'High Tamil':'Tamil',
 'Kanarese':'Kannada','Telugu':'Telugu','Kurukh':'Kurux','Gondi':'Gondi',
 'Brahui':'Brahui','Tuḷu':'Tulu','Maḷyāḷam':'Malayalam','Kolami':'Kolami','Naiki':'Naiki','Malto':'Malto'}
decisions=[]
reuses=[]
canonical_cache={}
for r in rows:
    label=r['source_language_label']
    d=dict(source_unit_key=r['source_unit_key'], input_label=label, source_label=label,
           language_id=None, dialect_id=None, status='unresolved_source_identity', note='')
    if label in mapping:
        d.update(language_id=mapping[label], status='existing_language')
        d['note']='Exact source label or established registry name alias; no source locality or modern coordinates inferred.'
    elif label=='Gondi (Mandla)':
        d.update(language_id='Gondi',dialect_id='mandla',status='existing_dialect',
                 note='Supplement explicitly gives Mandla only for this example. Existing mandla is the historical Gondi lect; do not propagate this locality to the following unqualified form.')
    elif label in {'Kōi','Kōi (Gondi context)'}:
        d.update(language_id='Gondi',dialect_id='koya',status='existing_historical_dialect_proposal',
                 note='LSI IV p473 explicitly places Kōi within the historical Gondi discussion while distinguishing its southernmost form. Existing historical koya registry route is Gondi/koya with glottocode koya1251; use that qualified source route, not a new locality, Gommu, or silent global migration to newly registered modern Koya. Root should approve this historical compatibility choice.')
    elif label=='Parji':
        d.update(source_label='Bhili of Basim',status='source_label_correction_and_registry_gap',
                 note='Original p566 explicitly says Bhili received from Basim and compares it with Kolami; the n/nd/d forms are that witness, not Parji. Nordic page metadata proposes Southwestern Kolami-Washim (nit), whereas existing Kolami is nort2699 and Naiki is naik1250. Neither existing base nor a northern Kolami locality is securely interchangeable. Preserve all five records under unresolved historical Bhili of Basim pending an approved identity route; never Parji or Indo-Aryan Bhili.')
    elif label.startswith('Korava'):
        d.update(note='Generic historical Korava comparison is not localized. Installed Yerukula/korvi_belgaum and other routes were justified specifically for the Belgaum/Jamkhandi/Bijapur chapter witnesses. These comparative suffixes have no such locality evidence; source Tamil-dialect classification alone cannot decide modern Tamil versus Yerukula. Keep source label with typed unresolved identity, candidate Yerukula only, and do not borrow a locality.')
    elif label=='Burgandi':
        d.update(note='Original LSI IV p343 states Burgandi is closely connected with Kaikadi, with unsatisfactory mixed Nimar/Central India material and principal Bagli specimen. Relatedness is not identity. No exact Burgandi base/dialect exists in current registry; retain source-specific unresolved Burgandi, not Kaikadi or Tamil by historical family placement.')
    elif label.startswith('generalized comparison'):
        d.update(status='documentary_generalized_morphology',note='Source gives common plural morphology rather than a localized language witness. Preserve the physical unit and source claim audit-only; do not multiply into all subsequent languages.')
    else:
        raise AssertionError(label)
    if d['language_id']: assert d['language_id'] in languages
    if d['dialect_id']:
        assert dialects[d['dialect_id']]['Language_ID']==d['language_id']
        d['dialect_tag']=dialects[d['dialect_id']]['Tag']
    if label=='High Tamil': d['note']='Route broad Tamil, retaining the explicit High Tamil register in Notes; not automatically Old Tamil or any spoken regional lect.'
    if label=='Tamil (contextual)': d['note']='Retain Tamil routing as contextual, not directly labelled at this suffix; the original comparison follows explicit Tamil discussion and uncertainty must survive.'
    if 'old Telugu' in r.get('note',''):
        d['note']='Explicit old Telugu qualification retained in source Notes; existing broad Telugu route, without inventing a dated lect or silently relabelling it modern conversational Telugu.'
    if r.get('status')=='source hypothetical form':
        d['attestation_status']='source_hypothesis_not_independent_attestation'
        d['note']+=' Preserve source hypothetical status and exact star presence/absence; no inferred etymological edges.'
    matches=r.get('verified_existing_occurrence_reuse',[])
    if matches:
        assert label=='Naiki' and len(matches)==1
        m=matches[0]; path=ROOT/m['file']
        if path not in canonical_cache:
            canonical_cache[path]={q[10]:q for q in csv.reader(path.open()) if len(q)>=15}
        q=m['row']; actual=canonical_cache[path].get(q[10])
        assert actual==q, (r['source_unit_key'],'canonical row drift')
        assert actual[0]=='Naiki' and actual[2]==r['form'] and actual[3]==r['gloss']
        expected_page=570 if ':p570:' in r['source_unit_key'] else 623
        assert f':p{expected_page}:' in q[10] and f'p. {expected_page}' in q[7]
        missing=set(r['tags'])-set(q[14].split())
        reuses.append(dict(source_unit_key=r['source_unit_key'], canonical_file=m['file'],
            canonical_file_sha256=sha(path), canonical_entry_key=q[10], exact_full_row_equal=True,
            original_equal=True, gloss_equal=True, source_page_equal=True,
            proposed_tags_missing_from_canonical=sorted(missing), canonical_tags=q[14],
            disposition='reuse exact installed occurrence, retaining canonical full analysis, Notes and citations; do not replace it with the shorter comparative draft'))
        d['reuse_key']=q[10]
    decisions.append(d)
assert len(reuses)==12
assert len({r['source_unit_key'] for r in decisions})==len(rows)==146
report=dict(reviewer='/root/asur_finish',date='2026-09-27',canonical_changed=False,
    status='complete source-local routing proposal with explicit identity holds',
    input_hashes={n:sha(P/n) for n in ['outside-comparands-first-reading.jsonl','outside-comparands-scope-supplement-20260927.json']},
    registry_hashes={n:sha(ROOT/'data/cldf'/n) for n in ['languages.csv','dialects.csv']},
    counts=dict(records=len(rows),base=len(base),supplement=len(supplement['units']),
                labels=dict(Counter(r['source_language_label'] for r in rows)),decisions=dict(Counter(d['status'] for d in decisions)),verified_reuses=len(reuses)),
    decisions=decisions,reuse_checks=reuses,
    evidence=[
      dict(source='LSI IV original p566, physical586',path='tmp/kui-root-outside-566.png',sha256=sha(ROOT/'tmp/kui-root-outside-566.png'),observation='Viewed complete original page: explicit Bhili from Basim, not Parji; n/nd/d paragraph contains the five disputed records.'),
      dict(source='LSI IV original p343, physical363',observation='Independently rendered and viewed original; Burgandi closely connected with Kaikadi, mixed Nimar/Central India material. Connection is not an exact language alias.'),
      dict(source='LSI IV original p473, physical493',observation='Independently rendered and viewed original; Kōi belongs to historical Gondi treatment and southernmost Kōi is distinguished. No locality is attached to the two comparator records.'),
      dict(source='LSI IV original p474, physical494',observation='Original treats Parji in Bastar separately and points to pp554ff; reinforces that p566 Basim witness must not be relabelled Parji.'),
      dict(source='grierson_korvi_1906/language-dialect-plan-20260926.json',observation='Existing Yerukula routes explicitly justified for named Belgaum/Jamkhandi/Bijapur witnesses; they do not prove every generic historical Korava label has the same modern identity.'),
      dict(source='nordic-volume-kui-mentions.json',observation='p566 metadata identifies Southwestern Kolami-Washim/nit. Retained as secondary routing lead, not promoted over original or used to overwrite existing northern Kolami identity.')],
    policy='No coordinates, new registries, lexical corrections, graph edges, or canonical edits. Explicit unresolved language identities remain preserved source evidence, not silent omissions or guessed assignments.')
(P/'outside-comparands-routing-reuse-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report['counts'],ensure_ascii=False))
