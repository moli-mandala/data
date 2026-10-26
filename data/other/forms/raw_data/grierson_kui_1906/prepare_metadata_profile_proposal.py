"""Source-local, literal-preserving inventory; not an installer or final freeze."""
import collections
import hashlib
import json
import unicodedata as ud
from pathlib import Path
from segments import Tokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE/'metadata-profile-proposal'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()]

def clusters(form):
    result = []
    for c in ud.normalize('NFD',form):
        if ud.combining(c) and result:
            result[-1] += c
        else:
            result.append(c)
    return result

def main():
    OUT.mkdir(exist_ok=True)
    paths = ['grammar-reviewed.jsonl','table-reviewed-reading.json',
             'specimens/interlinear-reviewed.jsonl','specimens/expression-spans-reviewed.jsonl',
             'outsidechapter-original-reviewed.jsonl','outside-comparands-first-reading.jsonl',
             'source-lect-mapping-approved-20260927.json','volume-notation-review-20260927.json',
             'independent-bibliography-addenda-review-20260927.json']
    pins = {p:digest(HERE/p) for p in paths}
    records = []
    for name,section in [('grammar-reviewed.jsonl','grammar'),
                         ('outsidechapter-original-reviewed.jsonl','outside-target'),
                         ('outside-comparands-first-reading.jsonl','outside-comparand')]:
        for r in lines(HERE/name):
            records.append(dict(key=r['source_unit_key'], section=section,
                                lect=r.get('route_key') or r.get('source_language_label') or r.get('source_lect','unlocalized'),
                                form=r['form'], review=r.get('status',r.get('review_status'))))
    for r in json.loads((HERE/'table-reviewed-reading.json').read_text())['readings']:
        for i,f in enumerate(r.get('forms',[]),1):
            records.append(dict(key=f'table:prompt{r["prompt_id"]}:answer{i}',section='table',
                                lect='khondmals',form=f,review='literal reviewed; metadata proposal pending'))
    for name,section in [('specimens/interlinear-reviewed.jsonl','specimen-atom'),
                         ('specimens/expression-spans-reviewed.jsonl','specimen-expression')]:
        for r in lines(HERE/name):
            records.append(dict(key=r['source_unit_key'],section=section,
                                lect=r.get('source_dialect',f'specimen{r["specimen"]}'),
                                form=r['raw_source_form'],review=r.get('review_status')))
    inventory = collections.defaultdict(collections.Counter)
    tokens = set()
    for r in records:
        inventory[r['section']+' / '+r['lect']].update(r['form'])
        tokens.update(clusters(r['form']))
    # This preservation proposal deliberately makes no phonemic mapping. Form and
    # Original can be identical; all source punctuation/marks stay reversible.
    profile = {' ': '#'}
    for token in tokens:
        if token == ' ':
            continue
        profile[token] = ud.normalize('NFC',token)
        profile[ud.normalize('NFC',token)] = ud.normalize('NFC',token)
    profile_path = OUT/'literal-preservation-proposed.txt'
    profile_path.write_text('Grapheme\tIPA\n'+''.join(f'{g}\t{v}\n' for g,v in sorted(profile.items())))
    tok = Tokenizer(str(profile_path))
    failed = []
    for r in records:
        for normalization in ('NFC','NFD'):
            form = ud.normalize(normalization,r['form'])
            converted = tok(form,column='IPA').replace(' ','').replace('#',' ')
            if converted != ud.normalize('NFC',form):
                failed.append(dict(key=r['key'],normalization=normalization,original=form,converted=converted))
    assert not failed, failed[:4]
    symbols = {label:[dict(symbol=c,codepoint=f'U+{ord(c):04X}',name=ud.name(c,'UNNAMED'),count=n)
                       for c,n in sorted(count.items())] for label,count in sorted(inventory.items())}
    (OUT/'unicode-inventory.json').write_text(json.dumps(symbols,ensure_ascii=False,indent=2)+'\n')
    (OUT/'candidate-forms.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    bibliography = json.loads((HERE/'independent-bibliography-addenda-review-20260927.json').read_text())
    mappings = json.loads((HERE/'source-lect-mapping-approved-20260927.json').read_text())
    metadata = dict(status='proposal only; no canonical/shared metadata modified',
        proposed_primary_key='grierson1906kui', parent_existing_key='grierson1906lsi4',
        primary=dict(author='Grierson, George Abraham',year='1906',
            title='Linguistic Survey of India. Volume IV: Munda and Dravidian Languages. Kui chapter and attributed comparisons',
            address='Calcutta',publisher='Office of the Superintendent of Government Printing, India',
            edition='Original 1906 edition; IA alternate image witness used for reading',
            scope='Complete printed457–471 including accounted blank461; full241-prompt standard column; explicit outside-chapter Kui and labelled comparison witnesses. Final counts pending integrated exporter.',
            license='Original1906 work public domain; do not infer a separate modern scan licence.',
            provenance='Final installed CSV/audit paths must be filled only after accepted integrated freeze.',
            ocr='Yes: keyed Nordic/extracted text contributed to grammar and comparison preparation; exact section methods retained.',
            etymology_provenance='source: attributed comparative/derivational claims retained, not automatically graph edges'),
        addendum=dict(existing_key=bibliography['addendum']['existing_key'],
            status='separate undated later witness; not1906',lexical_corrections='No applicable Kui lexical correction on inspected sheets17–18',
            scope='Bibliographic/historical context only; do not attach as independent attestation to1906 lexical rows'),
        auxiliary_works=bibliography['priority_auxiliary_identifications'],
        secondary_authority_policy='Create minimal auxiliary keys where an emitted claim depends on the named work; references are quoted through Grierson, not independently collated. Reuse S-Kuwi for1913 vocabulary. Do not duplicate shared bibliography declarations.',
        approved_lect_mapping=mappings,
        field_policy=dict(Original='exact source transcription in CSV Form',Form='preservation profile output, NFC-equivalent literal',
            Native='blank for Roman interlinears; do not copy romanization here',Phonemic='blank: no independently supplied IPA',
            Tags='source grammatical labels and typed uncertainty; no inference from English -ing alone',
            identity='Entry_Key dedupe; distinct physical witnesses retained unless exact reuse explicitly documented'),
        unresolved_profile_decisions=['Pure preservation route is proposed, not final house-profile approval. Any later lowercase/ŋ/v conversion must preserve Original and be independently reviewed.',
            'Smith nasal mark varies: provisional ñ remains uncertain, not converted to IPA.',
            'Broad lower r mark in specimen42 ṛai retains dot-versus-bar uncertainty.',
            'Marked h/ḥ and damaged nasal glyphs remain literal candidates pending their keyed review; no generic visarga interpretation.',
            'Source vowel-length disagreements across Letchmajee, Smith and supplied specimens are distinct witnesses, not normalizable errors.'])
    (OUT/'metadata-proposal.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    status = dict(status='source-local provisional preservation proposal; not installation-ready',inputs=pins,
        inventory_records=len(records),counts_by_section=dict(collections.Counter(r['section'] for r in records)),
        unicode_symbols=len(set(''.join(r['form'] for r in records))),profile_rules=len(profile),
        normalization_checks=2*len(records),failed=0,
        caveat='Inventory includes controls, repeated expressions and pending comparison candidates; counts are not installed lexical counts.',
        source_defined='Volume key macrons/upper nasalization and source-specific orthographic statements are preserved; no new phonemic assignment.',
        gates_pending=['outside comparator supplement and final independent disposition','grammar final independent reconciliation',
            'table metadata approval','whole output and expression boundary audit','final profile policy/actual parser approval',
            'canonical registry/bibliography/settings installation','full data build and full suite deferred by instruction','browser refresh not requested'],
        outputs={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='status.json'},canonical_changes=False)
    (OUT/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:status[k] for k in ('inventory_records','counts_by_section','unicode_symbols','profile_rules','normalization_checks','failed')},indent=2))

if __name__ == '__main__':
    main()
