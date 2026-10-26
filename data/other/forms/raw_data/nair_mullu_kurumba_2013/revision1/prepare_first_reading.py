"""Complete original-based first reading; no canonical installation or acceptance."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCOPE = HERE / 'scope'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
INVENTORY = SCOPE / 'source-position-inventory.jsonl'
assert sha(INVENTORY) == 'e37de419030701a646dfaf294c51e1c01dff3b4e06ef20c8720ba0754107056d'
units = [json.loads(s) for s in INVENTORY.read_text().splitlines()]
assert len(units) == 62
result = []
for unit in units:
    page = unit['printed_page']
    witnesses = []
    for lead in unit['witnesses']:
        scope = lead['source_scope']
        disposition = ('target' if scope == 'Mullu Kurumba target' else
            'explicit_comparison' if scope == 'explicit Malayalam comparison' else 'audit_only')
        if unit['disposition'] in ('context', 'documentary', 'blank_cell', 'zero_or_documentary'):
            disposition = unit['disposition']
        if unit['section'] == 'identity-ethnonym-leads': disposition = 'unresolved_language_documentary'
        witnesses.append(dict(witness_key=lead['witness_key'], source_scope=scope,
            literal_first_reading=lead['unaccepted_notation_lead'], printed_gloss=lead['printed_definition_lead'],
            disposition=disposition, source_locator=lead['locator'],
            accepted_for_installation=False, literal_review='Owner compared complete original page; independent review pending'))
    result.append(dict(source_unit_key=unit['source_unit_key'], printed_page=page, pdf_page=page,
        section=unit['section'], source_witnesses=witnesses, disposition=unit['disposition'],
        source_claim=unit['source_claim'], note=unit['caution'],
        original_image='scope/' + unit['original_image'],
        original_image_sha256=sha(SCOPE / unit['original_image']),
        accepted_for_installation=False, review='Complete original-based owner first reading; metadata and independent review pending'))

# These exact outside mentions contain classification/contents only. Betta-specific
# lexical material on p12 is explicitly another chapter and is not a Mullu witness.
for page, section, claim in [(3, 'contents', 'Contents lists Mullukurumba beginning on p55; documentary navigation only.'),
    (12, 'shared-classification-subgroups', 'Mullu Kurumba is named among Kurumba subgroups, with differences from the three principal subgroups; ethnographic classification, not a lexical quotation.'),
    (12, 'reported-thurston-classification', 'Nair reports Thurston treating Mullu Kurumba as a Malayalam dialect. The following independent-language authorities and Bettu quotation concern Betta Kurumba, not Mullu.')]:
    result.append(dict(source_unit_key=f'nair2013mullukurumba:p{page}:{section}', printed_page=page, pdf_page=page,
        section=section, source_witnesses=[], disposition='documentary', source_claim=claim,
        accepted_for_installation=False, review='Outside-name census and original p12 paragraph inspected; no additional target lexical quotation'))
assert len(result) == 65 and len({u['source_unit_key'] for u in result}) == 65
witnesses = [w for u in result for w in u['source_witnesses']]
assert len(witnesses) == len({w['witness_key'] for w in witnesses}) == 164
assert all(not any(0xE000 <= ord(c) <= 0xF8FF for c in w['literal_first_reading']) for w in witnesses)
path = HERE / 'physical-units-first-reading.jsonl'
path.write_text(''.join(json.dumps(u, ensure_ascii=False, sort_keys=True) + '\n' for u in result))
(HERE / 'first-reading-status.json').write_text(json.dumps(dict(status='Complete owner first reading; no accepted export',
    chapter_groups=62, outside_documentary_groups=3, audit_groups=65, scoped_witness_records=164,
    inputs={'inventory': sha(INVENTORY), 'font_controls': sha(SCOPE / 'detail-font-evidence.json')},
    first_reading_sha256=sha(path), original_pdf_sha256='4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af',
    transcription='F02C visibly round underdot mapped U0323; F019 tilde mapped U0303. Separate drawn lowlines encode l/n plus U0332. Exact source dashes and plain repeated vowels preserved; no phonemic inference.',
    scope='164 is scoped witness records, not distinct physical tokens; explicitly shared suffix witnesses and repeated commentary remain identified.',
    source_controls=['p56 third-person plural pronouns are gender-unspecified proximal/distal free variants',
        'p57 plural pronominal suffix alternatives occupy masculine row; female plural blank; neuter zero cells documentary',
        'en-attinu/from the trap is literal source conflict; no inferred first-person tag',
        'No numeral forms printed; existing Jeba2018 numerals remain separate',
        'Ethnonyms and ten settlements are documentary/unresolved-language, not inferred lemmata or elicitation sites'],
    canonical_changes=False), ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'audit_groups': 65, 'scoped_witness_records': 164, 'canonical_changes': False}))
