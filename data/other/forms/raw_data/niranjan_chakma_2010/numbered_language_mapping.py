"""Reproducible conservative identity proposals; no registry or lexical edits."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT.parents[4] / 'cldf/languages.csv'


def norm(s):
    # Only surrounding printed abbreviation punctuation; never internal letters.
    return s.strip().rstrip('.,').strip()


# Each non-identical spelling is an explicit proposed alias, not a fuzzy match.
ALIASES = {
    'কুরু': 'কুর', 'কোংক': 'কোং', 'কোঙ্ক': 'কোং',
    'নে': 'নেপ', 'লিম্': 'লি', 'মনি': 'মণি', 'মানি': 'মণি',
    'প্রা. বাং': 'প্রা. বাং', 'প্রা. বা': 'প্রা. বাং',
    'সাদরি': 'সাদ', 'গিয়া': 'গি',
}
DIRECT = {
    'অস': 'A', 'আ': 'Ar', 'ইরু': 'Irula', 'ওড়ি': 'Or', 'ওঁরা': 'Kurux',
    'কক': 'Kokborok', 'কুর': 'Kurux', 'কন্ন': 'Kannada', 'কোং': 'Ko',
    'খা': 'Khasi', 'গা': 'Garo', 'নেপ': 'N', 'নেও': 'New',
    'প্রা. বাং': 'OB', 'মণি': 'Manipuri', 'মৈথি': 'Mth',
    'বার্মি': 'Burmese', 'সং': 'Sk', 'সি': 'Sh', 'সিংহ': 'Si',
    'সাদ': 'Sadri', 'হি': 'H', 'হা': 'Hajong', 'হো': 'ho',
    'তা': 'Tamil', 'তি': 'Tib', 'বাং': 'B', 'ব্রা': 'Brahui',
    'রাজ': 'MH', 'রা': 'Rabha', 'লি': 'Limbu',
    # These undeclared but conventional shortened labels remain explicitly proposals.
    'সাঁও': 'sa', 'ফা': 'Pers', 'তে': 'Telugu', 'ডিমা': 'Dimasa',
    'গো': 'Gondi', 'বি.মনি': 'Bishnupriya', 'কথ্য বাং': 'B', 'কথ্যবাং': 'B',
}
REASONS = {
    'ব': 'Declaration says Burma, while two numbered cells use ব. Do not substitute Bengali from lexical resemblance; country/language and label collision remain unresolved.',
    'প্রা': 'Generic Prakrit is not the specifically Maharashtri registry node Pk.',
    'প্রাক': 'Apparent shortened Prakrit label; generic Prakrit is not Maharashtri Pk.',
    'মু': 'Source declares Munda, a broad grouping/community label; not enough to assign Mundari mu.',
    'কিরা': 'Kiranti is broader than a single language; no daughter lect inferred.',
    'পাহা': 'Pahari is a broad source label; no specific Pahari lect inferred.',
    'ইরা': 'Iranian does not uniquely establish Persian or any other Iranian language.',
    'কাই': 'Source Kaithi/Kayethi label does not uniquely name a language. Unicode documents Kaithi use for multiple languages; not Kaithal.',
    'সান': 'Declaration reads Sawar; not Santali and not automatically Sora. Historical/community identity needs evidence.',
    'শ': 'Undeclared short label. Do not expand to Shabar, Shina or Shan from resemblance.',
    'কো': 'Undeclared short label; could collide with Koch, Konkani or another name. No fuzzy expansion.',
    'কু': 'Undeclared short label; do not infer Kurukh rather than Kui or another language.',
    'চাক': 'Source abbreviation declaration চা=Chak differs from this label and elsewhere Chakma uses collide. Require occurrence-level identity evidence.',
    'চর্যা': 'Manuscript/corpus-stage attribution, not unqualified modern Bengali. Preserve Carya witness and seek explicit historical-language adjudication.',
    'বৈ': 'Explicit Vedic historical/register attribution; retain separately pending approved Sanskrit-stage routing, not silently modern/classical Sanskrit.',
    'ম': 'Declaration Mog is distinct from মগ=Magar. Mog is a historical exonym; no automatic Burmese or Arakanese mapping.',
    'মগ': 'Declaration Magar does not specify Eastern versus Western Magar; registry has distinct lects.',
    'মুর': 'Historical Murmi name does not resolve which Tamang variety; do not select one registered lect.',
    'খ': 'Historical Khas attribution is not enough to equate this witness with modern Nepali.',
    'ভু': 'Bhutani is geographic; do not assign Dzongkha, Tibetan or a precise lect without evidence.',
    'ব্রো': 'Declared Brokpa has multiple possible referents. Dedicated later Brok-Skad/Ladakh chapter supports bro there, but numbered instances have no Ladakh qualifier. Keep bro as candidate, not a confirmed mapping.',
    'শের': 'Sherpa label does not establish the registered Solu-Khumbu variety specifically.',
    'কাক': 'Kakmara field-research lead describes a Telugu-related Midnapore variety, but identity with this book is unproved; do not assign Telugu.',
    'চি': 'Generic Chinese does not identify a particular modern lect.',
    'মাল': 'Undeclared abbreviated label; neither Malayalam nor Mal Paharia inferred.',
    'চাখে': 'Declared name remains a compact-glyph and identity uncertainty; do not modernize by resemblance.',
    'খু': 'Declared Khutta reading/identity remains unresolved.',
    'গি': 'Declared Giyami name remains unresolved; no modern identity from resemblance.',
    'ওয়া': 'Declared Wangkho-like name has unresolved medial transcription and identity.',
    'থাং': 'Declared Tangsa is insufficient to select a particular variety or registry identity.',
    'ডেন': 'Declared Denwar is not sufficient to choose one of the registered Danuwar varieties.',
    'তু': 'Source Turkish/Turki label requires modern-language versus generic Turkic scope review; not Proto-Turkic.',
}
KNOWN_NAMES = {
    'অহো': 'Ahom', 'আরা': 'Arakanese', 'ধি': 'Dhimal', 'সিং': 'Singpho',
    'চে': 'Chenchu', 'টো': 'Toto', 'টাক': 'Takpa', 'থা': 'Thai',
    'শা': 'Shan', 'ভা': 'Vayu/Hayu', 'খি': 'Khyang', 'দ্রু': 'Drukpa',
    'বো': 'Bodo', 'লে': 'Lepcha',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'numbered-structured-review.jsonl'
    groups = [json.loads(x) for x in source.read_text().splitlines()]
    declarations = [json.loads(x) for x in (ROOT / 'abbreviations-first-reading.jsonl').read_text().splitlines()]
    independent = json.loads((ROOT / 'independent-abbreviations-full-review-20260927.json').read_text())
    reviewed = {x['source_unit_key']: x for x in independent['rows']}
    decl = defaultdict(list)
    for d in declarations:
        decl[norm(d['abbreviation_first_reading'])].append({
            'key': d['source_unit_key'], 'raw_abbreviation': d['abbreviation_first_reading'],
            'raw_expansion': d['expansion_first_reading'], 'independent_review': reviewed[d['source_unit_key']],
        })
    registry = {r['ID']: r for r in csv.DictReader(REGISTRY.open())}
    assert set(DIRECT.values()) <= registry.keys(), set(DIRECT.values()) - registry.keys()
    out = []
    for group in groups:
        for c in group['comparisons']:
            raw = c['source_language_label']
            label = norm(raw)
            lookup = ALIASES.get(label, label)
            matched = decl.get(lookup, [])
            language = DIRECT.get(lookup)
            reason = REASONS.get(lookup)
            if reason is None and language:
                reason = ('Declared source language corresponds to the existing registry base; no narrower location or dialect inferred.'
                          if matched else 'Explicitly proposed conventional abbreviation expansion; undeclared spelling is retained and requires parent adjudication.')
            if reason is None and lookup in KNOWN_NAMES:
                reason = f'Source declares {KNOWN_NAMES[lookup]}; no safely matching existing base ID established in the local registry. Retain named comparison, pending identity/registry review.'
            if reason is None:
                reason = ('Declaration retained, but a unique existing canonical identity has not been established.' if matched
                          else 'No matching declaration after punctuation-only normalization and explicit alias table; unresolved abbreviated source label. No guessed expansion.')
            reg = registry.get(language)
            status = 'supported base proposal' if language else 'unresolved identity'
            if language and (label in ALIASES or not matched):
                status = 'explicit abbreviation-alias proposal'
            scope = 'outside requested families; retain comparison control' if reg and reg['Clade'] == 'Other' else ('target-family identity proposal' if reg else 'scope unresolved')
            disposition = c['disposition']
            if disposition == 'explicit source blank':
                scope = 'explicit blank control; no lexical row'
            elif disposition == 'unresolved label/form segmentation':
                scope = 'segmentation hold; no lexical emission until adjudicated'
            stage = 'Old Bengali, as declared; no date inferred' if language == 'OB' else ('colloquial Bengali, region unspecified' if label.startswith('কথ্য') else None)
            out.append({
                'comparison_key': c['key'], 'group_key': group['source_unit_key'],
                'raw_label': raw, 'lookup_label': lookup,
                'lookup_policy': 'explicit alias proposal' if label in ALIASES else 'surrounding punctuation only',
                'raw_value': c['raw_source_value'], 'literal_form': c['literal_form'],
                'original_disposition': disposition, 'source_line_keys': c['source_line_keys'],
                'declarations': matched, 'proposed_base_id': language,
                'registry_record': {k: reg[k] for k in ('ID', 'Name', 'Glottocode', 'Clade')} if reg else None,
                'identity_status': status, 'scope_disposition': scope, 'decision_basis': reason,
                'source_stage_or_variety': stage, 'candidate_only_ids': ['bro'] if lookup == 'ব্রো' else [],
                'evidence': ['abbreviations-first-reading.jsonl', 'independent-abbreviations-full-review-20260927.json', 'cldf/languages.csv'] +
                            (['brok-skad-identity-review.json'] if lookup == 'ব্রো' else []) +
                            (['https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-15/'] if lookup == 'কাই' else []),
                'accepted_for_installation': False, 'graph_relation': 'comparison only; no ancestry or borrowing inferred',
            })
    assert len(out) == len({r['comparison_key'] for r in out}) == 395
    assert len(declarations) == len(reviewed) == 88
    assert Counter(r['original_disposition'] for r in out)['explicit source blank'] == 2
    assert Counter(r['original_disposition'] for r in out)['unresolved label/form segmentation'] == 1
    report = {
        'date': '2026-09-27', 'reviewer': 'asur_finish',
        'status': 'complete keyed mapping inventory; identity proposals require parent adjudication; no canonical changes',
        'inputs': {x: sha(ROOT / x) for x in [source.name, 'abbreviations-first-reading.jsonl', 'independent-abbreviations-full-review-20260927.json', 'further-language-mapping-review.json', 'brok-skad-identity-review.json', 'kakmara-identity-research-note.md']},
        'registry_sha256': sha(REGISTRY),
        'coverage': {'groups_in_current_file': len(groups), 'comparison_slots': len(out), 'declarations_reviewed': len(declarations),
                     'status_counts': dict(Counter(x['identity_status'] for x in out)), 'disposition_counts': dict(Counter(x['scope_disposition'] for x in out))},
        'policy': ['All 395 slots retained, including blanks and segmentation hold.', 'Raw declarations remain separate physical records, including duplicate Bengali declarations.',
                   'Historical stages, manuscript labels and dialect ambiguity are not flattened.', 'No new registry entries, coordinates, genealogies, ancestry links or lexical strings created.',
                   'Unsupported undeclared abbreviations remain explicit unresolved evidence, not omissions.',
                   'Documentary demographic table classifications are not promoted into the registry.'],
        'declaration_inventory': [dict(d, independent_review=reviewed[d['source_unit_key']]) for d in declarations],
        'comparators': out,
    }
    (ROOT / 'numbered-language-mapping-proposal.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    label_rows = defaultdict(list)
    for r in out:
        label_rows[r['raw_label']].append(r)
    lines = ['# Numbered comparison identity proposals', '',
             f'All {len(out)} comparison slots in {len(groups)} current groups are accounted for. All 88 physical abbreviation declarations and their independent review are retained. Canonical files are unchanged.', '',
             'Supported IDs are proposals, not installation approval. Two source blanks and one segmentation hold remain controls. Broad groups, historical exonyms, collided labels and manuscript stages are not silently converted to modern languages.', '',
             '| Raw label | Slots | Proposed base | Decision |', '|---|---:|---|---|']
    for label, items in label_rows.items():
        r = items[0]
        lines.append(f"| {label} | {len(items)} | {r['proposed_base_id'] or 'unresolved'} | {r['decision_basis']} |")
    (ROOT / 'numbered-language-mapping-summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(report['coverage'], ensure_ascii=False))


if __name__ == '__main__':
    main()
