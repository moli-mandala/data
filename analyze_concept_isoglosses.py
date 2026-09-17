"""Rank lexical concept contrasts in the compiled Jambu CLDF snapshot (stdlib only).

See stats/dardic-plains/README.md for sampling and score interpretation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from edges_util import load_edges, rank1_map

BASE = Path(__file__).resolve().parent
DARDIC = 'Chitrali,Kashmiric,Kohistani,Kunar,Pashai,Shinaic'
PLAINS = 'Bhil,Bihari,E. Hindi,Eastern,Gujaratic,Halbic,Lahndic,Marathi-Konkani,Punjabic,Rajasthanic,Sindhic,W. Hindi'


def read(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def resolve(fid, forms, parents, include_borrowed=False):
    """Follow accepted ancestry and redirects; never conflate derived/compound heads."""
    seen = set()
    cur = fid
    while True:
        if cur in seen:
            return '', 'cycle'
        if cur not in forms:
            return '', 'missing_node'
        seen.add(cur)
        redirect = forms[cur].get('Redirect', '')
        if redirect:
            cur = redirect
        elif cur != fid and forms[cur]['Language_ID'] in {'Indo-Aryan', 'Sk', 'IA', 'OIA', 'PIA'} and parents.get(cur, ('', ''))[1] != 'variant':
            return cur, 'linked'
        elif cur in parents:
            parent, kind = parents[cur]
            if kind == 'borrowed' and not include_borrowed:
                return '', 'borrowed'
            cur = parent
        else:
            if cur == fid or forms[cur].get('Status') == 'unlinked':
                return '', 'unlinked'
            return cur, 'linked'


def distribution(units):
    """A language has one vote, split equally among its distinct accepted families."""
    counts = Counter()
    linked = 0
    for families in units.values():
        if families:
            linked += 1
            for family in families:
                counts[family] += 1 / len(families)
    return {k: v / linked for k, v in counts.items()} if linked else {}, linked


def contrast(d, p):
    """Best opposing pair by mean directional frequency difference (0–1)."""
    pairs = [(0.5 * (d[a] - p.get(a, 0) + p[b] - d.get(b, 0)), a, b)
             for a in d for b in p if a != b]
    return max(pairs, default=(0, '', ''), key=lambda x: (x[0], x[1], x[2]))


def supported_score(separation, d_support, p_support, target=5):
    """Unknowns are neutral. Require positive focal-family evidence in both groups."""
    support_factor = min(1, d_support / target, p_support / target)
    return max(0, separation) * support_factor, support_factor


def entry_family_map(forms, edges):
    """Collapse explicitly linked numbered CDIAL subheads into their article family.

    This broader option includes subhead derivatives; unrelated derived entries,
    cross-references, compounds and spelling similarities are never merged.
    """
    out = {fid: fid for fid in forms}
    for edge in edges:
        child, parent = edge['Child_ID'], edge['Parent_ID']
        match = re.fullmatch(r'(\d+[a-z]?)-\d+x?', child)
        if (match and match[1] == parent and parent in forms and edge['Rank'] == '1'
                and edge['Kind'] in {'derived', 'reflex', 'variant'}):
            out[child] = parent
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cldf', type=Path, default=BASE / 'cldf')
    parser.add_argument('--out', type=Path, default=BASE / 'stats/dardic-plains')
    parser.add_argument('--dardic-clades', default=DARDIC)
    parser.add_argument('--plains-clades', default=PLAINS)
    parser.add_argument('--exclude-languages', default='D', help='Comma-separated IDs; Domaaki excluded by default')
    parser.add_argument('--include-borrowed', action='store_true')
    parser.add_argument('--family-level', choices=['head', 'entry'], default='entry',
                        help='head: separate OIA headwords (explicit variants already merged); entry: also group numbered CDIAL subheads')
    parser.add_argument('--min-languages', type=int, default=5, help='Minimum linked language units per group')
    parser.add_argument('--support-target', type=float, default=5, help='Full support at this many fractional language votes for each focal family')
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    if args.min_languages < 1 or args.top < 1 or not math.isfinite(args.support_target) or args.support_target <= 0:
        parser.error('min-languages, top and support-target must be positive and finite')
    groups = {'dardic': set(args.dardic_clades.split(',')), 'plains': set(args.plains_clades.split(','))}
    if groups['dardic'] & groups['plains']:
        parser.error('Group clades must not overlap')
    forms = {r['ID']: r for r in read(args.cldf / 'forms.csv')}
    languages = {r['ID']: r for r in read(args.cldf / 'languages.csv')}
    concepts = {r['ID']: r['Name'] for r in read(args.cldf / 'concepts.csv')}
    edges = load_edges(str(args.cldf / 'edges.csv'))
    parents = rank1_map(edges)
    family_map = entry_family_map(forms, edges) if args.family_level == 'entry' else {fid: fid for fid in forms}
    excluded = set(args.exclude_languages.split(','))
    selected = {}
    for lid, lang in languages.items():
        if lid in excluded or lang['Name'].startswith(('Old ', 'Middle ', 'Proto-')):
            continue
        for group, clades in groups.items():
            if lang['Clade'] in clades:
                # Collapse duplicated language codes / survey parents sharing a Glottocode.
                selected[lid] = (group, lang['Glottocode'] or lid)
    units = defaultdict(lambda: {'dardic': defaultdict(set), 'plains': defaultdict(set)})
    evidence = defaultdict(list)
    stale = 0
    seen = set()
    resolved = {}
    for link in read(args.cldf / 'form_concepts.csv'):
        fid, cid = link['Form_ID'], link['Concept_ID']
        if (fid, cid) in seen:
            continue
        seen.add((fid, cid))
        if fid not in forms:
            stale += 1
            continue
        row = forms[fid]
        if row['Language_ID'] not in selected:
            continue
        group, unit = selected[row['Language_ID']]
        if fid not in resolved:
            resolved[fid] = resolve(fid, forms, parents, args.include_borrowed)
        head, status = resolved[fid]
        family = family_map.get(head, head)
        bucket = units[cid][group][unit]
        if family:
            bucket.add(family)
        evidence[cid].append(dict(concept_id=cid, concept=concepts[cid], group=group,
            language_id=row['Language_ID'], language=languages[row['Language_ID']]['Name'],
            unit=unit, form_id=fid, form=row['Form'], gloss=row['Gloss'],
            head_id=head, head=forms[head]['Form'] if head else '',
            family_id=family, family=forms[family]['Form'] if family else '',
            status=status, source=row['Source']))
    results = []
    for cid, samples in units.items():
        d, nd = distribution(samples['dardic'])
        p, np = distribution(samples['plains'])
        separation, a, b = contrast(d, p)
        if not a or separation <= 0:
            continue
        coverage = {g: sum(bool(v) for v in s.values()) / len(s) if s else 0
                    for g, s in samples.items()}
        # Unknown etyma do not oppose a contrast. Support comes from focal-family votes.
        score, support_factor = supported_score(separation, d[a] * nd, p[b] * np, args.support_target)
        legacy_score = separation * math.sqrt(coverage['dardic'] * coverage['plains']) * min(1, nd / 10, np / 10)
        results.append(dict(concept_id=cid, concept=concepts[cid], score=round(score, 6),
            legacy_coverage_score=round(legacy_score, 6),
            support_factor=support_factor, dardic_positive_votes=d[a] * nd, plains_positive_votes=p[b] * np,
            separation=round(separation, 6), eligible=min(nd, np) >= args.min_languages,
            dardic_family_id=a, dardic_family=forms[a]['Form'],
            plains_family_id=b, plains_family=forms[b]['Form'],
            dardic_linked=nd, plains_linked=np,
            dardic_observed=len(samples['dardic']), plains_observed=len(samples['plains']),
            dardic_coverage=coverage['dardic'], plains_coverage=coverage['plains'],
            dardic_own=d[a], plains_own=p[b], dardic_cross=d.get(b, 0), plains_cross=p.get(a, 0),
            dardic_distribution=d, plains_distribution=p))
    results.sort(key=lambda r: (-r['score'], r['concept']))
    for rank, row in enumerate(sorted(results, key=lambda r: (-r['legacy_coverage_score'], r['concept'])), 1):
        row['legacy_rank'] = rank
    for rank, row in enumerate(results, 1):
        row['rank'] = rank
    args.out.mkdir(parents=True, exist_ok=True)
    files = ['forms.csv', 'edges.csv', 'languages.csv', 'concepts.csv', 'form_concepts.csv']
    payload = dict(metric='resolved-separation-with-positive-support-v3',
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        inputs={f: hashlib.sha256((args.cldf / f).read_bytes()).hexdigest() for f in files},
        stale_concept_links=stale, selected_languages={k: dict(group=v[0], unit=v[1], name=languages[k]['Name']) for k, v in selected.items()},
        results=results)
    (args.out / 'results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    flat = [{k: v for k, v in r.items() if not k.endswith('_distribution')} for r in results]
    for name, rows in [('ranking.csv', flat), ('evidence.csv', [r for cid in sorted(evidence) for r in evidence[cid]])]:
        with (args.out / name).open('w', encoding='utf-8', newline='') as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    chosen = [r for r in results if r['eligible']][:args.top]
    chosen += [r for r in results if r['concept'] in {'LONG', 'RIVER'} and r not in chosen]
    lines = ['# Dardic / Plains lexical contrast candidates', '',
        'Exploratory corpus results, not categorical isoglosses. See README.md for definitions and caveats.', '',
        '| Concept | Dardic family | Plains family | Own shares D/P | Cross shares D/P | Linked/observed D; P | Score |',
        '|---|---|---|---|---|---|---|']
    for r in chosen:
        lines.append(f"| {r['concept']} | {r['dardic_family']} ({r['dardic_family_id']}) | {r['plains_family']} ({r['plains_family_id']}) | {r['dardic_own']:.0%} / {r['plains_own']:.0%} | {r['dardic_cross']:.0%} / {r['plains_cross']:.0%} | {r['dardic_linked']}/{r['dardic_observed']}; {r['plains_linked']}/{r['plains_observed']} | {r['score']:.3f} |")
    for r in chosen:
        lines += ['', '## ' + r['concept']]
        for group, family in [('dardic', r['dardic_family_id']), ('plains', r['plains_family_id'])]:
            examples = []
            used = set()
            for e in sorted(evidence[r['concept_id']], key=lambda e: (e['language'], e['form_id'])):
                if e['group'] == group and e['family_id'] == family and e['unit'] not in used:
                    used.add(e['unit'])
                    examples.append(f"{e['language']} **{e['form']}** ‘{e['gloss']}’ [{e['form_id']}; {e['source']}]")
            lines.append(f"- {group.title()}: " + '; '.join(examples[:6]))
    lines += ['', f'Stale form–concept links skipped: {stale}. Full evidence includes unlinked and excluded-borrowing records.']
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:7 + len(chosen)]))
    print(f'\nSaved {len(results)} contrasts and {sum(map(len, evidence.values()))} evidence rows to {args.out}')


if __name__ == '__main__':
    main()
