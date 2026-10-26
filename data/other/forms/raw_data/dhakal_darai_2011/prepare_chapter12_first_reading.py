"""Complete Chapter 12 evidence inventory; not an installation or reviewed export."""
import json
import re
import unicodedata
from pathlib import Path

P = Path(__file__).resolve().parent


def build():
    plan = json.loads((P / 'chapter12-first-reading-plan-20260927.json').read_text())
    assert plan['native_pages_read'] == list(range(327, 348))
    pages = {}
    with (P / 'whole-native-pages.jsonl').open() as stream:
        for line in stream:
            page = json.loads(line)
            if page.get('printed_page') in plan['printed_pages']:
                pages[page['printed_page']] = page

    def text(n, i):
        return ''.join(s['decoded'] for s in pages[n]['lines'][i - 1]['spans']).strip()

    def evidence(n, lines):
        return [pages[n]['lines'][i - 1] for i in lines]

    def norm(value):
        return unicodedata.normalize('NFC', re.sub(r'\s+', ' ', value).strip())

    def row(n, key, kind, form, gloss, lines, **extra):
        return dict(source_unit_key=f'dhakal2011:p{n}:ch12:{key}',
                    printed_page=n, pdf_page=n + 22, kind=kind,
                    form=norm(form), gloss=norm(gloss), source_lines=lines,
                    evidence=evidence(n, lines),
                    status='Native-text first reading; original-page review pending', **extra)

    out = []
    for number, pairs in plan['surface_intervals'].items():
        n = int(number)
        for f, g in pairs:
            fs = f if isinstance(f, list) else [f]
            gs = g if isinstance(g, list) else [g]
            raw = ' '.join(text(n, i) for i in gs)
            ids = re.findall(r'\(([A-Za-z][A-Za-z.]+\.\d+(?:-\d+)?)\)', raw)
            gloss = re.sub(r'\s*\([A-Za-z][A-Za-z.]+\.\d+(?:-\d+)?\)', '', raw).strip('‘’\' ')
            raw_form = ' '.join(text(n, i) for i in fs)
            form = raw_form
            out.append(row(n, f'surface:l{fs[0]}', 'whole-expression', form, gloss,
                           fs + gs, surface_lines=fs, translation_lines=gs,
                           raw_surface=raw_form, raw_translation=raw, text_ids=ids,
                           source_surface_evidence=evidence(n, fs),
                           source_translation_evidence=evidence(n, gs),
                           interlinear_evidence=pages[n]['lines'][max(fs):min(gs) - 1]))
            unit = out[-1]
            lookup = f'{n}:{fs[0]}'
            groups = plan.get('translation_groups', {}).get(lookup)
            if groups:
                unit['source_translation_groups'] = [dict(lines=ls, raw=' '.join(text(n, i) for i in ls)) for ls in groups]
            solutions = plan.get('solution_lines', {}).get(lookup)
            if solutions:
                unit['source_solution'] = dict(lines=solutions, raw=' '.join(text(n, i) for i in solutions), evidence=evidence(n, solutions))
            if n == 343 and fs == [58]:
                unit['interlinear_evidence'] = pages[n]['lines'][58:74]
    seen = {}
    for item in plan['prose_candidates']:
        n, lines = item['page'], item['lines']
        k = (n, lines[0])
        seen[k] = seen.get(k, 0) + 1
        out.append(row(n, f'prose:l{lines[0]}:a{seen[k]}', 'prose-attestation',
                       item['form'], item['gloss'], lines, note=item.get('note', '')))
    for table in plan['tables']:
        n = table['page']
        for line, semantic, role in table['rows']:
            raw = text(n, line)
            head, gloss = re.split("[‘']", raw, maxsplit=1)
            out.append(row(n, f'table:l{line}', 'relative-correlative-table', head,
                           gloss.rstrip("’' "), [line], source_category=semantic,
                           source_role=role, source_table='11.2'))
    for item in plan['diagram_repetitions']:
        n, lines = item['page'], item['form_lines']
        out.append(row(n, f'diagram:l{lines[0]}', 'diagram-expression',
                       ' '.join(text(n, i) for i in lines), '', lines,
                       related_expression_key=f"dhakal2011:p{n}:ch12:surface:l{item['same_example_surface']}",
                       note='Separate diagram witness of the following example; no independent gloss supplied.'))
    for index, control in enumerate(plan['controls'], 1):
        for n in control['pages']:
            lines = control.get('lines', [l['line'] for l in pages[n]['lines']])
            out.append(row(n, f'control:{index}', 'source-context-control', '', '', lines,
                           disposition='non-Darai-context-control', note=control['scope']))
    assert len(out) == len({r['source_unit_key'] for r in out})
    return out


if __name__ == '__main__':
    records = build()
    (P / 'chapter12-first-reading.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    print(len(records), 'groups; all 21 pages inventoried; original review pending')
