"""Recover physical appendix paradigm cells; source-stage candidates only."""
import json
import re
import unicodedata
from pathlib import Path

P = Path(__file__).resolve().parent


def build():
    pages = {}
    with (P / 'whole-native-pages.jsonl').open() as stream:
        for line in stream:
            page = json.loads(line)
            if page.get('printed_page') == 367:
                pages[367] = page

    def text(n):
        return unicodedata.normalize('NFC', ''.join(s['decoded'] for s in pages[367]['lines'][n-1]['spans']).strip())

    def record(line, kind, form, gloss, evidence_lines, **extra):
        return dict(source_unit_key=f'dhakal2011:p367:appendix3a:l{line}',
                    printed_page=367, pdf_page=389, kind=kind, form=form,
                    gloss=gloss, source_lines=evidence_lines,
                    evidence=[pages[367]['lines'][i-1] for i in evidence_lines],
                    status='Owner original-page review passed; independent review pending', **extra)

    rows = []
    grammar = [text(i) for i in range(14, 19)]
    for line, label in zip(range(20,25), grammar):
        rows.append(record(line,'bound-morpheme',text(line),'',[line,line-6],source_grammar=label))
    roots = [(25,26,'1a'),(31,32,'1a'),(37,38,'1a'),(43,44,'1a'),(49,50,'1a'),
             (57,58,'1a'),(63,64,'1b'),(71,72,'1b'),(77,78,'2'),(85,86,'2'),
             (91,92,'3'),(97,98,'3'),(103,104,'3'),(111,112,'3'),(117,118,'4'),
             (125,126,'4'),(131,133,'5'),(140,141,'5')]
    for root_line, first, verb_type in roots:
        raw = text(root_line)
        match = re.fullmatch(r"(.+?)\s*-‘(.+?)’", raw)
        assert match, raw
        form, gloss = match.groups()
        root = record(root_line,'paradigm-root',form,gloss,[root_line],source_verb_type=verb_type)
        if '/' in form:
            root['source_alternates'] = form.split('/')
            root['note'] = 'Both root spellings are printed together; the inflected cells use bohə only.'
        rows.append(root)
        for offset, label in enumerate(grammar):
            line = first + offset
            rows.append(record(line,'paradigm-inflection',text(line),gloss,[line,root_line,14+offset],
                               source_grammar=label,source_verb_type=verb_type,
                               root_source_unit_key=root['source_unit_key']))
    assert len(rows)==113
    assert len({r['source_unit_key'] for r in rows})==113
    return rows


if __name__ == '__main__':
    rows=build()
    (P/'appendix3a-reviewed-records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(f'{len(rows)} physical cells, 18 roots, 90 inflections, 5 bound markers; no canonical installation')
