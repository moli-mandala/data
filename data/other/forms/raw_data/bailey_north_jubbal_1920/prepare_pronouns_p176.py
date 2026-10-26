"""Owner first reading of the complete pronoun page 176; not installed."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'whole-source-recovery'


def main():
    cases = [('nom-acc','Nom., Acc.',['nom','acc']), ('gen','Gen.',['gen']),
             ('dat','Dat.',['dat']), ('abl','Abl.',['abl']), ('agent','Agent',['erg'])]
    tables = [
        ('1sg','I', [['āŭ̃'],['mērū'],['mũ','mũ khi'],['mũ khu'],['mūẽ']]),
        ('1pl','we', [['āmme'],['mā’rū'],['amu khi'],['amu khu'],['āmmĕ']]),
        ('2sg','thou', [['tū'],['tērū'],['tāŭ̃','ta khi'],['ta khu'],['taũe']]),
        ('2pl','you', [['tūē'],['tā’rū'],['tamu','tamu khi'],['tamu khu'],['tūē']]),
        ('3sg','he; she; that', [['ŏsŏ'],['tĕs rū','tē rū'],['tē','tē khi','tĕs','tĕs khi'],['tē khu','tĕs khu'],['tĕnĕ']]),
        ('3pl','they; those', [['ŏsŏ'],['tīndrū'],['tīnd khi'],['tīnd khu'],['tēyĕ']]),
    ]
    records = []
    for person, gloss, cells in tables:
        for (case,label,tags), forms in zip(cases,cells):
            caveats = []
            if person.startswith('2') and case == 'nom-acc':
                label = 'Nom., Act.'
                caveats.append('Source prints Act., unlike parallel Acc. headings. Accusative interpretation is provisional and must be independently checked; raw label retained.')
            if (person,case) in [('1sg','nom-acc'),('2sg','dat')]:
                caveats.append('u appears to carry breve plus tilde; stacked-mark reading requires independent glyph review.')
            records.append({'source_unit_key':f'bailey1920northjubbal:p176:personal:{person}:{case}',
                            'printed_page':176,'pdf_page':202,'forms':forms,'gloss':gloss,
                            'source_case':label,'tags':['pron',person] + tags,
                            'typed_review_caveats':caveats,
                            'review':'owner-original-first-reading; not installed'})
    records.append({'source_unit_key':'bailey1920northjubbal:p176:personal:teh-alternative',
                    'printed_page':176,'pdf_page':202,'forms':['tēh'],'gloss':'that',
                    'tags':['pron','3sg','obl'],
                    'source_analysis':'Explicit alternative for tē. Do not automatically duplicate into every case phrase or invent unattested combinations.',
                    'review':'owner-original-first-reading; not installed'})
    assert len(records) == 31
    assert sum(len(r['forms']) for r in records) == 39
    extra_tables = [
        ('distal-feminine','that', ['pron','3sg','f'], cases[1:],
         [['tīã ru'],['tīã khi'],['tīã khu'],['tīõ']]),
        ('proximal-singular','this', ['pron','sg'], cases,
         [['ēdzā'],['ēh rū','ĕs rū'],['ē','ēh khi','ĕs','ĕs khi'],['ē khu','ĕs khu'],['ēne']]),
        ('proximal-plural','these', ['pron','pl'], cases,
         [['ēdze'],['īnd rū'],['īn','īn khi'],['īn khu'],['ēyĕ']]),
        ('proximal-feminine','this', ['pron','sg','f'], cases,
         [['ēdze'],['ī̃au ru'],['ī̃au khi'],['ī̃au khu'],['ī̃au']]),
        ('interrogative','who', ['pron','interr','sg','pl'], cases[:2],
         [['kūṇ'],['kauh ru']]),
        ('relative-singular','who (relative)', ['pron','rel','sg'], cases,
         [['dzū','jū'],['jē ra','jĕs ra'],['jē khi','jĕs khi'],['jē khu','jĕs khu'],['jĕnē']]),
        ('relative-plural','who (relative)', ['pron','rel','pl'], cases,
         [['jē'],['jīnd ru'],['jīn','jīn khi'],['jīn khu'],['jēye']]),
    ]
    for section, gloss, base_tags, headings, cells in extra_tables:
        for (case,label,tags), forms in zip(headings,cells):
            caveats = []
            if section == 'distal-feminine' and case == 'agent':
                caveats.append('Final o mark read as tilde in enlarged original; independent glyph review pending.')
            if section == 'proximal-feminine' and case != 'nom-acc':
                caveats.append('Initial i bears macron and tilde; preserve both in source transcription.')
            if section.startswith('relative') and case == 'nom-acc':
                case,label,tags = 'nom','Nom.',['nom']
            records.append({'source_unit_key':f'bailey1920northjubbal:p176:{section}:{case}',
                            'printed_page':176,'pdf_page':202,'forms':forms,'gloss':gloss,
                            'source_case':label,'tags':base_tags + tags,
                            'typed_review_caveats':caveats,
                            'source_analysis': 'Source explicitly states plural same as singular.' if section == 'interrogative' else '',
                            'review':'owner-original-first-reading; not installed'})
    assert len(records) == 62
    assert len({r['source_unit_key'] for r in records}) == len(records)
    report = {'status':'complete page owner first reading; independent audit pending',
              'reviewer':'root','evidence':'Original full page at3x; all paradigms at8x in overlapping crops.',
              'source_units':records,
              'remaining_on_page':[],
              'policy':'Combined case headings remain one cell. Each explicit alternative retained. No unseen paradigms generated from prose. Agent is provisionally represented as erg with raw label retained.'}
    (OUT / 'pronouns-p176-first-reading-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"{len(records)} source units /{sum(len(r['forms']) for r in records)} forms; page first reading complete; no installation")


if __name__ == '__main__':
    main()
