"""Record bounded original/registry identity evidence; no lexical or canonical writes."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(ROOT / 'data'))
from dialects import dialect_tag

OUT = Path(__file__).parent
def pin(path):
    p = ROOT / path
    return {'path': path, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    langs = {r['ID']: r for r in csv.DictReader((ROOT / 'data/cldf/languages.csv').open())}
    audit_path = 'data/data/other/forms/raw_data/grierson_kui_1906/installed-audit.jsonl'
    held = []
    for line in (ROOT / audit_path).open():
        row = json.loads(line)
        key = row.get('unit_key', row.get('source_key', row.get('key', '')))
        # Find the immutable physical key without assuming an exporter schema.
        if any(f'grierson1906kui:p566:outside-comparand:{i:02}' in line for i in range(1, 6)):
            held.append({'historical_record': row, 'proposed_language_id': 'Naikri',
                         'action': 'Preserve installed hold unchanged; reconcile this exact physical occurrence in the complete new source audit. No persistent form-ID alias is inferred from an audit-only key.'})
    assert len(held) == 5
    assets = [pin('tmp/pdfs/LSI-V4.djvu'), pin('tmp/pdfs/lsi-sprakbanken.xml.bz2'), pin(audit_path),
              pin('data/cldf/languages.csv'), pin('data/cldf/dialects.csv')]
    pages = []
    for p in range(561, 570):
        directory = 'basim-identity-originals' if p in (562,563,565,566,567) else 'kolami-discovery-originals'
        pages.append(dict(printed_page=p, physical_page=p+20,
                          **pin(f'tmp/jambu-next-source/{directory}/printed-{p}.png')))
    lect = {'ID': 'grierson1906_naikri_basim', 'Language_ID': 'Naikri',
            'Source_Language_ID': 'Naikri-Basim-LSI1906', 'Name': 'Bhili of Basim (LSI 1906)',
            'Glottocode': '', 'Latitude': '', 'Longitude': '', 'Clade': langs['Naikri']['Clade'],
            'Location': 'Historical Basim district, Berar; subsection also discusses Pusad', 'Quality': ''}
    lect['Tag'] = dialect_tag(lect['Language_ID'], lect['Source_Language_ID'], lect['Name'])
    result = {
        'status': 'source-grounded routing proposal; root independently accepted identity; no installation',
        'reviewer': '/root/norton_finish', 'date': '2026-09-27',
        'scope': 'Bounded identity/structure review of all original printed pp. 561–569, not full lexical transcription acceptance',
        'assets': assets, 'original_pages_visually_reviewed': pages,
        'primary_bridge': {'existing_bib_key': 'dedr',
          'url': 'https://dsal.uchicago.edu/dictionaries/burrow/frontmatter/frontmatter.html',
          'locator': 'Bibliography §§27–28', 'retrieved': '2026-09-27',
          'short_exact_quote': 'Naikṛi (Nk.), a dialect of Kolami (= LSI, Bhili of Basim).',
          'interpretation': 'Section 27 explicitly equates the historical Basim label with Naikri. Section 28 treats Naiki of Chanda separately; this supports the current distinct canonical routes.'},
        'original_context': [
          {'pages':[561], 'finding':'Kolami/Basim/Chanda are grouped as Dravidian varieties, not as ordinary Indo-Aryan Bhili.'},
          {'pages':[565], 'finding':'Page is shared: upper text continues Kolami specimen 75; the Basim subsection starts below a horizontal rule. Its running header is not a page-wide language attribution.'},
          {'pages':[565,566], 'finding':'The demographic Bhil total does not prove that all Bhils in Pusad/Basim speak this dialect. The actual received Basim specimen is explicitly compared with Kolami.'},
          {'pages':[568,569], 'finding':'Specimen 76 explicitly identifies the Dravidian-family so-called Bhili of district Basim.'}],
        'registry_rows': {k:langs[k] for k in ('Kolami','Naikri','Naiki','Bhili')},
        'proposed_basim_lect': lect,
        'nordic_conflict': {
          'Basim_565_567': {'language':'KOL-Bhili of Basim','modern_label':'Southwestern Kolami-Washim','iso':'nit','latitude':'20.113646','longitude':'77.134602'},
          'shared_heading_561_563': {'language':'Kolami, Naiki etc (KOL)','modern_label':'Southwestern Kolami','iso':'nit','latitude':'20.1022','longitude':'78.4934'},
          'Chanda_570': {'language':'GON-Naiki','modern_label':'Naiki','iso':'nit','latitude':'19.69','longitude':'78.59'},
          'decision':'Catalogue labels/ISO and coordinates do not establish occurrence-level identity or historical site coordinates. Use the explicit DEDR bridge and original sectional boundaries, preserving current distinct canonical bases.'},
        'five_prior_kui_holds': held,
        'excluded_routes': {'Bhili':'Indo-Aryan route contradicted for this specimen by primary source family and DEDR bridge.',
          'Naiki':'Chanda is explicitly distinct in DEDR §28 and the source; preserve its installed identity.',
          'Kolami':'Source umbrella relation is retained in provenance, but DEDR explicitly identifies the Basim subtype as Naikri, which already has a canonical base.'},
        'limitations': ['No claim of modern lect equivalence, exact village or geocoordinates.',
          'No registry merge, ancestry edge, inherited lexical identity, or phonemic assertion follows from this decision.',
          'The five old audit-only keys have no emitted persistent form IDs to alias.',
          'Source-wide census, first readings, independent output audit, metadata/parser and installation gates remain separate.']}
    (OUT/'basim-identity-adjudication-20260927.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('Recorded 9 original pages, 4 registry routes and 5 unchanged historical holds.')

if __name__ == '__main__':
    main()
