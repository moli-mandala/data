"""Validate independent manual child witnesses against the owner physical census."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[7]
OUT=Path(__file__).parent
PKG=OUT.parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

DOCUMENTARY={
 '561:shared-introduction:1':'Printed umbrella KŌLĀMĪ, NAIKĪ, ETC.; shared classification with Kui/Gondi/Telugu/Kanarese-Tamil and named Basim/Chanda. Documentary classification, no lexical form paired with meaning.',
 '561:kolami-distribution:1':'Hislop geographical description; Wun estimate17,000 contrasted with 1901Kolam264. Wardha census1,505. Historical points not lect coordinates.',
 '561:community-classification:1':'Haig social classification attributed; no target lexical quotation. Preserve as documentary, no ancestry inference.',
 '561:linguistic-classification:1':'Kolami contrasted with Gondi, Telugu, Kanarese, Toda. Historical classification hypotheses remain attributed, no lexical form.',
 '561:pusad-population:1':'A few speakers probably Pusad/Basim; nearly-identical description qualified. Not proof all local Bhils speak one lect.',
 '561:population-table:1':'Wun17,000.', '561:population-table:2':'Amraoti4,500.',
 '561:population-table:3':'Wardha1,600.', '561:population-table:4':'Total23,100.',
 '562:authorities:1':'Hislop, Rev. Stephen; Papers relating to the Aboriginal Tribes of the Central Provinces; edited with notes and preface by R. Temple; [Nagpore]1866; Note on tribe Parti pp10andf; Vocabulary Partii pp1andff. Reported authority only.',
 '562:authorities:2':'Haig, Captain Wolseley; A Comparative Vocabulary of the Gōṇḍī and Kōlāmī Languages; Journal Asiatic Society Bengal Vol.lxvi Parti1897 pp185andff. Reported authority only.',
 '562:source-basis:1':'Notes based on printed parable, two other versions and standard wordlist; only one translated; Grierson supplied interlinear. Unprinted materials are not invented wordlist witnesses.',
 '562:principal-features:1':'Principal features can be ascertained, qualified by limited materials.',
 '562:gender:1':'No feminine noun instances; Naiki feminine/neuter singular same, probably also Kolami. No forms printed; no feminine forms invented.',
 '562:pronoun-number-caveat:1':'Plural pronouns apparently seldom used; singular substituted.',
 '562:verb-number-caveat:1':'Plural and singular interchange in actual use; examples of all forms difficult.',
 '563:further-details:1':'Pointer to following specimen; no new lexical witness.',
 '565:basim-introduction:1':'Below rule only. Bhils of Pusad/Basim or at least some nearly identical Kolami;375speakers estimated but unknown whether all same dialect. Continuation on566.',
 '566:basim-introduction-continuation:1':'Received Basim specimen belongs here and agrees with above Kolami; demographic tally not linguistic IA Bhili claim.',
 '566:verbs-introduction:1':'Conjugation presents points of interest; no printed new form.',
 '567:summary:1':'Basim agrees in many preceding and other features with Kolami; pointer to followingparable. No additional form.'}

def main():
    census_path=PKG/'scope/physical-unit-census.jsonl'
    census=[json.loads(x) for x in census_path.read_text().splitlines()]
    selected={r['source_unit_key']:r for r in census if r['printed_page'] in (561,562,563,566,567) or (r['printed_page']==565 and r['section']=='basim-introduction')}
    rows=[];by_parent=defaultdict(list)
    for r in csv.DictReader((OUT/'literal-witnesses.tsv').open(),delimiter='\t'):
        parent='grierson1906kolamibasim:p'+r.pop('parent')
        assert parent in selected,parent
        n=len(by_parent[parent])+1
        qualification=r.pop('qualification')
        uncertainties=[]
        if any(s in qualification.lower() for s in ('provisional','needs','conflicts')):
            uncertainties=[{'type':'source-numbering' if 'conflicts' in qualification else 'transcription','detail':qualification}]
        unit=selected[parent]
        child={'witness_key':parent+f':independent:w{n}','physical_parent_key':parent,
               **r,'qualification':qualification,'typed_uncertainties':uncertainties,
               'printed_page':unit['printed_page'],'physical_page_onebased':unit['physical_page_onebased'],
               'original_image':unit['source_image'],'original_image_sha256':unit['source_image_sha256'],
               'review_status':'independent original child reading; reconciliation pending'}
        assert child['form'] and not any(c in child['form'] for c in '\ufffd')
        rows.append(child);by_parent[parent].append(child)
    assert len({r['witness_key'] for r in rows})==len(rows)
    coverage=[]
    for key,unit in selected.items():
        shorthand=key.split(':p',1)[1]
        assert by_parent[key] or shorthand in DOCUMENTARY,('unaccounted',key)
        coverage.append({'physical_parent_key':key,'unit_type':unit['unit_type'],
          'printed_page':unit['printed_page'],'child_witness_keys':[r['witness_key'] for r in by_parent[key]],
          'documentary_note':DOCUMENTARY.get(shorthand,''),
          'status':'all quoted children inventoried; residual mark/label holds explicit' if by_parent[key] else 'original documentary/bibliographic material accounted; no lexical emission'})
    assert sum(c['unit_type']=='paradigm-cell' for c in coverage)==58
    for name,data in [('independent-child-witnesses.jsonl',rows),('independent-parent-coverage.jsonl',coverage)]:
        (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    report={'status':'complete bounded independent literal child inventory; reconciliation/acceptance pending',
      'reviewer':'/root/norton_finish','scope':'All grammar, paradigm, bibliography and documentary spans561–563;565below rule–567. No specimen transcription or canonical writes.',
      'source_asset':{'path':'tmp/pdfs/LSI-V4.djvu','sha256':sha(ROOT/'tmp/pdfs/LSI-V4.djvu')},
      'physical_census':{'path':str(census_path.relative_to(ROOT)),'sha256':sha(census_path)},
      'physical_units':len(coverage),'paradigm_cells':58,'child_witnesses':len(rows),
      'per_page':dict(Counter(str(r['printed_page']) for r in rows)),
      'language_labels':dict(Counter(r['language_label'] for r in rows)),
      'residual_typed_cases':[r for r in rows if r['typed_uncertainties']],
      'source_classes':[
        'Parenthetical letters retained as literal forms; no automatic optional-letter expansion.',
        'Pure suffix/person/TAM labels retained separately from lexical definitions.',
        'Each physical pronoun gloss group retained even when several forms share it.',
        'Same physical comparator has separate explicitly labelled language interpretations, not invented separate physical tokens.',
        'Number/gender ambiguity retained for source2.m.; no forced3sg emendation.',
        'A/ū original correspondence corrects earlier preliminary a/ā proposal context; no global substitution.',
        'Source word variants and inconsistent quantities retained per occurrence; no cross-source harmonization.',
        'Unprinted wordlist/parables and zero imperative suffix claims remain documentary.'
      ],
      'controls':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted((OUT/'original-controls').glob('*.png'))],
      'outputs':{n:sha(OUT/n) for n in ('literal-witnesses.tsv','independent-child-witnesses.jsonl','independent-parent-coverage.jsonl')}}
    (OUT/'independent-grammar-inventory-status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('physical_units','paradigm_cells','child_witnesses','per_page','language_labels')},ensure_ascii=False))
if __name__=='__main__':main()
