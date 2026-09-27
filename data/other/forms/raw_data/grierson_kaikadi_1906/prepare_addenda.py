"""Literal later addendum evidence; never rewrite the original-edition table."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[5]
source='grierson_addenda_minora_iv_boundin'
image=ROOT/'tmp/korvi-full-scope/alternate-leaf708.jpg'
table={a['prompt']:a for a in map(json.loads,(P/'table-reviewed.jsonl').read_text().splitlines())}
units=[]
for prompt in [95,146,147,148,149,207,210,211]:
 t=table[prompt];emit=prompt in [146,147,148,149]
 a={'source_unit_key':f'{source}:p18:kaikadi-table:{prompt}','printed_page':18,'section':'later-addendum','source_citation_key':source,'target_original_page':t['printed_page'],'target_prompt':prompt,'source_district':'Sholapur','forms':['nāy' if prompt in [146,147]else 'nāyāṅg']if emit else [],'gloss':t['gloss'],'scope':'target'if emit else 'editorial-evidence','linked_original_entry_key':f'grierson1906lsi4:kaikadi_sholapur:{prompt}','original_edition_forms':t['forms'],'review':'original later addendum inspected independently from the1906 table','source':str(image.relative_to(ROOT)),'source_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'note':'Later printed correction to the1906 table; the original edition reading is retained as a separate witness, not silently replaced. Addendum publication date is unverified.'}
 if prompt==95:
  a['printed_correction_description']='Read a marked a followed by n; the compact upper mark cannot securely be distinguished as macron or breve.'
  a['uncertainty']='Correction vowel mark unresolved (ā/ă); no guessed corrected lexical row emitted.'
 elif prompt in [146,147]:a['printed_correction_description']='for nāi, read nāy'
 elif prompt in [148,149]:a['printed_correction_description']='for nāyaṅg, read nāyāṅg'
 elif prompt==207:a['printed_correction_description']='English He goest corrected to He goes; already retained editorial gloss confirmed.'
 elif prompt==210:a['printed_correction_description']='for hōgākāng, read hōgākāṅg; confirms the dottedṅ in the independently read original scan.'
 elif prompt==211:a['printed_correction_description']='Correct the number; confirms mapping of printed11 to logical211.'
 units.append(a)
(P/'addenda-reviewed.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n'for a in units))
report={'status':'all relevant Kaikadi later addendum statements accounted separately from1906 edition','printed_pages_examined':[17,18],'page17':'Direct original review: no chapter333–342 lexical correction;343Burgandi title change belongs to next chapter. Original order jumps303 to343 to434.','page18_target_prompts':[95,146,147,148,149,207,210,211],'four_new_corrected_attestations':[146,147,148,149],'unresolved':[{'prompt':95,'issue':'ā/ă mark cannot be distinguished securely; exact image and description retained'}],'bibliography':'Separate undated bound-in witness; volume catalog date is not copied onto this later sheet. At least post1922 from reference on adjacentp16. Exact1927SupplementII identity unverified.','origins':[{'file':str(f.relative_to(ROOT)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}for f in [ROOT/'tmp/kaikadi-completion/addenda-leaf706.jpg',image]],'original_1906_inputs_unchanged':True}
(P/'addenda-scope-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
