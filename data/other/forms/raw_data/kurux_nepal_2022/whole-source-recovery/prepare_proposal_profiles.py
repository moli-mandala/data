"""Review-only routing, preserving the established phonetic house profile."""
import csv,json,unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent;DATA=HERE.parents[5];OUT=HERE/'proposal'
SOURCE='shackelford-swenson-chaudhary-maggard2022kurux'
def main():
 base=(DATA/'conversion/kurux-nepal.txt').read_text();base=base.replace('\\t','\t')
 entries=dict(line.split('\t',1) for line in base.splitlines()[1:] if line)
 canonical_aliases={}
 for grapheme,value in list(entries.items()):
  for norm in ['NFC','NFD']:
   alias=unicodedata.normalize(norm,grapheme)
   if alias not in entries:canonical_aliases[alias]=value
 entries.update(canonical_aliases)
 base='Grapheme\tIPA\n'+''.join(f'{g}\t{v}\n' for g,v in entries.items())
 existing=set(entries)
 forms=[r[2] for b in ['wordlist','grammar'] for r in csv.reader((OUT/f'{b}.csv').open())]
 chars={c for f in forms for n in ['NFC','NFD'] for c in unicodedata.normalize(n,f)}
 additions=sorted(chars-existing)
 # Unmapped printed characters are preserved literally, not phonologically
 # interpreted. In particular raisedᵊ/ᶺ are never replaced by ordinary vowels.
 (OUT/'kurux-nepal-whole-phonetic.txt').write_text(base.rstrip()+'\n'+''.join(f'{c}\t'+('#' if c==' ' else c)+'\n' for c in additions))
 for bucket in ['native','roman']:
  fs=[r[2] for r in csv.reader((OUT/f'{bucket}.csv').open())]
  cs=sorted({c for f in fs for n in ['NFC','NFD'] for c in unicodedata.normalize(n,f)})
  (OUT/f'kurux-nepal-whole-{bucket}.txt').write_text('Grapheme\tIPA\n'+''.join(f'{c}\t'+('#' if c==' ' else c)+'\n' for c in cs))
 for bucket in ['wordlist','grammar','native','roman']:
  profile='kurux-nepal-whole-'+('phonetic' if bucket in ['wordlist','grammar'] else bucket)
  text=f'''file: {bucket}.csv
defaults:
  transcription:
    profile: {profile}
    preserve_hyphens: true
  identity:
    dedupe_by_entry_key: true
  forms:
    split_alternates: false
  gloss:
    grammar_tags: false
'''
  if bucket=='wordlist':text+=f'''sources:
  {SOURCE}:
    forms:
      split_alternates: false
    gloss:
      grammar_tags: false
    identity:
      dedupe_by_entry_key: true
'''
  (OUT/f'{bucket}.yaml').write_text(text)
 (OUT/'profile-decisions.json').write_text(json.dumps(dict(base_profile='conversion/kurux-nepal.txt',literal_additions=additions,canonical_equivalent_aliases=canonical_aliases,policy='Existing reviewed house conversions remain. Canonically equivalent NFC/NFD graphemes share those same outputs, including nasal-plus-length sequences. Newly recovered raised vowels and unmatched source symbols preserve literal shape, with no inferred phonemic normalization. Native and conventional Roman routes are entirely literal. No source-wide profile override leaks across the four files.'),ensure_ascii=False,indent=2)+'\n')
 print('Profile additions:',repr(additions))
if __name__=='__main__':main()
