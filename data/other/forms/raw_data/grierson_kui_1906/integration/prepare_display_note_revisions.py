"""Record exact display-only revisions; retain full source/review evidence."""
import csv,hashlib,json
from pathlib import Path
Q=Path(__file__).resolve().parent
rows=list(csv.reader((Q/'combined-candidate.csv').open()))
bykey={r[10]:r for r in rows}
changes={}
def setnote(k,n,reason='Separate scholarly source facts from editorial review instructions.'):
 r=bykey[k];changes[k]={'entry_key':k,'old_notes':r[6],'new_notes':n,'old_tags':r[14],'new_tags':r[14],'reason':reason}
def key(p,s):return f'grierson1906kui:p{p}:{s}'
for p,n,i in [(652,47,1),(652,51,1),(660,99,2),(672,175,1),(672,175,2)]:
 k=key(p,f'khondmals-table:prompt{n}:v{i}')
 setnote(k,'','Uncertainty belongs only to another printed alternative; this alternative is independently clear.')
 changes[k]['new_tags']=' '.join(t for t in bykey[k][14].split() if t!='uncertain')
for n in range(138,156):
 k=key(668,f'khondmals-table:prompt{n}')
 note='The English prompt contrasts animal sex and number; it does not establish grammatical gender.'
 if n==153:note+=' Source spacing remains uncertain between sitā li and sitāli.'
 setnote(k,note)
setnote(key(652,'khondmals-table:prompt47:v2'),'Source nasal-mark encoding remains uncertain: tāñji or tāñ̄ji. The vowel reading Ābā in the other alternative is independently clear.')
setnote(key(680,'khondmals-table:prompt231'),'The comparative adjective’s nasal mark and vowel quantity remain uncertain.')
for i in [1,2]:
 setnote(key(462,f'pronoun-table2-nom-c4:1:v{i}'),'The printed cell lists ēwi and ēwa (-skā); the parenthetical suffix follows only the second alternative, and its full scope is uncertain.')
setnote(key(463,'causal-example:2'),'Source alternative causal construction. Initial giw- is legible; the following fused vowels and their quantities remain uncertain.')
setnote(key(210,'outside:father'),'The example follows the Kui suffix discussion without a repeated language label. Attribution to Kui is contextual and uncertain; the source locality is unspecified.')
for n in [1,2,3]:setnote(key(291,f'outside:plural-{n}'),'Explicit Kui plural-suffix comparison. The source conjectures an originally neuter use and reports use with rational nouns. Locality is unspecified.')
setnote(key(296,'outside:thou-doest'),'The source describes this indefinite tense as generally present and often also future. This occurrence belongs to its present-tense comparison; Kui locality is unspecified.')
setnote(key(481,'outside:thou'),'The source prints īmu with m, differing from īnu elsewhere; whether this difference is a source typo is unresolved. Kui locality is unspecified.')
setnote(key(210,'outside-comparand:01'),'The source conjectures an originally first-person suffixed pronoun but reports general use without first-person reference. The italic g/j glyph remains uncertain. Source language: Juang.')
for n in [7,8]:setnote(key(413,f'outside-comparand:{n:02}'),'One of two printed Kurukh alternatives in the ablative comparison.')
setnote(key(481,'outside-comparand:16'),'The source describes this as the old singular form, now used as an oblique and verbal base. Source language: Gondi.')
setnote(key(623,'outside-comparand:01'),'Brahui instrumental suffix; the source tentatively compares it with the following from-suffixes.')
setnote(key(290,'outside-comparand:08'),'Whole Gondi inflected example. Mandla is explicitly assigned only to the preceding example; this occurrence has no separate locality label.')
# The Letchmajee shortening notes are authentic reported disagreement, not editor instructions.
report={'status':'Keyed display-note and uncertainty-scope correction proposal','baseline_csv_sha256':hashlib.sha256((Q/'combined-candidate.csv').read_bytes()).hexdigest(),'authority':'../final-output-review/field-audit.json and root final-fix instruction','changes':list(changes.values()),'retained_source_disagreement_keys':[key(459,f'letchmajee-shortening:{n}') for n in [1,2,3]],'policy':'Form, Gloss, Native, Phonemic, keys, graph fields and citations unchanged. Full original notes and physical cell uncertainty remain in audit.'}
(Q/'display-note-revisions.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(len(changes))
