"""Literal orthographic profile proposal; author IPA stays a separate field."""
import csv
import json
import sys
import unicodedata as ud
from pathlib import Path
import regex
from segments import Tokenizer

P=Path(__file__).resolve().parent
D=P.parents[4]
sys.path.insert(0,str(D))
import profile_policy

rows=list(csv.reader((P/'whole-proposed.csv').open()))
forms=[r[2] for r in rows]
graphemes={g for f in forms for g in regex.findall(r'\X',ud.normalize('NFC',f)) if not g.isspace()}
rules={g:g for g in graphemes}
for g in list(rules):
 rules[g]=profile_policy.house_output(g,g,rules,'kholosi-source')
# Length colons occur only in the unglossed fieldwork text. Preserve that raw
# notation in Original; display uses the explicit house length convention.
length_rules={}
for f in forms:
 for base in regex.findall(r'(\X)[:ː]',f):
  if base.strip() and ud.category(base[0]).startswith('L'):
   length_rules[ud.normalize('NFC',base+':')]=profile_policy.lengthen(rules.get(base,base))
rules.update(length_rules)
for g,o in list(rules.items()):
 rules[ud.normalize('NFD',g)]=o
path=P/'whole-profile-proposed.txt'
path.write_text('Grapheme\tIPA\n'+''.join(g+'\t'+o+'\n' for g,o in sorted(rules.items())))
tokenizer=Tokenizer(str(path))
def display(f):
 return ud.normalize('NFC',tokenizer(ud.normalize('NFC',f).strip('1234⁴5⁵67⁷,;.'),column='IPA').replace(' ','').replace('#',' '))
unknown=[]
for f in forms:
 if '�' in display(f):unknown.append(f)
assert not unknown,unknown
violations=[]
for g,o in rules.items():
 want=profile_policy.house_output(g,o,rules,'kholosi-source')
 if ud.normalize('NFC',want)!=ud.normalize('NFC',o):violations.append([g,o,want])
assert not violations,violations
report=dict(status='literal profile proposal; independent acceptance pending',rows=len(rows),rules=len(rules),unknown_forms=unknown,
 policy_violations=violations,length_rules=length_rules,
 explicit_changes={g:o for g,o in rules.items() if ud.normalize('NFC',g)!=ud.normalize('NFC',o)},
 rationale='Use source orthographic form, not a reconstructed phonemic interpretation. Conventional house consonants and written length colons are normalized for display. Other vowel/notation distinctions remain literal. Source supplied phonetic transcription is preserved separately, including its contrasts with spelling.',
 original_and_phonemic_unchanged=True)
(P/'whole-profile-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('profile rows',len(rows),'rules',len(rules),'unknown',len(unknown),'policy violations',len(violations))
