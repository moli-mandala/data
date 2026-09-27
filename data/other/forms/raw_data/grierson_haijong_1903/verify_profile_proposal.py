"""Bounded Haijong profile proposal and coverage check; no canonical writes."""
from pathlib import Path
import collections,hashlib,json,sys,unicodedata
import regex
from segments import Tokenizer
from prepare_full_source import prepare_roman_assembly
P=Path(__file__).resolve().parent
DATA=P.parents[4]
sys.path.insert(0,str(DATA))
import profile_policy

def nfc(s):return unicodedata.normalize('NFC',s)
def display(s):return nfc(s.lower().replace('ṅ','ŋ').replace('w','v').rstrip('.?'))
def verify():
 rows,_=prepare_roman_assembly();forms=[nfc(r[2]) for r in rows]
 clusters=collections.Counter(c for s in forms for c in regex.findall(r'\X',s))
 old=DATA/'conversion/grierson-haijong-1903.txt'
 rules={}
 for line in old.read_text().splitlines()[1:]:
  a,b=line.split('\t');rules[a]=b
 for c in clusters:rules[c]='#' if c==' ' else '' if c in '.?' else display(c)
 target=P/'full-profile-proposed.txt'
 target.write_text('Grapheme\tIPA\n'+''.join(k+'\t'+rules[k]+'\n' for k in sorted(rules)))
 tok=Tokenizer(str(target))
 def convert(s):return nfc(tok(nfc(s),column='IPA').replace(' ','').replace('#',' '))
 results=[]
 for row in rows:
  s=row[2]
  assert not any(c in s[:-1] for c in '.?'), ('nonterminal punctuation',s)
  actual=convert(s);expected=display(s)
  assert actual==expected and '�' not in actual,(s,actual,expected)
  results.append({'entry_key':row[10],'original':s,'display':actual})
 checks={'t̲s̲ārābāk':'t̲s̲ārābāk','un̲g̲kāni':'un̲g̲kāni','sāikkhˢāt':'sāikkhˢāt','Īshᵛar-ṭhāi':'īshᵛar-ṭhāi','thākkʸā':'thākkʸā','Ăkra':'ăkra','Nǎn̲g̲':'nǎn̲g̲','gaïnyai':'gaïnyai','Hē̃':'hē̃','jiṅgiyāsē':'jiŋgiyāsē','khāwālē-dāwālē':'khāvālē-dāvālē','Talāk ki nām?':'talāk ki nām','uriyā-phĕlālē':'uriyā-phĕlālē'}
 for s,expected in checks.items():assert convert(s)==expected,(s,convert(s),expected)
 violations=[{'grapheme':k,'output':v,'policy':profile_policy.house_output(k,v,rules,'grierson-haijong-1903')} for k,v in rules.items() if profile_policy.house_output(k,v,rules,'grierson-haijong-1903')!=v]
 assert not violations,violations
 report={'status':'all896candidate_rows_tokenize_and_match_declared_display_policy','candidate_rows':len(rows),'distinct_input_forms':len(set(forms)),'observed_grapheme_clusters':len(clusters),'profile_rules':len(rules),'profile_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'input_form_key_sha256':hashlib.sha256(json.dumps([(r[10],r[2]) for r in rows],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),'canonical_profile_unchanged_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'punctuation_terminal_only_count':sum(s.endswith(('.', '?')) for s in forms),'house_policy_violations':violations,'focused_examples':checks,'inventory':dict(sorted(clusters.items())),'normalization':'NFC before and after tokenizer; complete combining grapheme rules','yaml_requirements':{'transcription':{'profile':'grierson-haijong-1903','convert':True},'preserve_hyphens':True},'deferred':['Canonical profile installation by root','Final installed-form coverage if assembly changes','Full build and browser QA forbidden until user requests database build']}
 (P/'full-profile-proposal-verification-20260926.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['candidate_rows','observed_grapheme_clusters','profile_rules','punctuation_terminal_only_count','profile_sha256']}))
if __name__=='__main__':verify()
