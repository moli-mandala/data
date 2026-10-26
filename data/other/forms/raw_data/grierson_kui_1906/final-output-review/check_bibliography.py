import hashlib,json
from pathlib import Path
from pybtex.database import parse_file
import pybtex
P=Path(__file__).resolve().parent.parent
D=next(p for p in P.parents if (p/'tags.py').exists())
candidate=P/'proposed-references.bib'
new=parse_file(str(candidate)); old=parse_file(str(D/'cldf/sources.bib'))
collision=sorted(set(new.entries)&set(old.entries))
assert not collision,collision
assert len(new.entries)==5
formatted={};errors={}
engine=pybtex.PybtexEngine()
for key,entry in new.entries.items():
    try:formatted[key]=engine.format_from_string(entry.to_string('bibtex'),'plain',output_backend='markdown')[3:].strip()
    except Exception as exc:errors[key]=str(exc)
report={'reviewer':'/root/asur_finish','status':'formatter repair required' if errors else 'passed reported-authority bibliography review and formatter',
 'proposal_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
 'canonical_bib_sha256':hashlib.sha256((D/'cldf/sources.bib').read_bytes()).hexdigest(),
 'pybtex_parse':True,'keys':list(new.entries),'key_collisions':collision,
 'actual_reference_formatter':formatted,'reference_formatter_errors':errors,
 'original_page_checks':{
 '458':'Frye, John Percival; full grammar/reading title, Cuttack 1851 match visible authority listing.',
 '459':'Lingum Letchmajee title, May/June 1853 original, second edition revised by Descombes and Grierson, Calcutta 1902 all match. Smith J. McD., Handbook, Cuttack 1876 match. Friend-Pereira J. E., Marriage Customs, JASB lxxi/71 part iii 1902 pp18ff and invocation description match.'},
 'evidence_images':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D.parent/'tmp/kui-458.png',D.parent/'tmp/kui-grammar-original-leaf480.jpg']},
 'limits':['These are authorities reported in Grierson; no claim that four original auxiliary works were independently collated.', 'Primary included/provenance fields require final installed counts/paths after acceptance.', 'Actual SourceMeta declarations and formatted-reference validation remain installation-fixture gates.'],
 'canonical_changes':False}
(P/'final-output-review/bibliography-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Five references parse; zero key collisions;',len(errors),'formatter errors; original reported fields agree')
