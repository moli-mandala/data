"""Exact-key native reading proposals; preserves immutable first readings."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

PINS = {568:'7ac1c77dbb6f03e2f530ef13ce18eb0a2212fa173f97165418ec8d02ce54c2fd',
        569:'a831c8b55fca88ff9c3a1c44f937c3c844932e03a1a2343ab34b0fff15a71f36'}

def main():
    by = {}
    for page, pin in PINS.items():
        path = HERE / f'p{page}-first-reading.jsonl'
        assert sha(path) == pin
        for s in path.read_text().splitlines():
            for atom in json.loads(s)['target_atoms']:
                by[page,atom['physical_line'],atom['physical_word']] = atom
    crops = {(r['printed_page'],r['name']):r for r in json.loads((HERE/'sharp-remaining-glyph-control-locators.json').read_text())}
    corrections = []
    for page,line,word,old,gloss,new,crop,decision in [
        (568,3,5,'ūrn','village','ūru','village-far','root and owner accepted'),
        (568,3,6,'dhān','far','dhāu','village-far','root and owner accepted'),
        (568,5,3,'avande','he','avandē','servants','root and owner accepted'),
        (568,11,6,'chākāudār-laṅgānu','servant-like','chākardār-laṅgānu','servant11','root and owner accepted'),
        (568,17,5,'kērkul','shoes','kērīkuḷ','shoes-isolated','root and owner accepted'),
        (569,4,1,'Sāldārākīn','Servant','Sāldārākūn','servant-said4','root and owner accepted'),
        (569,7,7,'varaha','years','varsha','years7-v2','root and owner accepted'),
        (569,11,6,'īnyetī','thine','inyetī','only11-v2','root and owner accepted'),
    ]:
        atom = by[page,line,word]
        assert atom['raw_source_form'] == old, atom
        assert atom['raw_interlinear_gloss'] == gloss, atom
        corrections.append(dict(source_unit_key=atom['source_unit_key'],field='raw_source_form',
                                old=old,new=new,raw_interlinear_gloss=gloss,decision=decision,
                                evidence=crops[page,crop],accepted_for_installation=False,
                                source_witness_caveat='Sharp alternate source shapes; no demonstrated editorial edition difference.'))
    rejects = []
    for page,line,word,old,hypothesis,crop in [
        (568,3,8,'udhaḷvārī','ūdhaḷvārī','squandering'),
        (568,5,4,'dhandā-takten.','dhāndā-takten.','servants'),
        (568,5,7,'mipen','miyen','servants'),
        (568,7,3,'avandu','āvandu','he7'),
        (568,8,5,'naukarnā','nōkarnā','servants8'),
        (568,10,6,'puṭtān,','puṭkān,','fault10'),
        (568,12,2,'śedden.','śeddēn.','went-saw'),
        (568,12,4,'keiken','kēiken','saw12-v2'),
        (568,13,8,'samten.','sāmtēn.','kiss13'),
        (568,15,9,'gaḍyākūn','gāḍyākūn','servants15'),
        (568,16,4,'‘dhaḍāpan','‘dhaḍēpan','clothes16'),
        (568,16,8,'tepp.','tēpp.','clothes16'),
        (568,17,1,'keit','kēt','hands-shoes'),
        (569,2,1,'Sagḷē','Sāgḷē','all2'),
        (569,3,6,'vatten,','vattēn,','came3'),
        (569,6,2,'vattīn;','vāttīn;','anger6-v2'),
        (569,10,1,'seiten,','sēiten,','giving10-v2'),
    ]:
        atom=by[page,line,word]
        assert atom['raw_source_form'] == old, atom
        rejects.append(dict(source_unit_key=atom['source_unit_key'],retained=old,rejected=hypothesis,
                            evidence=crops[page,crop],reason='Native isolated source shape does not support preliminary broad-image hypothesis.'))
    report=dict(status='All247 physical atoms of p568/p569 visually compared; one small ink-identity uncertainty retained',
                first_reading_sha256={str(p):h for p,h in PINS.items()},
                corrections=corrections,rejected_preliminary_hypotheses=rejects,
                top_mark_policy='Native i/u top marks compared occurrence by occurrence; no phonetic reinterpretation or analogical harmonization.',
                pending=[dict(source_unit_key=by[569,9,7]['source_unit_key'],old='kasbirā',
                              reason='Small raised ink adjoining r may be its shoulder or detached mark; no inserted punctuation without glyph control.',evidence=crops[569,'women9-v2'])],
                crop_manifest_sha256=sha(HERE/'sharp-remaining-glyph-control-locators.json'),
                canonical_changes=False)
    (HERE/'sharp-p568-p569-review-decisions.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__ == '__main__': main()
