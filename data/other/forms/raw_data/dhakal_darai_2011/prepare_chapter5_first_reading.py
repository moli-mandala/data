"""Explicit Chapter 5 first reading; no automatic acceptance or installation.

Line maps follow the complete native-page snapshot. Original-image checking,
interlinear token alignment, prose coverage and grammar interpretation are pending.
"""
import json
import re
import unicodedata
from pathlib import Path

P = Path(__file__).resolve().parent
SURFACES = {
125: [(13,21),(23,28),(30,35),(37,41)],
130: [(5,[14,15]),(17,35)], 131: [(66,74)],
132: [(4,13),(15,[29,30]),(32,38),(45,49),(52,59),(62,71)],
133: [(5,12),(14,21),(34,49),(51,59),(62,72)],
134: [(4,16),(25,32),(34,40),(47,55),(57,66)],
135: [(15,32),(34,53),(56,68)],
136: [(4,[15,16]),(26,36),(39,49),(60,71)],
137: [(4,17),(20,31),(33,44)],
138: [(54,[62,63]),(66,79),(82,89)],
139: [(4,10),(18,[39,40]),(53,65)],
140: [(4,16),(21,34),(36,46),(52,64),(66,83),(85,103)],
141: [(4,12),(18,31),(34,48),(60,74),(76,86)],
142: [(4,17),(22,27),(30,38),([41,42],58)],
143: [(5,17),(19,25),(33,38),(40,47),(57,67)],
144: [(5,19),(22,39),(41,52),(55,61),(63,67)],
}
# Each pair is explicit printed head line and quoted English line(s).
LISTS = {
126: [(16,17),(19,20),(21,23),(24,26),(27,28),(29,30),(31,32),(33,34),(35,36),(37,38),(39,40),(41,42),(43,45)],
127: [(5,6),(7,8),(9,10),(12,13),(14,15),(16,17),(18,19),(26,27),(28,28),(29,30),(31,33),(34,35),(36,37),(38,39),(40,41),(49,50),(51,52),(53,54),(55,56),(57,58),(59,60),(61,62),(63,64)],
128: [(3,4),(5,6),(13,15),(16,18),(19,21),(22,25),(26,28),(29,31),(32,34),(35,37),(38,40),(41,42),(43,46),(47,50),(51,54),(55,57),(58,60),(61,64),(65,67),(68,70),(71,73)],
129: [(5,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,17)],
130: [(39,40),(41,42),(43,45),(46,47),(48,49),(50,52),(53,54),(55,56),(57,58),(59,61),(62,63),(64,66),(67,69),(70,71),(72,73),(74,75),(76,77),(78,79),(80,81)],
131: [(4,6),(7,8),(9,10),(11,12),(13,14),(15,17),(18,19),(20,22),(35,36),(37,38),(44,45),(46,47),(49,50),(51,52),(54,55),(57,58),(62,63)],
137: [(61,62),(63,64),(67,68),(69,[70,71]),(73,[74,75]),(76,[77,78]),(79,[80,81]),(82,[83,84]),(86,87),(88,89),(90,91),(92,93)],
138: [(16,17),(18,27),(28,29),(31,41),(42,43),(45,47)],
}
PROSE = [
 (125,[42],'dzəbbəḍ','urgent','adj',''), (125,[43],'kam','work','noun',''),
 (129,[24],'ek','one','num',''),(129,[25],'-ṭa','classifier','suffix classifier',''),
 (132,[40],'səbei','all','quantifier','Source describes both attributive and independent NP use.'),
 (133,[28,30],'ijə','this','demonstrative determiner prox',''),
 (133,[28,30],'uwə','that','demonstrative determiner dist',''),
 (134,[17],'ijə','this','demonstrative determiner prox',''),
 (134,[17],'uwə','that','demonstrative determiner dist',''),
 (134,[18],'-həi','emphasis','suffix emph',''),
 (135,[10,11],'etka','this much/this number','determiner prox quantifier',''),
 (135,[10,11],'etkai','this much/this number','determiner prox quantifier',''),
 (135,[10,11],'otka','that much/that number','determiner dist quantifier',''),
 (135,[10,11],'otkai','that much/that number','determiner dist quantifier',''),
 (136,[19,20],'ebḍə','this much big','determiner prox',''),
 (136,[20],'obḍə','that much big','determiner dist',''),
 (136,[20,21],'ebḍə','this much big','determiner prox','Second explicit occurrence describes interactional context.'),
 (136,[21,22],'obḍə','that much big','determiner dist','Second explicit occurrence describes distal context.'),
 (136,[52],'esnə','of this type','determiner prox','Source says it modifies nouns and sometimes verbs.'),
 (136,[52],'osnə','of that type','determiner dist','Source says it modifies nouns and sometimes verbs.'),
 (136,[54],'esnə','of this type','determiner prox',''),
 (136,[55,56],'osnə','of that type','determiner dist',''),
 (137,[97],'ḍʰilaha','slow','adj',''),(137,[97],'cʰiṭo','quick','adj',''),
 (137,[99],'ɡarə','difficult','adj',''),(137,[99],'əskərə','difficult','adj',''),
 (137,[100],'sədzilo','easy','adj',''),
 (139,[12,13],'-ei','emphasis','suffix emph','Source says this qualifies intensity or sometimes approximation.'),
 (139,[12,13],'əi','emphasis','affix emph','Second source alternative lacks printed initial hyphen; functional claim is emphatic clitic.'),
 (139,[44],'-se','comparison marker','suffix degree',''),
 (139,[44],'kəte','than','degree',''),(139,[45],'bahun','Brahmin','',''),
 (139,[46],'məi','I','pron 1sg',''),(139,[46],'bəḍkə','big','adj',''),
 (139,[47],'bəlijə','strong','adj',''),(139,[47],'hame','we','pron 1pl',''),
 (139,[47],'toi','you','pron',''),
 (140,[17,18],'-se','comparison marker','suffix degree','Source says it is more frequent with human nouns and the comparative markers are interchangeable.'),
 (140,[47,48],'dzəsnə','like','degree','Source describes symmetrical comparison.'),
 (140,[48],'dzəhin','like','degree','Source describes symmetrical comparison.'),
 (140,[48],'səri','equal to','degree','Source describes symmetrical comparison.'),
 (141,[13],'səbse','all-from','degree','Source describes superlative construction with this form before the head.'),
 (141,[14],'səb','all','quantifier','Standard of comparison in the source superlative description.'),
 (142,[69],'ramrə','good','adj',''),(143,[30],'aṭə-','','stem copula',''),
 (143,[30],'rəhə-','','stem copula',''),
 (143,[50,51],'kancʰi','the youngest','adj','Source contrasts adjectival and nominal use in the following examples.'),
 (145,[3],'-ə','adjective ending','suffix','Contemporary ending contrasted with Hodgson historical list.'),
 (145,[4],'-o','adjective ending','suffix','Historical Hodgson comparison; no automatic Chitwan attribution.'),
]


def build():
    pages = {p['printed_page']: p for p in map(json.loads, (P/'whole-native-pages.jsonl').open()) if p['printed_page'] and 125 <= p['printed_page'] <= 145}
    def txt(n, i):
        return ''.join(s['decoded'] for s in pages[n]['lines'][i-1]['spans']).strip()
    def normalize(t):
        return unicodedata.normalize('NFC', re.sub(r'\s+', ' ', t).strip())
    result = []
    for n, pairs in SURFACES.items():
        for f, g in pairs:
            fs = f if isinstance(f,list) else [f]
            gs = g if isinstance(g,list) else [g]
            translation = ' '.join(txt(n,i) for i in gs)
            form = normalize(' '.join(txt(n,i) for i in fs))
            result.append({'source_unit_key':f'dhakal2011:p{n}:ch5:surface:l{fs[0]}',
                'printed_page':n,'pdf_page':n+22,'kind':'whole-expression',
                'surface_lines':fs,'translation_lines':gs,'form':form,
                'raw_translation':translation,
                'gloss':re.sub(r'\s*\([A-Za-z][A-Za-z.]+\.\d+\)','',translation).strip('‘’\'" '),
                'text_ids':re.findall(r'\(([A-Za-z][A-Za-z.]+\.\d+)\)',translation),
                'source_surface_evidence':[pages[n]['lines'][i-1] for i in fs],
                'source_translation_evidence':[pages[n]['lines'][i-1] for i in gs],
                'interlinear_evidence':pages[n]['lines'][max(fs):min(gs)-1],
                'status':'Native-text first reading; original-page and interlinear review pending',
                'unresolved_glyphs':re.findall(r'⟦U\+[0-9A-F]+⟧',form)})
    for n, pairs in LISTS.items():
        for f,g in pairs:
            gs = g if isinstance(g,list) else [g]
            form = txt(n,f)
            if f in gs:
                form = re.split('[‘\']',form,maxsplit=1)[0].strip()
            raw = ' '.join(txt(n,i) for i in gs)
            match = re.search('[‘\'](.*)',raw)
            gloss = match.group(1).strip('’\' ') if match else raw
            result.append({'source_unit_key':f'dhakal2011:p{n}:ch5:list:l{f}',
                'printed_page':n,'pdf_page':n+22,'kind':'listed-head-group',
                'form':normalize(form),'gloss':gloss,
                'source_lines':sorted(set([f]+gs)),
                'evidence':[pages[n]['lines'][i-1] for i in sorted(set([f]+gs))],
                'status':'Native-text first reading; explicit slash alternatives require separate attestation review',
                'tags':[]})
    # Preserve the printed duplicated 5 label rather than silently repairing six.
    for index, start in enumerate(range(33,61,3)):
        label=txt(129,start)
        for offset,kind in [(1,'cardinal'),(2,'ordinal')]:
            result.append({'source_unit_key':f'dhakal2011:p129:ch5:numeral:l{start+offset}',
                'printed_page':129,'pdf_page':151,'kind':kind,'form':normalize(txt(129,start+offset)),
                'printed_numeral_label':label,'table_row':index+1,
                'evidence':pages[129]['lines'][start-1:start+2],
                'status':'Native-text first reading; table labels retained literally',
                'uncertainty':'Source repeats label 5 for the sixth row; do not silently treat printed label as six.' if index==5 else ''})
    # Comparative historical witnesses are not automatically Chitwan forms.
    for old,new in [(10,12),(13,15),(16,18),(19,21)]:
        gloss=re.search("'(.*)'",txt(145,new)).group(1)
        for line,witness in [(old,"Hodgson's (1985) list, as cited by Dhakal"),(new,'Word list today')]:
            result.append({'source_unit_key':f'dhakal2011:p145:ch5:comparison:l{line}',
                'printed_page':145,'pdf_page':167,'kind':'attributed-comparison',
                'form':normalize(txt(145,line).split("'")[0]),'gloss':gloss,'witness':witness,
                'dialect_policy':'No automatic Chitwan assignment for historical witness' if line==old else 'Thesis default Chitwan',
                'evidence':[pages[145]['lines'][line-1]],'status':'Native-text first reading; original review pending'})
    occurrences={}
    for n,lines,form,gloss,tags,note in PROSE:
        pair=(n,lines[0]);occurrences[pair]=occurrences.get(pair,0)+1
        result.append({'source_unit_key':f'dhakal2011:p{n}:ch5:prose:l{lines[0]}:a{occurrences[pair]}',
            'printed_page':n,'pdf_page':n+22,'kind':'prose-attestation','form':form,'gloss':gloss,
            'tags':tags.split(),'note':note,'source_lines':lines,
            'evidence':[pages[n]['lines'][i-1] for i in lines],
            'status':'Native-text first reading; original-page review pending'})
    assert len({r['source_unit_key'] for r in result})==len(result)
    return result


if __name__=='__main__':
    rows=build()
    (P/'chapter5-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(len(rows),'first-reading groups; prose/interlinear accounting and original review pending; no canonical installation')
