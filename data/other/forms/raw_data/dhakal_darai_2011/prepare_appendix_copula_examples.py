"""Original-reviewed copular/possessive examples with manually checked word alignment."""
import json,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
# An integer selects the whole native line; (line, token) selects a physically aligned word.
PLAN={
378:[(9,16,[( (10,0),12),((10,1),13),(11,14)]),(18,25,[((19,0),21),((19,1),22),(20,23)]),
     (27,35,[(28,31),((30,0),(33,0)),((30,1),(33,1))]),(38,45,[(39,41),((40,0),(43,0)),((40,1),(43,1))]),
     (50,60,[(51,55),(52,(56,0)),(53,(56,1)),(54,58)]),
     (62,72,[(63,67),((64,0),(68,0)),((64,1),(68,1)),(66,70)])],
379:[(7,18,[(8,12),(9,13),(10,14),(11,16)]),(20,32,[(21,25),(22,26),(23,28),(24,30)]),
     (37,44,[((38,0),40),((38,1),41),(39,42)]),(46,53,[((47,0),49),((47,1),50),(48,51)]),
     (55,62,[((56,0),58),((56,1),59),(57,60)]),(64,71,[((65,0),67),((65,1),68),(66,69)])],
380:[(6,13,[(7,(10,0)),((8,0),(10,1)),((8,1),(10,2)),(9,11)]),
     (15,23,[((16,0),19),((16,1),(20,0)),(17,(20,1)),(18,21)]),
     (25,34,[(26,30),((28,0),(31,0)),((28,1),(31,1)),(29,32)]),
     (36,45,[(37,41),((39,0),(42,0)),((39,1),(42,1)),(40,43)]),
     (50,62,[(51,56),(53,57),(54,59),(55,60)]),(64,76,[(65,70),(67,71),(68,73),(69,74)])],
381:[(4,16,[(5,10),(7,11),(8,13),(9,14)]),(18,30,[(19,24),(21,25),(22,27),(23,28)])]}
PLAN[383]=[(10,20,[(11,15),(13,16),(14,18)]),(22,32,[(23,27),(25,28),(26,30)]),(37,46,[(38,43),(40,44),(42,45)]),(48,58,[(49,54),(51,56),(53,57)])]
NOTES={
(383,22):'Surface teurake differs from analyzed tohe-ke.',
(383,48):'Surface uṭʰilekan differs from analyzed uṭʰ-tai-kan; full translation and interlinear are nonpast, retained without harmonization.',
(378,18):'Surface tohe and full translation specify you, but interlinear məi is glossed I; both witnesses retained.',
(378,50):'Surface həkhim differs from analyzed həkhə-m; literal vowel difference retained.',
(378,62):'Surface həkhir differs from həkhə-r, and toi-rə is literally glossed I-[OBL]GEN despite your in the translation.',
(379,20):'Whole translation says priest, while cʰeɡri is aligned with goat; contradictory source meanings retained.',
(379,37):'Surface aṭim differs from interlinear aṭə-m.',
(379,46):'Source tohe is glossed I although the whole translation says you; surface aṭiu differs from aṭə-u.',
(379,64):'Source plural they is paired with a copula labeled 3SG.H; not asserted as a consistent person-number analysis.',
(380,15):'Surface aṭir differs from interlinear aṭə-r.',
(380,25):'Surface toherəke and aṭiu differ from analyzed tohe-ke and aṭə-u.',
(380,36):'Source unhen-ke is glossed you while whole translation and copula specify he/3SG.H.',
(381,4):'Surface ukʰrə differs from analyzed u-rə.',
(381,18):'Surface unhekə differs from analyzed unhen-kə.'}

def build():
    pages={}
    for s in (P/'whole-native-pages.jsonl').open():
        r=json.loads(s)
        if r.get('printed_page') in PLAN:pages[r['printed_page']]=r
    out=[]
    for n,examples in PLAN.items():
        def get(ref):
            line=ref if isinstance(ref,int) else ref[0]
            text=unicodedata.normalize('NFC',''.join(s['decoded'] for s in pages[n]['lines'][line-1]['spans']).strip())
            return text if isinstance(ref,int) else text.split()[ref[1]]
        def line(ref):return ref if isinstance(ref,int) else ref[0]
        for surface,translation,words in examples:
            for index,(f,g) in enumerate([(surface,translation)]+words):
                kind='whole-expression' if index==0 else 'aligned-word'
                ls=sorted({line(f),line(g)})
                out.append(dict(source_unit_key=f'dhakal2011:p{n}:appendix3{"f" if n==383 else "e"}:l{surface}:{kind}:{index}',
                                printed_page=n,pdf_page=n+22,kind=kind,form=get(f),gloss=get(g).strip('‘’ '),
                                source_lines=ls,source_form_selection=f,source_gloss_selection=g,
                                evidence=[pages[n]['lines'][i-1] for i in ls],
                                note=NOTES.get((n,surface),''),
                                status='Owner original-page review passed; independent review pending'))
    assert len(out)==108
    return out

if __name__=='__main__':
    rows=build();(P/'appendix-copula-examples-reviewed-records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(len(rows),'records: 24 expressions and 84 aligned words')
