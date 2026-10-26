"""Manual original-first reading; no canonical exports or grammatical inference."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'data/source_meta.py').exists())
PAGE=564
LINES = '''
Ōkkōn=One|mās=man|āṇḍen.=was.|Amnuṅg=Him-to|iddar=two|bālākōl=sons|āṇḍēr.=were.|Āttān=Them-of
tsinnām=the-younger|enten,=said,|‘bā,=‘father.|amnet=my|vāṭū=share|anu=me|kōr.’=give.’|Mārī=Then|amd=he|avaruṅg=them-to
dhan=property|pāyāknā=dividing|siten.=gave.|Mārī=Then|kōnnūṅg=some|divasānī=in-days|tsinnām=he-younger|bālā=son
attānā=all|ghumkāknā=collecting|laya=very|dhāv=far|sedeṇ.=went.|Attīn=There|amd=he|mājiltālā=riotously|tān=his
jiṅgī=property|uḍāpten.=squandered.|Mārī=Then|amd=he|attānā=all|kharchipten,=spent,|add=that|muluk=country|mahāg=famine
paṭṭīn.=fell.|Ad-muḷ=Therefore|amnuṅg=him-to|aṇchan=difficulty|paṭṭīn.=became.|Addī=That|vakhōt=time|amd=he|ad=that
muḷuk=country(-of)|ōkōn=one|māsan-āttin=man-with|actnā=going|tākten.=lived.|Amd=He|amnuṅg=him|turel=swine
kāyeṅg=to-feed|tānet=his|vegāḍuṅg=field-to|pānākten.=sent.|Addī=That|vakhōt=time|turel=swine|āṅse=husks|tinhā=ate
ādnāḍ=those-from|amd=he|pōṭā=belly|niḍinā=fill|anāṅg=thus|amnun=him-to|vāṭlīn.=it-appeared.|Amnuṅg=Him-to|yēnā=any-one
siyeten.=gave-not.|Mārī=Then|amd=he|avaletnā=coming-to-senses|enten,=said,|‘amnet=‘my|ambān=father’s|māsuruṅg=servants-to
pheret=much|īpāṭe=bread|anḍā,=is,|ān=I|kārut=with-hunger|tikhātūn.=die.|Ān=I|sūltnā=arising|am-bānāṅg=my-father-near
serātūn=will-go|amnuṅg=him-to|yenātūn,=will-say,|“bā,=“father,|ān=I|diyamnet=God-of|innet=thy|muṭt=in-presence|pāp=sin
kāktān.=did.|In-din-tānāt=This-day-from|innet=thy|bālā=son|enet=to-say|lāṅg=worthy|tōten.=am-not.|Innet=Thy|ōkōn=one
tsākarī-parmān=servant-like|īḍ.”’=keep.”’|Mārī=Then|amd=he|sūltnā=arising|tān=his|bānāṅg=father-to|sedden.=went.|Mārī=Then
amd=he|dhāv=far|anḍān=is|ittet=then|amnet=his|tāk=father-|ōltnā,=seeing,|lōbha=pity|vāttīn,=came,|amd=he
laīltnā=running|amne=his|mākkun=neck-on|ārāgā-paṭṭīn=fell|amnet=his|mukā=kiss|samnet.=took.|Mārī=Then|bālā=son|amd=that
enten,=said,|‘bā=‘father|diyamnet=God-of|innet=thy|muṭt=before|ān=I|pāp=sin|kāktān.=did.|Indintānāt=Henceforward|innet=thy
'''.strip().splitlines()


def main():
    image=ROOT/'tmp/jambu-next-source/kolami-specimen-originals/printed-564-2800.png'
    source_hash=hashlib.sha256(image.read_bytes()).hexdigest()
    census={r['source_unit_key']:r for r in map(json.loads,(HERE.parent/'scope/physical-unit-census.jsonl').read_text().splitlines())}
    rows=[]
    for line,text in enumerate(LINES,1):
        parent=f'grierson1906kolamibasim:p564:specimen75-line:{line}'
        assert parent in census
        atoms=[]
        for word,pair in enumerate(text.split('|'),1):
            form,gloss=pair.split('=',1)
            atoms.append(dict(source_unit_key=f'{parent}:word:{word}',printed_page=PAGE,physical_page_onebased=PAGE+20,
                physical_line=line,physical_word=word,raw_source_form=form,raw_interlinear_gloss=gloss,
                source_scope='Specimen75 / District Wun / Kolami',review='manual first reading; independent review pending'))
        rows.append(dict(source_unit_key=parent,printed_page=PAGE,physical_page_onebased=PAGE+20,
            specimen=75,source_scope='District Wun / Kolami',source_image=str(image.relative_to(ROOT)),source_image_sha256=source_hash,
            target_atoms=atoms,raw_target_line=' '.join(a['raw_source_form'] for a in atoms),
            raw_aligned_english_line=' '.join(a['raw_interlinear_gloss'] for a in atoms),
            source_punctuation_note='A separated dash follows dhan on physical line3; retained as layout punctuation, not a lexical head.' if line==3 else '',
            typed_uncertainties=[dict(type='transcription',reason='Fine marks need independent original-controlled review; no analogy-based harmonization.')],
            review='complete page manual original-first reading; not accepted or exported'))
    assert len(rows)==17
    output=HERE/'p564-first-reading.jsonl'
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if output.exists():assert output.read_text()==text,'Frozen first reading must not be overwritten'
    else:output.write_text(text)
    report=dict(page=PAGE,physical_lines=len(rows),physical_atoms=sum(len(r['target_atoms']) for r in rows),
        output_sha256=hashlib.sha256(text.encode()).hexdigest(),status='First-reading evidence, no canonical proposal',
        final_expression_spans='Pending source-boundary review; line17 continues on565',canonical_changes=False)
    (HERE/'p564-first-reading-status.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
