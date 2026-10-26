"""Owner first reading of all 99 final-wordlist forms; independent review pending."""
import hashlib
import json
import unicodedata
from pathlib import Path

HERE=Path(__file__).resolve().parent
READINGS={
 (108,'left'): ['মুই','তুই','অমি','ইয়ান','সিয়ান','কেন্না','কি','নয়','বেগে','বেয়াল','এগা','দিবা','ভনা','লাম্বা','চিগোন','মিলা','মরদ','জন','মাচ','পেখ পাখি','কুত্তর','উগুন','গাচ'],
 (108,'right'): ['বীজি','পাদা','শিঙোর','বাগল','চাম','হেরা','লো','হাড়','দাম','বড়া','শিং','লেচ','ফোর','চুল','শিরা','কানকান','চোক','নাক','মুহ্','দাত','জিল','ডাবি','আধু','হাথ'],
 (109,'left'): ['পেত','গরদনা','দুথ','চিং','অজভ’','পিয়ানা','খানা','কামরানা','দেঘানা','শুননা','মেহুত','ঘুমযানা','মরানা','কাবানা','সাজুরানা','উড়না','হাধানা','এখানা','মিঝ্যা','বঝানা','থিয়ানা','দেনা','কহ্না','বেল','চান','তারা'],
 (109,'right'): ['পানি','বর','শিল','করোলি','পিথিমী','মেঘ','খানা','আগুন','ছেই','পুড়িখানা','পধ','মোন','রাঙা','রেল','হোলত্যা','ধুপ','কালা','রেত','গরম','জুরো','পুনো','নুয়ো','গম','পেংগুল','শুঘুনো','নাঙ'],
}
# These are explicit first-reading uncertainties, not reasons to drop cells.
FLAGS={
 (108,'left',11):'Review the medial consonant/gemination in the printed one form.',
 (108,'left',12):'Review initial consonant and vowel attachment; OCR suggests দ্বিবা.',
 (108,'left',21):'Review medial consonant cluster; do not substitute Bengali কুকুর.',
 (108,'right',3):'Review the medial nasal/consonant cluster against larger original.',
 (108,'right',6):'Review vowel placement in the flesh form; OCR gives হ্রো.',
 (108,'right',24):'Review final aspirated consonant versus plain dental.',
 (109,'left',4):'Review final nasal sign versus visarga-like typography.',
 (109,'left',5):'Review final consonant and apostrophe shape; keep mark in raw first reading.',
 (109,'left',11):'Review final consonant/diacritic; no semantic correction from English Know.',
 (109,'left',14):'Review medial ব/ক consonant against original.',
 (109,'left',15):'Review nasal/conjunct in swim form.',
 (109,'left',23):'Review medial conjunct in say form.',
 (109,'right',2):'Review initial ব/ঝ difference; whole-page and column OCR disagree.',
 (109,'right',14):'Review initial consonant and vowel sequence in green form.',
 (109,'right',21):'Review medial conjunct in full form.',
 (109,'right',26):'Review final nasal sign in name form.',
}


def main():
    inventory=HERE/'basic-cell-inventory-proposal.jsonl'
    units=[json.loads(line) for line in inventory.read_text().splitlines()]
    assert len(units)==99
    rows=[]
    for unit in units:
        key=(unit['pdf_page'],unit['block'],unit['row'])
        forms=READINGS[key[:2]]
        form=unicodedata.normalize('NFC',forms[key[2]-1])
        row=dict(source_unit_key=unit['source_unit_key'],pdf_page=unit['pdf_page'],
            printed_page=unit['printed_page'],block=unit['block'],row=unit['row'],
            raw_owner_first_reading=form,printed_english=unit['english_original'],
            original_ocr=unit['source_fields']['chakma'],
            review_status='Owner first reading only; independent original review pending',
            source_caveats=unit['source_caveats'],accepted_for_installation=False,
            native_script='Bengali',phonemic_asserted=False)
        if key in FLAGS:
            row['typed_uncertainty']=dict(type='transcription',reason=FLAGS[key])
        rows.append(row)
    for (page,block),forms in READINGS.items():
        assert len(forms)==sum(u['pdf_page']==page and u['block']==block for u in units)
    target=HERE/'basic-first-reading.jsonl'
    assert not target.exists(),'Preserve immutable first readings; reconcile corrections separately.'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report=dict(stage='All99 target forms have owner first readings; no lexical acceptance',
        rows=99,typed_transcription_flags=len(FLAGS),source_gloss_caveats=sum(bool(r['source_caveats']) for r in rows),
        inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
        first_reading_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        evidence='Pinned original PDF108–109; displayed full pages, no reconstructed IPA or translated glosses.',
        next='Independent full original rereading, Bengali control transcription and whole-book integration',
        canonical_rows_changed=0)
    (HERE/'basic-first-reading-status.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
