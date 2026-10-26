"""Apply position-keyed reviewed changes without rewriting the first reading."""
import json
import unicodedata
from pathlib import Path
from record_specimens_first_reading import PAGES

HERE = Path(__file__).resolve().parent

def main():
    review = json.loads((HERE / 'grayscale-corrections.json').read_text())
    changes = {(p,l,w):(old,new) for p,l,w,old,new in review['changes']}
    glosses = {(p,l,w):(old,new) for p,l,w,old,new in review['gloss_changes']}
    assert len(changes) == len(review['changes'])
    seen = set()
    output = []
    for page, lines in PAGES.items():
        for line, pairs in enumerate(lines, 1):
            for word, (form, gloss) in enumerate(pairs, 1):
                key = page,line,word
                original = form,gloss
                if key in changes:
                    old,form = changes[key]
                    assert original[0] == old, (key,original,old)
                    seen.add(key)
                if key in glosses:
                    old,gloss = glosses[key]
                    assert original[1] == old, (key,original,old)
                output.append(dict(page=page,line=line,word=word,
                    first_form=original[0],first_gloss=original[1],
                    form=unicodedata.normalize('NFC',form),gloss=gloss,
                    grayscale_page_reviewed=page in review['reviewed_pages'],
                    independent_review_pending=True))
    assert seen == set(changes)
    assert len(output) == 537
    (HERE/'specimens-second-reading.jsonl').write_text(''.join(
        json.dumps(row,ensure_ascii=False)+'\n' for row in output))
    print(json.dumps(dict(units=len(output),form_corrections=len(changes),
        gloss_corrections=len(glosses),reviewed_pages=review['reviewed_pages'])))

if __name__ == '__main__':
    main()
