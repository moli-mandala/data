"""Apply a pinned independent contextual review with explicit owner decisions."""
import copy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REVIEW_SHA='5289e80990ecc673a67a7d06905d0c2a25e0f5a4ff7d89f4c5d5ce32a4ae5578'


def decisions():
    path=HERE/'independent-pending-metadata-review.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==REVIEW_SHA, 'Stale independent specimen metadata review'
    reviewed=json.loads(path.read_text())['decisions']
    assert len(reviewed)==114
    output={}
    for original in reviewed:
        row=copy.deepcopy(original)
        row['owner_decision']='Accept contextual interpretation, restricted to this exact source occurrence.'
        row['proposed_gloss']=row['gloss']
        row['support']=[]
        tags=row['proposed_tags']
        # Object English pronouns do not by themselves resolve source acc/dat.
        # In particular annē is a genitive in the grammar but means me here.
        if 'obj' in tags:
            tags.remove('obj')
        if row['form']=='annē' and row['gloss']=='me':
            row['uncertainty']=dict(type='source-analysis',reason=(
                'This occurrence is glossed me, whereas p570 pronoun table '
                'glosses annē my. Preserve occurrence translation; no genitive, '
                'accusative or dative assigned.'))
            row['owner_decision']='Accept personal-pronoun function; preserve conflict with grammar table and leave case unresolved.'
        # Unlike a generic English preposition, this printed ending is
        # explicitly named as locative by the source grammar on p570.
        if row['form'] in {'Aunnē-lōpul','Ādi-yauga-lōpul','aunē-lōpul','ittē-lōpul','Ōnē-lōpul'}:
            tags.append('loc')
            row['proposed_gloss']=row['gloss'].rsplit('-',1)[0]
            row['support']=['grierson1906naiki:p570:prose:locative']
            row['owner_decision']='Add source-explicit locative -lōpul analysis; preserve full word and do not infer person or number from stem resemblance.'
        if 'dat' in tags:
            gloss=row['gloss']
            row['proposed_gloss']=gloss[3:] if gloss.casefold().startswith('to-') else gloss.removesuffix('-to')
        if row['uncertainty'] and 'uncertain' not in tags:
            tags.append('uncertain')
        key=row['source_unit_key']
        assert key not in output
        output[key]=row
    return output
