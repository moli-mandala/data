"""All98 exact-source grammar decisions, proposed only; no canonical emission."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'tags.py').exists())
sys.path.insert(0,str(DATA))
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
SOURCE=PACKAGE/'grammar/child-witnesses-reviewed.jsonl'
OUT=HERE/'grammar-metadata-proposal-r2'
ANALYSIS_TAGS={
 'first person singular copula':'verb copula pres 1sg sg',
 'strong masculine noun nominative singular ending':'affix suffix strong m nom sg',
 'strong masculine noun oblique singular ending':'affix suffix strong m obl sg',
 'locative agreement ending':'affix suffix loc',
 'contrastive non-locative agreement ending':'affix suffix',
 'whole example':'multiword-expression',
 'possessive locative agreement':'pron poss loc',
 'contrasted possessive form':'pron poss',
 'locative noun':'noun loc',
 'locative singular ending':'affix suffix loc sg',
 'masculine locative singular':'noun m loc sg',
 'masculine consonant noun locative ending':'affix suffix m loc',
 'masculine locative noun':'noun m loc',
 'dative and agent case suffix':'affix suffix dat erg',
 'genitive suffix':'affix suffix gen',
 'whole sentence; future passive participle construction':'sentential',
 'pronoun in agent case':'pron personal erg',
 'future passive participle; same form as infinitive':'verb participle fut pass',
 'past copula':'verb copula pret',
 'genitive locative used for dative':'pron poss loc dat',
 'neuter':'n',
 'first person singular agent pronoun':'pron personal 1sg sg erg',
 'agent case marker':'affix suffix erg',
 'possessive pronoun':'pron poss',
 'possessive/reflexive meaning':'pron poss refl',
 'masculine oblique':'pron poss refl m obl',
 'demonstrative':'pron demonstrative',
 'feminine demonstrative':'pron demonstrative f',
 'demonstrative oblique singular':'pron demonstrative obl sg',
 'demonstrative oblique plural':'pron demonstrative obl pl',
 'relative/demonstrative contextual use':'pron relative',
 'relative pronoun':'pron relative',
 'interrogative pronoun':'pron interr',
 'past feminine copula':'verb copula pret f',
 'past masculine plural copula':'verb copula pret m pl',
 'parenthetical regional past copula alternative':'verb copula pret',
 'second person plural present copula; source-attributed borrowing':'verb copula pres 2pl pl loanword',
 'added copular syllable':'affix suffix pres',
 'compared added copular syllable':'affix suffix pres',
 'present copular expression':'verb copula pres',
 'compared copular expression':'verb copula pres',
}
ROUTES={'Ahirwati':'mewati_jhambaus','Mewati':'mewati_akera','Bangaru':'kaithal','Khandeshi':'Khandesi','Panjabi':'P'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def records(p):return [json.loads(s) for s in p.read_text().splitlines()]
def key(group,word,page=51):return f'lsi9.2:p{page}:{group}:w{word}'
VARIANT_GROUPS=[
 ('locative-noun-alternatives',[1,2]),('locative-noun-alternatives',[3,4]),
 ('dative-agent-suffix-alternatives',[1,2]),('personal-pronoun-me-alternatives',[1,2]),
 ('own-pronoun-alternatives',[1,2]),('demonstrative-this-nominative',[1,2]),
 ('demonstrative-this-oblique',[1,2]),('demonstrative-that-nominative',[1,2]),
 ('demonstrative-that-oblique',[1,2,3]),('present-copula:p2:sg',[1,2]),('present-copula:p2:pl',[1,2])]
def main():
    assert sha(SOURCE)=='5ac645b13ae1a3132ac094431fbce9376037705f93c0c59d9861e6e7ef999ee7'
    rr=records(SOURCE);assert len(rr)==len({r['witness_key'] for r in rr})==98
    by={r['witness_key']:r for r in rr};graphs=[]
    for group,words in VARIANT_GROUPS:
        parent=key(group,words[0]);assert parent in by
        for w in words[1:]:
            child=key(group,w);a=by[child];b=by[parent]
            assert a['disposition']==b['disposition']=='eligible' and a['language_label']==b['language_label']
            graphs.append({'child_key':child,'parent_keys':[parent],'relation':'variant',
              'exact_child_form_guard':a['source_form'],'exact_parent_form_guards':[b['source_form']],
              'source_quote':a['source_quote'],'qualification':'Explicit source-listed alternatives under the same functional slot; not an assertion of unrestricted free interchangeability.',
              'accepted':False})
    assert len(graphs)==12
    compound=key('Panjabi-comparison-whole-copulas',3)
    parents=[key('present-copula:p3:sg',1),key('Panjabi-comparison-whole-copulas',1)]
    graphs.append({'child_key':compound,'parent_keys':parents,'relation':'component',
       'exact_child_form_guard':by[compound]['source_form'],
       'exact_parent_form_guards':[by[k]['source_form'] for k in parents],
       'source_quote':by[compound]['source_quote'],
       'qualification':'Source explicitly says gō is added to the present copula. The separately printed third-singular sai matches the he-is example; endpoint choice and component relation require root review.',
       'accepted':False})
    variants={g['child_key'] for g in graphs if g['relation']=='variant'}
    result=[]
    for r in rr:
        eligible=r['disposition'] in ['eligible','eligible_comparator'];analysis=r['source_analysis'];tags=[];notes=[];etymology=''
        if eligible:
            if analysis.startswith('present copula; person '):
                _,person,number=analysis.split('; ');person=person.split()[-1]
                tags=['verb','copula','pres',f'{person}{number}',number]
            else:
                assert analysis in ANALYSIS_TAGS,(r['witness_key'],analysis)
                tags=ANALYSIS_TAGS[analysis].split()
            if r['typed_uncertainties']:tags.append('uncertain')
            if r['witness_key'] in variants:tags.append('alternate')
            if analysis=='future passive participle; same form as infinitive':
                notes.append('Grierson says this future passive participle has the same form as the infinitive.')
            if 'Bawal-Kot-Kasam' in r['witness_key']:
                notes.append('Grierson attributes these past-copula forms to the country around Bawal in Nabha and Kot Kasam in Jaipur; gender pairing of the parenthetical forms is not separately stated.')
            if r['witness_key']==key('relative-when-example',1):
                notes.append('Grierson says the relative is also used in a demonstrative sense, glossing this occurrence then as well as when.')
            if analysis=='demonstrative oblique plural':
                notes.append('Grierson labels this form oblique plural and says it is also often used with singular meaning.')
            if analysis=='second person plural present copula; source-attributed borrowing':
                etymology='Grierson explicitly attributes this second-plural present copula to borrowing from Bagri.'
        assert set(tags)<=GRAMMATICAL_TAGS|GENDER_TAGS
        result.append({'source_unit_key':r['witness_key'],'physical_parent_key':r['parent_key'],
          'exact_form_guard':r['source_form'],'exact_gloss_guard':r['source_gloss'],
          'exact_language_label_guard':r['language_label'],'exact_disposition_guard':r['disposition'],
          'proposed_language_id':ROUTES.get(r['language_label']) if eligible else None,
          'proposed_tags':list(dict.fromkeys(tags)),'proposed_notes':notes,'proposed_etymology':etymology,
          'source_grammatical_evidence':{'printed_page':r['printed_page'],'source_analysis':analysis,'source_quote':r['source_quote'],
            'raw_qualification':r['qualification'],'typed_uncertainties':r['typed_uncertainties']},
          'person_number_policy':'Explicit grammatical analysis/paradigm labels only; no number from English was/your, no inferred neuter/agent/POS beyond the source-defined scope.',
          'metadata_accepted':False,'disposition':r['disposition']})
    assert sum(r['proposed_language_id'] is not None for r in result)==75
    OUT.mkdir(exist_ok=True)
    for name,value in [('all98-keyed-grammar-metadata-proposal.jsonl',result),('source-graph-proposal.jsonl',graphs)]:
        p=OUT/name;data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in value)
        if p.exists():assert p.read_text()==data,('Immutable proposal differs',name)
        else:p.write_text(data)
    report={'status':'Complete keyed source-local proposal; no acceptance or canonical changes',
      'inputs':{'grammar/child-witnesses-reviewed.jsonl':sha(SOURCE)},
      'outputs':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='status.json'},
      'counts':{'interpretations':98,'eligible':75,'documentary':19,'unattested_hypotheses':4,'proposed_variants':12,'proposed_component_sets':1},
      'metadata_gap':'Four Bawal/Kot Kasam regional copulas require qualified regional provenance or an approved source-lect proposal; they must not inherit the Gurgaon lect silently.',
      'held_graph_classes':['Regional sō/sī/sā parentheses may be a gender paradigm, not unrestricted variants.','Punjabi hai-gā lacks a separately printed eligible hai endpoint.','The four unencountered pronoun obliques are documentary, never graph endpoints.','Source Bagri borrowing claim lacks certified exact donor endpoint.'],
      'canonical_changes':False}
    p=OUT/'status.json';data=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if p.exists():assert p.read_text()==data
    else:p.write_text(data)
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
