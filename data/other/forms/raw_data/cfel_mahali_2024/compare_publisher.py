"""Cache bounded public searches and propose correspondences; never accept or install them."""
import argparse,datetime,hashlib,json,re,time,unicodedata
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
ROOT=Path(__file__).resolve().parent
ENDPOINT='https://cfelvb.in/api/dictionary/getDictionaryData.php'
def query_text(s):return ' '.join(re.sub(r'-\s*\n\s*','-',s).split())
def ipa(s):
    # Comparison-only typographic equivalences; no vowel, stop or uncertainty merger.
    s=re.sub(r'-[ \t]*\n[ \t]*','-',s)
    return unicodedata.normalize('NFC',' '.join(s.strip().strip('/').split())).replace('͡','').replace('ʤ','dʒ').replace('ʧ','tʃ')
def domain(s):return ''.join(c.lower() for c in s if c.isalnum())
def cache_path(cache,query):return cache/(hashlib.sha256(query.encode()).hexdigest()+'.json')
def read_response(request):
    """Retry transient public-read failures at most twice; never cache failures."""
    for attempt in range(3):
        try:
            with urlopen(request,timeout=25) as response:
                body=response.read(2_000_001)
            assert len(body)<=2_000_000,'unexpected response size'
            return body
        except HTTPError as error:
            if error.code not in (408,429,500,502,503,504) or attempt==2:raise
        except (URLError,TimeoutError):
            if attempt==2:raise
        time.sleep(2**attempt)
def fetch(cache,limit):
    rows=[json.loads(x) for x in (ROOT/'positioned-candidates.jsonl').read_text().splitlines()]
    queries=list(dict.fromkeys(query_text(r['candidate_english']) for r in rows));cache.mkdir(parents=True,exist_ok=True);done=0
    for query in queries:
        path=cache_path(cache,query)
        if path.exists():continue
        if done>=limit:break
        # Exact public UI search. Returned IDs must be reviewed, not guessed from labels.
        params={'limit':100,'offset':0,'search':query,'source':1,'target':303}
        url=ENDPOINT+'?'+urlencode(params)
        request=Request(url,data=b'',headers={'User-Agent':'Jambu source review'})
        body=read_response(request)
        payload=json.loads(body)
        if payload==0:payload={'count':0,'data':[]}
        assert isinstance(payload,dict) and isinstance(payload.get('data'),list)
        record={'query':query,'url':url,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'response_sha256':hashlib.sha256(body).hexdigest(),'response':payload}
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,ensure_ascii=False)+'\n');temp.replace(path)
        done+=1;print(f"cached {done}: {query} ({len(payload['data'])} results)",flush=True);time.sleep(.3)
    return done

def compare(cache):
    rows=[json.loads(x) for x in (ROOT/'positioned-candidates.jsonl').read_text().splitlines()]
    domains={r['entry_key']:r['domain'] for r in json.loads((ROOT/'domains.json').read_text())['assignments']}
    proposals=[]
    for row in rows:
        query=query_text(row['candidate_english']);path=cache_path(cache,query)
        if not path.exists():continue
        cached=json.loads(path.read_text());assert cached['query']==query
        response=cached['response'];candidates=[]
        for item in response['data']:
            if item.get('language')!=303:continue
            if ipa(item.get('ipa') or '')!=ipa(row['raw_ipa']):continue
            if domain(item.get('domain_name') or '')!=domain(domains[row['entry_key']]):continue
            candidates.append({k:item.get(k) for k in ['id','abs_ref','word','word2','word3','ipa','domain_name']})
        proposals.append({'entry_key':row['entry_key'],'query':query,'source_ipa':row['raw_ipa'],'source_native':row['raw_native'],'source_domain':domains[row['entry_key']],'api_candidates':candidates,'api_result_count':len(response['data']),'reported_api_count':response.get('count'),'cache_file':path.name,'response_sha256':cached['response_sha256'],'status':'correspondence proposal; visual/native/edition review required'})
    return proposals
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--fetch',type=int,default=0);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert 0<=a.fetch<=50,'bounded batches only'
    if a.fetch:fetch(a.cache,a.fetch)
    proposals=compare(a.cache);a.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in proposals));print(f'{len(proposals)} proposals; none accepted automatically')
