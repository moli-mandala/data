"""Apply only the two root-authorized language records, with idempotent guards."""
import csv,hashlib,io,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;DATA=ROOT.parents[4];WORKSPACE=DATA.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 proposal_path=ROOT/'language-registry-proposal.json';p=json.loads(proposal_path.read_text());path=DATA/'cldf/languages.csv';before=path.read_bytes();existing=list(csv.DictReader(io.StringIO(before.decode())));fields=list(existing[0]);add=[]
 assert len(p['records'])==2
 for r in p['records']:
  assert list(r)==fields
  matched=[x for x in existing if x['ID']==r['ID'] or x['Glottocode']==r['Glottocode']]
  if matched:assert matched==[r],('Conflicting identity',r['ID'])
  else:add.append(r)
  assert r['Latitude']==r['Longitude']==r['Location']==r['Quality']==''
  assert r['Clade'] in (WORKSPACE/'jambu-static/src/lib/clades.ts').read_text()
 if add:
  assert len(add)==2,'Partial previous addition requires explicit reconciliation'
  assert sha(path)==p['inputs']['cldf/languages.csv'],'Registry changed since approval; inspect before append'
  buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=fields,lineterminator='\r\n' if b'\r\n' in before else '\n');writer.writerows(add)
  assert before.endswith(b'\n');path.write_bytes(before+buf.getvalue().encode())
 after=list(csv.DictReader(path.open()));assert after[:len(existing)]==existing
 assert all(r in after for r in p['records'])
 report={'status':'two exact root-authorized language records installed; source lexical CSV unchanged','approval':'Parent /root explicitly approved Tanchangya/Chenchu metadata additions on2026-09-27 after primary authority verification','proposal_sha256':sha(proposal_path),'registry_before_sha256':hashlib.sha256(before).hexdigest(),'registry_after_sha256':sha(path),'added_ids':[r['ID'] for r in add],'all_previous_records_unchanged':True,'frontend_supported_clades':[r['Clade'] for r in p['records']],'blank_source_locations_preserved':True,'routes':p['routes'],'source_rows_installed':0}
 out=ROOT/'language-registry-installation.json'
 if not out.exists() or add:out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'added_ids':report['added_ids'],'registry_sha256':sha(path)}))
if __name__=='__main__':main()
