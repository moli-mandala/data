import {createRequire} from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root=path.resolve('jambu-static');
const require=createRequire(root+'/package.json');
const {build}=require('esbuild');const Database=require('better-sqlite3');
const {readVarints}=await import(pathToFileURL(root+'/src/lib/dbShared.ts'));
const {unicodeSearchFold}=await import(pathToFileURL(root+'/src/lib/unicodeSearch.ts'));
const db=new Database(root+'/.dbwork/jambu.db',{readonly:true});
db.function('unicode_fold',v=>unicodeSearchFold(String(v??'')));
let sets=new Map();db.function('vin_in',(b,i)=>readVarints(b).some(v=>sets.get(Number(i))?.has(v))?1:0);
globalThis.__researchQuery=async(sql,params=[],supplied=[])=>{sets=new Map(supplied.map(([i,m])=>[i,new Set(m)]));return db.prepare(sql).all(...params)};
const outfile=path.resolve('tmp/dravidian-query.mjs');
await build({entryPoints:[root+'/src/lib/query.ts'],outfile,bundle:true,platform:'node',format:'esm',plugins:[{name:'research-db',setup(b){
b.onResolve({filter:/db\.svelte$/},()=>({path:'db',namespace:'stub'}));b.onResolve({filter:/^\$app\/paths$/},()=>({path:'paths',namespace:'stub'}));b.onLoad({filter:/.*/,namespace:'stub'},a=>({contents:a.path==='db'?'export const query=(...a)=>globalThis.__researchQuery(...a);export const queryOne=async(...a)=>(await query(...a))[0]??null;':'export const base="";',loader:'js'}));}}]});
const {fetchLemmaList}=await import(pathToFileURL(outfile));
const requests=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));const out=[];
for(const opts of requests){const result=await fetchLemmaList(opts);for(let page=2;page<=Math.min(20,Math.ceil(result.count/50));page++){const more=await fetchLemmaList({...opts,params:{...opts.params,page}});result.rows.push(...more.rows)}out.push({opts,...result,rows:result.rows.map(({id,word,gloss,language_id,origin_lemma_id,etymon_id,references})=>({id,word,gloss,language_id,origin_lemma_id,etymon_id,references:references?.map(({id,locator})=>({id,locator}))}))});}
fs.writeFileSync(process.argv[3],JSON.stringify(out,null,2));console.log(JSON.stringify({queries:out.length,rows:out.reduce((n,x)=>n+x.rows.length,0)}));db.close();
