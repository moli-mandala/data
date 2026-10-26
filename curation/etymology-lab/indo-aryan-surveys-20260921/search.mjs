// Run the application's actual homepage query code against its existing read-only database.
import {createRequire} from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const workspace=path.resolve(import.meta.dirname,'../../../..');
const app=path.join(workspace,'jambu-static');
const require=createRequire(path.join(app,'package.json'));
const {build}=require('esbuild');
const out='/tmp/jambu-surveys-20260921-search.cjs';
const adapter=`
import Database from '${app}/node_modules/better-sqlite3/lib/index.js';
import {makeVinIn,makeVdeltaIn} from '${app}/src/lib/dbShared.ts';
import {unicodeSearchFold} from '${app}/src/lib/unicodeSearch.ts';
const db=new Database('${app}/.dbwork/jambu.db',{readonly:true,fileMustExist:true});
const registry=new Map();
db.function('vin_in',{deterministic:true},makeVinIn(id=>registry.get(id)));
db.function('vdelta_in',{deterministic:true},makeVdeltaIn(id=>registry.get(id)));
db.function('unicode_fold',{deterministic:true},unicodeSearchFold);
export async function query(sql,params=[],sets=[]){
  for(const [id,members] of sets)registry.set(id,new Set(members));
  try{return db.prepare(sql).all(...params)}finally{for(const [id] of sets)registry.delete(id)}
}
export async function queryOne(sql,params=[],sets=[]){return (await query(sql,params,sets))[0]??null;}
`;
await build({entryPoints:[path.join(app,'src/lib/query.ts')],outfile:out,bundle:true,platform:'node',format:'cjs',packages:'external',nodePaths:[path.join(app,'node_modules')],plugins:[{name:'readonly-local-database',setup(b){
 b.onResolve({filter:/better-sqlite3\/lib\/index\.js$/},args=>({path:args.path,external:true}));
 b.onResolve({filter:/^marked$/},()=>({path:require.resolve('marked'),external:true}));
 b.onResolve({filter:/db\.svelte$/},()=>({path:'adapter',namespace:'survey'}));
 b.onLoad({filter:/.*/,namespace:'survey'},()=>({contents:adapter,loader:'ts',resolveDir:app}));
 b.onResolve({filter:/^\$app\/paths$/},()=>({path:'paths',namespace:'paths'}));
 b.onLoad({filter:/.*/,namespace:'paths'},()=>({contents:"export const base='';",loader:'js'}));
}}]});
const {fetchLemmaList}=require(out);
const inputs=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const results=[];
for(const input of inputs){
 const {mode='reflexes',...params}=input;
 const result=await fetchLemmaList({mode,params:{relaxed:true,...params},withOrigin:true});
 results.push({params,...result});
}
fs.writeFileSync(process.argv[3],JSON.stringify(results,null,1)+'\n');
console.log(results.map(r=>({params:r.params,count:r.count,rows:r.rows.length})));
