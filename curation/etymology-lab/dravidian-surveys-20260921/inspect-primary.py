import csv,gzip,json,sys,collections
p='data/curation/etymology-lab/dravidian-surveys-20260921/'
r=json.load(open(p+'survey-inventory.json'))
ids=set(sys.argv[1:]);articles={}
for x in csv.DictReader(gzip.open('data/data/dedr/entry-texts-audit.csv.gz','rt')):
 if x['Entry_ID'] in ids and x.get('Status') != 'duplicate-excluded':articles[x['Entry_ID']]=x
for i in sys.argv[1:]:
 x=articles.get(i);print('\n'+i+' '+(x['Raw_Article'] if x else 'MISSING'))
json.dump(articles,open(p+'inspected-'+('-'.join(sys.argv[1:]))+'.json','w'),ensure_ascii=False,indent=2)
