import json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[5];sys.path.insert(0,str(R));from concepts import map_glosses
samples=['nest','flea','get up','3s (generic/male)','to stop','to rise','cane','fat (meat)',"father's sister",'bolt','scatter','rear','within','3s (female)','left-hand','to receive','hair of the head']
print(json.dumps(map_glosses(samples),ensure_ascii=False,sort_keys=True,indent=2))
