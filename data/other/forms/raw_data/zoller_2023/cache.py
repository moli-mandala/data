"""Read the checked-in source snapshot independently of machine-local temporary files."""
import gzip,json
from pathlib import Path
RAW=Path(__file__).parent
def load_records():
    with gzip.open(RAW/'records.json.gz','rt',encoding='utf8') as f:return json.load(f)
