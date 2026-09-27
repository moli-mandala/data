"""Compatibility entrypoint for the complete Sikalgari source importer."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from import_source_full import generate, main
if __name__=="__main__":main()
