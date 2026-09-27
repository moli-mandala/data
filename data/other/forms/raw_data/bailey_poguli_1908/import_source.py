"""Compatibility entry point for the complete Poguli chapter importer."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from import_source_full import *
if __name__ == "__main__": main()
