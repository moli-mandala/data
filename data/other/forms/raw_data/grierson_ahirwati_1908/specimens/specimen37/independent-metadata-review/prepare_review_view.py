"""Compact read-only view of frozen No37 candidate; no input writes."""
import csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "independent-review/metadata-proposal-revision2/candidate.csv"
with SOURCE.open(newline="") as stream, (HERE / "all390-review-view.tsv").open("w") as out:
    out.write("ordinal\tkey\tform\tgloss\ttags\tnotes\n")
    for n, r in enumerate(csv.reader(stream), 1):
        key = r[10].replace("lsi9.2:", "").replace(":aligned-line:", ":l").replace(":word:", ":w")
        tags = " ".join(t for t in r[14].split() if not t.startswith("dialect:"))
        out.write("\t".join([str(n), key, r[2], r[3], tags, r[6]]) + "\n")
