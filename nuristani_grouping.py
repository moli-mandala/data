"""Group Nuristani evidence under mapped CDIAL entries without choosing a transmission route.

The historical cognate/borrowing catalogs remain evidence for the CDIAL correspondence, not
instructions to construct PII → PNur trees. This final graph pass follows durable-ID assignment
and editorial overlays, so neither an importer nor an old assignment can restore those trees.

For compatibility with Jambu's attestation-tree schema, a grouping uses a rank-1 ``reflex`` edge,
explicitly qualified by ``grouping:cdial`` in Note and ``etymology-group`` in Tags. It asserts
neither inheritance from Indo-Aryan nor borrowing. PNur reconstructions and ordinary attestations
are siblings. Attested variants retain their true lexical target within the group. The obsolete
blank PII grouping nodes become redirect stubs; genuine reconstructions retain their content.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUP_TAG = "etymology-group"
GROUP_NOTE = "grouping:cdial; inheritance versus borrowing unresolved"
ATTESTATION_KINDS = {"reflex", "borrowed", "variant"}


def is_cdial(row):
    # Historical subsection aliases can resolve to durable f_ IDs (e.g. 8125-2).
    return row.get("Language_ID") == "Indo-Aryan" and (
        row["ID"][:1].isdigit()
        or any(s.strip().split("[", 1)[0] == "CDIAL" for s in row.get("Source", "").split(";"))
    )


def read_catalog(root=ROOT):
    """Return source-local head→CDIAL correspondences and retired blank grouping nodes."""
    groups, redirects = {}, {}

    def put(mapping, child, parent):
        if child in mapping and mapping[child] != parent:
            raise ValueError(f"Conflicting Nuristani CDIAL groups for {child}")
        mapping[child] = parent

    with (root / "data/nuristani_cognates.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            put(groups, r["Proto_Nuristani_ID"], r["Indo_Aryan_ID"])
            put(redirects, r["Ancestor_ID"], r["Indo_Aryan_ID"])
    with (root / "data/nuristani_borrowings.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            put(groups, r["Proto_Nuristani_ID"], r["Indo_Aryan_ID"])
    with (root / "data/nuristani_cdial_groups.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            put(groups, r["Group_ID"], r["CDIAL_ID"])
    return groups, redirects


def group_graph(forms, edges, groups, redirects, aliases, nuristani_languages):
    """Mutate forms/edges in place; return an audit of changed graph/content fields.

    forms may be scoped to Nuristani nodes, mapped heads, and IA entries. All edges are supplied
    so ancestor resolution, redirect retirement, and preservation of other families are checked.
    Catalog IDs may be source-local or durable. No lexical spellings, citations, or IDs change.
    """
    by_id = {r["ID"]: r for r in forms}

    def resolve(i):
        seen = set()
        while i not in by_id and i in aliases:
            if i in seen:
                raise ValueError(f"Alias cycle at {i}")
            seen.add(i)
            i = aliases[i]
        # A native CDIAL addendum can itself be a live redirect.
        seen = set()
        while by_id.get(i, {}).get("Redirect"):
            if i in seen:
                raise ValueError(f"Redirect cycle at {i}")
            seen.add(i)
            i = by_id[i]["Redirect"]
        return i

    mapped = {}
    for child, parent in groups.items():
        child = child if child in by_id else aliases.get(child, child)
        parent = resolve(parent)
        if child not in by_id or parent not in by_id:
            raise ValueError(f"Missing Nuristani CDIAL mapping: {child} → {parent}")
        if not is_cdial(by_id[parent]):
            raise ValueError(f"Nuristani group target is not a CDIAL entry: {parent}")
        if child in mapped and mapped[child] != parent:
            raise ValueError(f"Conflicting resolved Nuristani groups for {child}")
        mapped[child] = parent
    retired = {}
    for child, parent in redirects.items():
        child = child if child in by_id else aliases.get(child, child)
        parent = resolve(parent)
        if child not in by_id:
            continue  # no placeholder was ever emitted for this source-local ID
        r = by_id[child]
        if r["Language_ID"] != "Indo-ir" or r.get("Form") or r.get("Original"):
            raise ValueError(f"Refusing to retire a nonblank PII reconstruction: {child}")
        if parent not in by_id or by_id[parent]["Language_ID"] != "Indo-Aryan":
            raise ValueError(f"Missing CDIAL redirect target: {parent}")
        retired[child] = parent

    rank1 = {e["Child_ID"]: e for e in edges
             if str(e["Rank"]) == "1" and e["Kind"] in ATTESTATION_KINDS}
    # Snapshot before mutation: input source order cannot affect routing of descendants.
    old_parent = {i: e["Parent_ID"] for i, e in rank1.items()}
    targets = dict(mapped)
    for r in forms:
        i = r["ID"]
        if i in mapped or i in retired or is_cdial(r) or r.get("Redirect"):
            continue
        cur, seen = i, set()
        while cur and cur not in seen:
            seen.add(cur)
            if cur in mapped:
                targets[i] = mapped[cur]
                break
            if cur in retired:
                targets[i] = retired[cur]
                break
            p = old_parent.get(cur)
            if p in by_id and is_cdial(by_id[p]):
                if r["Language_ID"] in nuristani_languages:
                    targets[i] = resolve(p)
                break
            cur = p
        else:
            if cur:
                raise ValueError(f"Ancestry cycle while grouping {i}")

    audit = []
    for i, target in targets.items():
        r = by_id[i]
        e = rank1.get(i)
        # Preserve genuine variant-of-lemma relations, but never a PNur intermediate parent.
        lexical_variant = (e is not None and e["Kind"] == "variant"
                           and e["Parent_ID"] in targets
                           and targets[e["Parent_ID"]] == target
                           and by_id[e["Parent_ID"]]["Language_ID"] != "PNur"
                           and e["Parent_ID"] not in mapped)
        parent, kind = (e["Parent_ID"], "variant") if lexical_variant else (target, "reflex")
        previous = dict(e) if e else {}
        if e is None:
            e = dict(Child_ID=i, Parent_ID=parent, Kind=kind, Rank="1", Pos="", Source="", Note="")
            edges.append(e)
            rank1[i] = e
        e.update(Parent_ID=parent, Kind=kind, Rank="1", Pos="")
        notes = [v for v in e.get("Note", "").split("; ")
                 if v and v not in GROUP_NOTE.split("; ")]
        if not lexical_variant:
            notes.extend(GROUP_NOTE.split("; "))
        e["Note"] = "; ".join(dict.fromkeys(notes))
        tags = r.get("Tags", "").split()
        old_tags = " ".join(tags)
        tags = [t for t in tags if t != GROUP_TAG]
        if not lexical_variant:
            tags.append(GROUP_TAG)
        r["Tags"] = " ".join(tags)
        old_status = r.get("Status", "")
        r["Status"] = ""
        if previous != e or old_tags != r["Tags"] or old_status:
            audit.append(dict(Form_ID=i, Old_Parent=previous.get("Parent_ID", ""),
                              New_Parent=parent, Old_Kind=previous.get("Kind", ""),
                              New_Kind=kind, Action="group"))

    for i, parent in retired.items():
        r = by_id[i]
        if r.get("Redirect") != parent or i in rank1:
            audit.append(dict(Form_ID=i, Old_Parent=old_parent.get(i, ""), New_Parent=parent,
                              Old_Kind=rank1.get(i, {}).get("Kind", ""), New_Kind="", Action="redirect"))
        r.update(Redirect=parent, Status="entry")

    kept, seen = [], set()
    for e in edges:
        child, parent = e["Child_ID"], e["Parent_ID"]
        if child in retired:
            continue
        if parent in retired:
            if child == retired[parent]:
                # Remove the IA → former placeholder edge, rather than manufacture a self-edge.
                if e is rank1.get(child):
                    by_id[child]["Status"] = "entry"
                    audit.append(dict(Form_ID=child, Old_Parent=parent, New_Parent="",
                                      Old_Kind=e["Kind"], New_Kind="", Action="detach-placeholder"))
                continue
            e["Parent_ID"] = retired[parent]
        # A secondary reference to a flattened PNur/PII head should address its CDIAL group.
        if str(e["Rank"]) != "1" and e["Parent_ID"] in mapped and child in targets:
            e["Parent_ID"] = mapped[e["Parent_ID"]]
        primary = rank1.get(child)
        if str(e["Rank"]) != "1" and primary and e["Parent_ID"] == primary["Parent_ID"]:
            continue
        key = (child, e["Parent_ID"], e["Kind"], str(e["Rank"]), e.get("Pos", ""))
        if child == e["Parent_ID"]:
            raise ValueError(f"Grouping would create a self-edge: {child}")
        if key not in seen:
            kept.append(e)
            seen.add(key)
    edges[:] = sorted(kept, key=lambda e: (
        e["Child_ID"], e["Kind"], int(e["Rank"] or 1), int(e.get("Pos") or 0), e["Parent_ID"]))
    return audit


def apply_to_build(forms, edges_path, aliases, root=ROOT):
    """Final pipeline hook, reusing forms already loaded by assign_form_ids."""
    from edges_build import EDGES_HEADER, validate_edge_dicts
    groups, redirects = read_catalog(root)
    with (root / "cldf/languages.csv").open(encoding="utf-8") as f:
        langs = {r["ID"] for r in csv.DictReader(f) if r["Clade"] == "Nuristani"}
    with edges_path.open(encoding="utf-8") as f:
        edges = list(csv.DictReader(f))
    audit = group_graph(forms, edges, groups, redirects, aliases, langs)
    validate_edge_dicts(edges, {r["ID"]: r.get("Status", "") for r in forms})
    with edges_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EDGES_HEADER, lineterminator="\n")
        w.writeheader()
        w.writerows(edges)
    return audit


def main():
    """Apply the policy to an existing build, streaming the corpus and realigning changed nodes."""
    import argparse
    import json
    import os
    import shutil
    from edges_build import EDGES_HEADER, validate_edge_dicts
    from edges_util import aligned_parent, rank1_map
    import align as phonetic

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'tmp/nuristani-cdial-grouping')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    groups, redirects = read_catalog()
    cldf = ROOT/'cldf'
    if args.out.resolve() == cldf.resolve():
        raise ValueError('--out must be a staging directory, not cldf itself')
    source_stat={name:(cldf/name).stat().st_mtime_ns for name in ('forms.csv','edges.csv','alignments.csv')}
    with (cldf/'form-id-aliases.csv').open() as f:
        aliases = {r['Legacy_ID']:r['Form_ID'] for r in csv.DictReader(f)}
    wanted = {aliases.get(i,i) for i in groups.keys() | redirects.keys()}
    with (cldf/'languages.csv').open() as f:
        langs = {r['ID'] for r in csv.DictReader(f) if r['Clade']=='Nuristani'}
    with (cldf/'edges.csv').open() as f:edges=list(csv.DictReader(f))
    # Source articles can also attach a comparative Dameli (etc.) form to a PNur head.
    # Those descendants must move alongside the reconstructed head too.
    children = {}
    for e in edges:
        if e['Rank']=='1' and e['Kind'] in ATTESTATION_KINDS:
            children.setdefault(e['Parent_ID'], []).append(e['Child_ID'])
    queue=list(wanted)
    while queue:
        for child in children.get(queue.pop(), ()):
            if child not in wanted:
                wanted.add(child)
                queue.append(child)
    del children
    selected, statuses = [], {}
    with (cldf/'forms.csv').open() as f:
        reader=csv.DictReader(f); fields=reader.fieldnames
        for r in reader:
            statuses[r['ID']]=r['Status']
            if r['Language_ID'] in langs|{'Indo-Aryan'} or r['ID'] in wanted:
                selected.append(r)
    before = rank1_map(edges)
    metadata = {r['ID']:(r['Form'],r['Language_ID']) for r in selected}
    language_of = {i:meta[1] for i,meta in metadata.items()}
    old_align = {i:aligned_parent(before,language_of,i,phonetic.PROTO_LANGS) for i in metadata}
    audit = group_graph(selected,edges,groups,redirects,aliases,langs)
    statuses.update({r['ID']:r['Status'] for r in selected})
    validate_edge_dicts(edges,statuses)
    after=rank1_map(edges)
    new_align={i:aligned_parent(after,language_of,i,phonetic.PROTO_LANGS) for i in metadata}
    changed={i for i in metadata if old_align[i]!=new_align[i]}
    # Check the complete accepted graph for cycles, not just variant chains.
    done=set()
    for start in after:
        cur=start; seen=set()
        while cur in after and cur not in done:
            if cur in seen:raise ValueError(f'Accepted ancestry cycle at {cur}')
            seen.add(cur);cur=after[cur][0]
        done.update(seen)
    # Idempotence must hold on the full affected slice before any files are installed.
    if group_graph(selected,edges,groups,redirects,aliases,langs):
        raise ValueError('Nuristani grouping is not idempotent')
    if not audit:
        print(json.dumps(dict(unchanged=True, total_nodes=len(statuses), total_edges=len(edges))))
        return
    by_id={r['ID']:r for r in selected}
    with (args.out/'forms.csv').open('w',newline='') as out, (cldf/'forms.csv').open() as inp:
        w=csv.DictWriter(out,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in csv.DictReader(inp):w.writerow(by_id.get(r['ID'],r))
    with (args.out/'edges.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=EDGES_HEADER,lineterminator='\n');w.writeheader();w.writerows(edges)
    tok=phonetic.segmenter(phonetic.load_graphemes(str(ROOT/phonetic.PROFILE)))
    new_cells=0
    with (cldf/'alignments.csv').open() as inp, (args.out/'alignments.csv').open('w',newline='') as out:
        reader=csv.DictReader(inp)
        w=csv.DictWriter(out,fieldnames=reader.fieldnames,lineterminator='\n');w.writeheader()
        for r in reader:
            if r['Form_ID'] not in changed:w.writerow(r)
        es_cache={}
        for i in sorted(changed):
            parent=new_align[i]
            if not parent or parent not in metadata:continue
            word,lang=metadata[parent]
            if lang not in phonetic.PROTO_LANGS or not word or not metadata[i][0]:continue
            if parent not in es_cache:
                es=phonetic.segments(tok,word)
                for k,s in enumerate(es):s.idx=k
                es_cache[parent]=es
            es=es_cache[parent];rs=phonetic.segments(tok,metadata[i][0])
            if not es or not rs:continue
            for pos,(a,b) in enumerate(phonetic.align(es,rs)):
                w.writerow(dict(Form_ID=i,Origin_ID=parent,Pos=pos,Etymon_Idx=a.idx if a else -1,
                    Etymon_Seg=a.raw if a else '',Reflex_Seg=b.raw if b else '',Change=phonetic.describe(a,b),
                    Prev_Seg=es[a.idx-1].raw if a and a.idx>0 else '#' if a else '',
                    Next_Seg=es[a.idx+1].raw if a and a.idx+1<len(es) else '#' if a else ''))
                new_cells+=1
    report=dict(actions=dict(Counter(r['Action'] for r in audit)),changed_alignment_targets=len(changed),
                new_alignment_cells=new_cells,total_nodes=len(statuses),total_edges=len(edges),
                grouped_nodes=sum(GROUP_TAG in r['Tags'].split() for r in selected),
                validation='all edge endpoints/statuses, accepted-graph acyclicity, idempotence passed')
    with (args.out/'audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['Form_ID','Old_Parent','New_Parent','Old_Kind','New_Kind','Action']);w.writeheader();w.writerows(audit)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.install:
        if any((cldf/name).stat().st_mtime_ns!=stamp for name,stamp in source_stat.items()):
            raise ValueError('CLDF changed during preparation; inspect concurrent work before installing')
        backup=args.out/'before';backup.mkdir(exist_ok=True)
        for name in source_stat:
            if not (backup/name).exists():shutil.copy2(cldf/name,backup/name)
        for name in source_stat:
            temp=cldf/(name+'.nuristani-tmp')
            shutil.copy2(args.out/name,temp);os.replace(temp,cldf/name)
    print(json.dumps(dict(report,installed=args.install),indent=2))


if __name__=='__main__':
    main()
