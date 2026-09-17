"""Per-source settings, declared in a YAML file next to each lexical source.

Every lexical source CSV may have a sibling ``<stem>.yaml`` (``data/other/forms/<stem>.yaml``;
dictionary inputs use ``data/<dictionary>/source.yaml``; parameter sources
``data/other/params/<stem>.yaml``).  The YAML replaces the per-source ``if source_key == …``
branches that used to live in the pipeline scripts.  Only *settings* belong here; genuine
one-off data repairs stay in code.

Schema::

    file: 20230621-shina.csv            # informative; must match the YAML's stem
    defaults:                           # apply to every row of THIS file
      transcription: <rules>            # see below
      forms:
        exclude_languages: [Urdu, Pashto]   # rows in these languages are not published
      gloss:
        grammar_tags: true              # form_grammar strips checked grammatical annotations
      identity:
        legacy_ids: stem                # legacy ids are <stem>-<row> (appended source)
        append_order: 12                # processing position among appended sources
      importer:                         # how to regenerate this CSV: `make ingest SOURCE=<stem>`
        commands:
          - [data/other/forms/raw_data/sil_irula_2018/import_irula.py, --install]
        note: needs tmp/pdfs/… (working PDF)
    sources:                            # per citation key (column 8 before "["); a key may be
      schmidt:                          #   declared in exactly one YAML repo-wide
        transcription: <rules>
        forms:
          split_alternates: false       # commas are source notation, not alternate forms
        identity:
          dedupe_by_entry_key: true     # rows dedupe on (lang, param, form, Entry_Key)
          legacy_ids: file-order        # keep <file>-<row> ids even under a non-numeric etymon
          key_dialect_prefix: "dialect:Kho:"  # Entry_Key is per dialect attestation
          keyed_duplicates_allowed: true      # a repeated Entry_Key still yields stable keys
        gloss:
          source_defined_tags: true     # verb labels printed in the source table
        notes:
          audit_only: true              # Notes are ingestion metadata, kept out of public notes
        reference:
          editor: "Aryaman Arora; OpenAI Codex"
          ocr: true
          etymology_provenance: source  # source | source-mapped | jambu | mixed | none

``<rules>`` is a mapping or a list of mappings ``{profile, convert, languages,
exclude_languages, language_prefixes}``.  Rules are tried in order — the citation key's rules first, then the file's
``defaults`` — and the first whose language filter matches wins.  A rule naming a ``profile``
converts unless it says ``convert: false``; ``convert: false`` alone disables conversion.  With no
matching rule the row is not converted.

Loading needs PyYAML (available in the project environment: run tools with ``uv run``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
FORMS_DIR = ROOT / "data/other/forms"
PARAMS_DIR = ROOT / "data/other/params"
DICTIONARY_DIRS = ("cdial", "dedr", "munda", "dbia")

TOP_LEVEL_KEYS = {"file", "defaults", "sources"}
SECTIONS = {
    "transcription": None,  # rules; validated separately
    "forms": {"split_alternates", "exclude_languages"},
    "gloss": {"grammar_tags", "source_defined_tags"},
    "identity": {
        "legacy_ids", "append_order", "dedupe_by_entry_key", "key_dialect_prefix",
        "keyed_duplicates_allowed",
    },
    "notes": {"audit_only"},
    "reference": {"editor", "ocr", "etymology_provenance"},
    # file-level only: how to regenerate the source CSV (argv lists run with the project python
    # from the repo root, in order); `make ingest SOURCE=<stem>` runs them.
    "importer": {"commands", "note"},
}
RULE_KEYS = {"profile", "convert", "languages", "exclude_languages", "language_prefixes"}


class SourceMetaError(ValueError):
    pass


def yaml_files() -> list[Path]:
    files = sorted(FORMS_DIR.glob("*.yaml")) if FORMS_DIR.is_dir() else []
    files += sorted(PARAMS_DIR.glob("*.yaml")) if PARAMS_DIR.is_dir() else []
    for name in DICTIONARY_DIRS:
        path = ROOT / "data" / name / "source.yaml"
        if path.exists():
            files.append(path)
    return files


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment problem, not data
        raise SourceMetaError(
            "PyYAML is required to read per-source settings; run with `uv run python …`"
        ) from exc
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SourceMetaError(f"{path}: top level must be a mapping")
    return data


def _rules(value: Any, where: str) -> list[dict]:
    if value is None:
        return []
    rules = value if isinstance(value, list) else [value]
    out = []
    for rule in rules:
        if not isinstance(rule, dict) or not (set(rule) <= RULE_KEYS) or not rule:
            raise SourceMetaError(f"{where}: bad transcription rule {rule!r}")
        for key in ("languages", "exclude_languages", "language_prefixes"):
            if key in rule and not isinstance(rule[key], list):
                raise SourceMetaError(f"{where}: {key} must be a list")
        out.append(rule)
    return out


def _validate_block(block: Any, where: str) -> dict:
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise SourceMetaError(f"{where}: expected a mapping")
    for section, allowed in SECTIONS.items():
        if section not in block:
            continue
        if section == "transcription":
            block[section] = _rules(block[section], f"{where}.transcription")
            continue
        if not isinstance(block[section], dict):
            raise SourceMetaError(f"{where}.{section}: expected a mapping")
        unknown = set(block[section]) - allowed
        if unknown:
            raise SourceMetaError(f"{where}.{section}: unknown keys {sorted(unknown)}")
    unknown = set(block) - set(SECTIONS)
    if unknown:
        raise SourceMetaError(f"{where}: unknown sections {sorted(unknown)}")
    return block


class SourceMeta:
    """All per-source settings, indexed by file stem and by citation key."""

    def __init__(self, paths: Iterable[Path] | None = None):
        self.files: dict[str, dict] = {}       # stem (or "<dictionary>/source") -> defaults block
        self.file_paths: dict[str, Path] = {}
        self.sources: dict[str, dict] = {}     # citation key -> settings block
        self.source_owner: dict[str, str] = {}  # citation key -> stem declaring it
        for path in (yaml_files() if paths is None else paths):
            self._add(Path(path))

    def _stem(self, path: Path) -> str:
        if path.name == "source.yaml":
            return f"{path.parent.name}/source"
        return path.stem

    def _add(self, path: Path) -> None:
        data = _load_yaml(path)
        unknown = set(data) - TOP_LEVEL_KEYS
        if unknown:
            raise SourceMetaError(f"{path}: unknown top-level keys {sorted(unknown)}")
        stem = self._stem(path)
        declared = data.get("file")
        if declared and Path(str(declared)).stem != path.stem and path.name != "source.yaml":
            raise SourceMetaError(f"{path}: `file: {declared}` does not match the YAML name")
        self.files[stem] = _validate_block(data.get("defaults"), f"{path}:defaults")
        self.file_paths[stem] = path
        sources = data.get("sources") or {}
        if not isinstance(sources, dict):
            raise SourceMetaError(f"{path}: `sources` must be a mapping of citation keys")
        for key, block in sources.items():
            if key in self.sources:
                raise SourceMetaError(
                    f"{path}: citation key {key!r} is already declared in "
                    f"{self.file_paths[self.source_owner[key]]}"
                )
            self.sources[str(key)] = _validate_block(block, f"{path}:sources.{key}")
            self.source_owner[str(key)] = stem

    # ------------------------------------------------------------------ lookups

    @staticmethod
    def stem_for(file: str | os.PathLike[str]) -> str:
        """Stem used to find a file's defaults: ``20230621-shina`` or ``munda/source``."""
        path = Path(file)
        parts = path.parts
        if len(parts) >= 2 and parts[-2] in DICTIONARY_DIRS:
            return f"{parts[-2]}/source"
        return path.stem

    def file_defaults(self, file: str | os.PathLike[str]) -> dict:
        return self.files.get(self.stem_for(file), {})

    def source(self, citation_key: str) -> dict:
        return self.sources.get(citation_key, {})

    def flag(self, citation_key: str, section: str, name: str, default: Any = None) -> Any:
        return self.source(citation_key).get(section, {}).get(name, default)

    def file_flag(self, file: str | os.PathLike[str], section: str, name: str, default: Any = None) -> Any:
        return self.file_defaults(file).get(section, {}).get(name, default)

    def keys_where(self, section: str, name: str, value: Any = True) -> frozenset[str]:
        return frozenset(k for k, b in self.sources.items() if b.get(section, {}).get(name) == value)

    def files_where(self, section: str, name: str, value: Any = True) -> list[str]:
        return [s for s, b in self.files.items() if b.get(section, {}).get(name) == value]

    def transcription(
        self, citation_key: str, file: str | os.PathLike[str], language: str
    ) -> tuple[str | None, bool]:
        """(profile, convert) for one row: the citation key's rules, then the file's defaults."""
        rules = list(self.source(citation_key).get("transcription", []))
        rules += self.file_defaults(file).get("transcription", [])
        for rule in rules:
            if "languages" in rule and language not in rule["languages"]:
                continue
            if "exclude_languages" in rule and language in rule["exclude_languages"]:
                continue
            if "language_prefixes" in rule and not language.startswith(tuple(rule["language_prefixes"])):
                continue
            profile = rule.get("profile")
            convert = rule.get("convert", profile is not None)
            return (profile if convert else profile), bool(convert)
        return None, False


@lru_cache(maxsize=1)
def load() -> SourceMeta:
    return SourceMeta()


def importer_commands(meta: "SourceMeta", stem: str) -> list[list[str]]:
    block = meta.files.get(stem, {}).get("importer", {})
    commands = block.get("commands") or []
    if not commands or not all(isinstance(c, list) and c for c in commands):
        raise SourceMetaError(f"{stem}: no `importer.commands` declared in its YAML")
    return [[str(part) for part in command] for command in commands]


def run_importer(stem: str, dry_run: bool = False) -> int:
    """Regenerate one source CSV by running its declared importer commands from the repo root."""
    import subprocess
    import sys

    meta = load()
    commands = importer_commands(meta, stem)
    note = meta.files[stem].get("importer", {}).get("note")
    if note:
        print(f"note: {note}")
    for command in commands:
        argv = [sys.executable, *command]
        print("$", " ".join(argv))
        if not dry_run:
            subprocess.run(argv, cwd=ROOT, check=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate and summarise per-source YAML settings.")
    parser.add_argument("--profiles", default=str(ROOT / "conversion"),
                        help="directory of transcription profiles to check rule names against")
    sub = parser.add_subparsers(dest="command")
    ingest = sub.add_parser("ingest", help="run a source's declared importer (make ingest SOURCE=<stem>)")
    ingest.add_argument("stem")
    ingest.add_argument("--dry-run", action="store_true")
    sub.add_parser("importers", help="list sources that declare an importer")
    args = parser.parse_args(argv)
    if args.command == "ingest":
        return run_importer(args.stem, dry_run=args.dry_run)
    if args.command == "importers":
        meta = load()
        for stem in sorted(meta.files):
            block = meta.files[stem].get("importer")
            if block:
                print(f"{stem:45s} {' && '.join(' '.join(c) for c in block['commands'])}")
        return 0
    meta = SourceMeta()
    profiles = {p.stem for p in Path(args.profiles).glob("*.txt")}
    bad = []
    for scope, blocks in (("file", meta.files), ("source", meta.sources)):
        for name, block in blocks.items():
            for rule in block.get("transcription", []):
                if rule.get("profile") and rule["profile"] not in profiles:
                    bad.append((scope, name, rule["profile"]))
    print(f"{len(meta.files)} source files with settings, {len(meta.sources)} citation keys")
    if bad:
        print("transcription profiles that do not exist under conversion/:")
        for item in bad:
            print("  ", *item)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
