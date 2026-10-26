"""Read-only candidate metadata/formatter check; report outside lexical freeze."""
import hashlib
import json
from pathlib import Path
import yaml
from pybtex.database import parse_file
from pybtex.plugin import find_plugin

HERE = Path(__file__).resolve().parent


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    bib = HERE / "proposed-references.bib"
    settings_path = HERE / "proposed-settings.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    data = parse_file(str(bib))
    assert set(data.entries)==set(settings["sources"])=={"nair2013mannan","joseph1980devikolam"}
    assert settings["defaults"]["transcription"]=={"convert":False,"preserve_hyphens":True}
    assert settings["defaults"]["identity"]["dedupe_by_entry_key"] is True
    formatted = []
    style = find_plugin("pybtex.style.formatting","plain")()
    for entry in style.format_bibliography(data):
        text = entry.text.render_as("text")
        assert "Devikolam" in text if entry.key=="joseph1980devikolam" else "Kerala" in text
        formatted.append(dict(key=entry.key,text=text))
    for key,entry in data.entries.items():
        assert entry.fields["ocr"]=="No"
        assert entry.fields["jambu_editor"]=="OpenAI Codex"
        assert settings["sources"][key]["reference"]["ocr"] is False
        assert settings["sources"][key]["reference"]["etymology_provenance"]==entry.fields["etymology_provenance"]
    assert data.entries["nair2013mannan"].fields["pages"]=="45--49"
    assert data.entries["joseph1980devikolam"].fields["year"]=="1980"
    report = dict(status="passed source-local metadata/formatting only",inputs={"Bib":sha(bib),"YAML":sha(settings_path)},
        formatted_references=formatted,source_keys_once_each=True,ocr_false=True,primary_etymology_source=True,
        authority_scope="Annie Joseph1980 is reported background; no invented thesis page/edition or independently collated lexical claim.",
        no_canonical_writes=True)
    (HERE / "reference-formatter-review.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__": main()
