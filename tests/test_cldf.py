from pycldf import Dataset

def test_validate():
    d = Dataset.from_metadata("cldf/Wordlist-metadata.json")
    assert d.validate()