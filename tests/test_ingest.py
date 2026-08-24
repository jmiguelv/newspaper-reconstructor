import json

from src.newspaper_reconstructor.ingest import load_article_json


class TestLoadArticleJson:
    def test_loads_key_value_pairs(self, tmp_path):
        data = {"r_1": "Hello", "r_2": "World"}
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_article_json(str(p))
        assert len(result) == 2

    def test_extracts_id_and_text(self, tmp_path):
        data = {"r_abc": "Some OCR text"}
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_article_json(str(p))
        assert result[0]["id"] == "r_abc"
        assert result[0]["text"] == "Some OCR text"

    def test_preserves_all_entries(self, tmp_path):
        data = {"r_1": "A", "r_2": "B", "r_3": "C"}
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_article_json(str(p))
        ids = {f["id"] for f in result}
        assert ids == {"r_1", "r_2", "r_3"}

    def test_handles_empty_text(self, tmp_path):
        data = {"r_1": "", "r_2": "Text"}
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_article_json(str(p))
        assert len(result) == 2
        empty = next(f for f in result if f["id"] == "r_1")
        assert empty["text"] == ""

    def test_handles_unicode_text(self, tmp_path):
        data = {"r_1": "فکرج٢ ايس"}
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        result = load_article_json(str(p))
        assert result[0]["text"] == "فکرج٢ ايس"

    def test_empty_json(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text("{}", encoding="utf-8")
        result = load_article_json(str(p))
        assert result == []

    def test_fragment_dict_keys(self, tmp_path):
        data = {"r_1": "text"}
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_article_json(str(p))
        assert set(result[0].keys()) == {"id", "text"}
