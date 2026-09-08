import json

from conftest import make_ocr_output_dict

from newspaper_reconstructor.ingest import (
    load_article_json,
    load_fragments_file,
    load_ocr_output_json,
)


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


class TestLoadOcrOutputJson:
    def test_loads_module_format_regions(self, tmp_path):
        p = tmp_path / "page.json"
        p.write_text(json.dumps(make_ocr_output_dict()), encoding="utf-8")
        result = load_ocr_output_json(str(p))
        assert len(result) == 1

    def test_fragment_fields(self, tmp_path):
        p = tmp_path / "page.json"
        p.write_text(json.dumps(make_ocr_output_dict()), encoding="utf-8")
        result = load_ocr_output_json(str(p))[0]
        assert result["id"] == "r_1"
        assert result["text"] == "Hello world"
        assert result["type"] == "text"
        assert result["hpos"] == 10
        assert result["vpos"] == 20
        assert result["width"] == 200
        assert result["height"] == 100

    def test_joins_multiple_lines(self, tmp_path):
        data = make_ocr_output_dict()
        second = dict(data["regions"][0]["line_ocr"][0])
        second["text"] = "second line"
        data["regions"][0]["line_ocr"].append(second)
        p = tmp_path / "page.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_ocr_output_json(str(p))[0]
        assert result["text"] == "Hello world second line"

    def test_empty_regions(self, tmp_path):
        data = make_ocr_output_dict(regions=[])
        p = tmp_path / "page.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        assert load_ocr_output_json(str(p)) == []


class TestLoadFragmentsFile:
    def test_dispatches_module_format(self, tmp_path):
        p = tmp_path / "page.json"
        p.write_text(json.dumps(make_ocr_output_dict()), encoding="utf-8")
        result = load_fragments_file(str(p))
        assert result[0]["id"] == "r_1"
        assert "hpos" in result[0]

    def test_dispatches_article_json(self, tmp_path):
        p = tmp_path / "page.json"
        p.write_text(json.dumps({"r_1": "Hello"}), encoding="utf-8")
        result = load_fragments_file(str(p))
        assert result == [{"id": "r_1", "text": "Hello"}]

    def test_dispatches_fragment_list(self, tmp_path):
        fragments = [{"id": "r_1", "text": "Hello", "type": "text"}]
        p = tmp_path / "page.json"
        p.write_text(json.dumps(fragments), encoding="utf-8")
        assert load_fragments_file(str(p)) == fragments


class TestSlimFormat:
    def test_load_ocr_output_json_slim(self, tmp_path):
        p = tmp_path / "page.json"
        p.write_text(json.dumps(make_ocr_output_dict()), encoding="utf-8")
        result = load_ocr_output_json(str(p), slim=True)
        assert len(result) == 1
        assert set(result[0].keys()) == {"id", "text"}
        assert result[0]["id"] == "r_1"
        assert result[0]["text"] == "Hello world"

    def test_load_ocr_output_json_default_keeps_geometry(self, tmp_path):
        p = tmp_path / "page.json"
        p.write_text(json.dumps(make_ocr_output_dict()), encoding="utf-8")
        result = load_ocr_output_json(str(p))
        assert {"hpos", "vpos", "width", "height", "type"} <= set(result[0])

    def test_load_fragments_file_slim_module_format(self, tmp_path):
        p = tmp_path / "page.json"
        p.write_text(json.dumps(make_ocr_output_dict()), encoding="utf-8")
        result = load_fragments_file(str(p), slim=True)
        assert set(result[0].keys()) == {"id", "text"}

    def test_load_fragments_file_slim_fragment_list(self, tmp_path):
        fragments = [{"id": "r_1", "text": "Hello", "type": "text", "hpos": 1}]
        p = tmp_path / "page.json"
        p.write_text(json.dumps(fragments), encoding="utf-8")
        result = load_fragments_file(str(p), slim=True)
        assert result == [{"id": "r_1", "text": "Hello"}]

    def test_load_fragments_file_slim_article_json(self, tmp_path):
        p = tmp_path / "page.json"
        p.write_text(json.dumps({"r_1": "Hello"}), encoding="utf-8")
        result = load_fragments_file(str(p), slim=True)
        assert result == [{"id": "r_1", "text": "Hello"}]
