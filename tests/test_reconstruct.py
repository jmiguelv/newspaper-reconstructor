"""Tests for reconstruct.py — ALTO parsing, LLM client, article reconstruction."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from llm import make_client
from reconstruct import (
    alto_to_json,
    load_fragments,
    load_fragments_cached,
    reconstruct_articles,
    reconstruct_articles_cached,
)

TEST_SYSTEM_PROMPT = "You are an expert at reconstructing articles from text fragments."
TEST_USER_PROMPT_TEMPLATE = "Fragments:\n\n{fragments}\n\nReturn ONLY a JSON array."

# ─── alto_to_json ───────────────────────────────────────────────────────────

ALTO_SAMPLE = """<?xml version='1.0' encoding='utf-8'?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout>
    <Page ID="test_page" WIDTH="1000" HEIGHT="2000">
      <PrintSpace HPOS="0" VPOS="0" WIDTH="1000" HEIGHT="2000">
        <Illustration ID="r_ill1" HPOS="10" VPOS="10" WIDTH="100" HEIGHT="100" TYPE="image" />
        <TextBlock ID="r_text1" HPOS="20" VPOS="30" WIDTH="200" HEIGHT="400" TYPE="text">
          <TextLine ID="line1_0" HPOS="20" VPOS="30" WIDTH="200" HEIGHT="50">
            <String ID="s1" CONTENT="Hello" HPOS="20" VPOS="30" WIDTH="100" HEIGHT="50" />
          </TextLine>
          <TextLine ID="line1_1" HPOS="20" VPOS="80" WIDTH="200" HEIGHT="50">
            <String ID="s2" CONTENT="World" HPOS="20" VPOS="80" WIDTH="100" HEIGHT="50" />
          </TextLine>
        </TextBlock>
        <TextBlock ID="r_text2" HPOS="300" VPOS="30" WIDTH="200" HEIGHT="100" TYPE="text">
          <TextLine ID="line2_0" HPOS="300" VPOS="30" WIDTH="200" HEIGHT="100">
            <String ID="s3" CONTENT="Solo" HPOS="300" VPOS="30" WIDTH="100" HEIGHT="100" />
          </TextLine>
        </TextBlock>
        <TextBlock ID="r_empty" HPOS="500" VPOS="30" WIDTH="200" HEIGHT="100" TYPE="text">
          <TextLine ID="line3_0" HPOS="500" VPOS="30" WIDTH="200" HEIGHT="100">
            <String ID="s4" CONTENT="" HPOS="500" VPOS="30" WIDTH="100" HEIGHT="100" />
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>"""


class TestAltoJson:
    def test_parses_text_blocks(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_text(ALTO_SAMPLE, encoding="utf-8")
        result = alto_to_json(str(p))
        assert len(result) == 2  # ill skipped, empty filtered

    def test_extracts_id_and_text(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_text(ALTO_SAMPLE, encoding="utf-8")
        result = alto_to_json(str(p))
        first = next(r for r in result if r["id"] == "r_text1")
        assert first["text"] == "Hello World"

    def test_extracts_bounding_box(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_text(ALTO_SAMPLE, encoding="utf-8")
        result = alto_to_json(str(p))
        first = next(r for r in result if r["id"] == "r_text1")
        assert first["hpos"] == 20
        assert first["vpos"] == 30
        assert first["width"] == 200
        assert first["height"] == 400

    def test_extracts_type(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_text(ALTO_SAMPLE, encoding="utf-8")
        result = alto_to_json(str(p))
        first = next(r for r in result if r["id"] == "r_text1")
        assert first["type"] == "text"

    def test_skips_illustration(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_text(ALTO_SAMPLE, encoding="utf-8")
        result = alto_to_json(str(p))
        assert all(r["id"] != "r_ill1" for r in result)

    def test_filters_empty_text(self, tmp_path):
        p = tmp_path / "test.xml"
        p.write_text(ALTO_SAMPLE, encoding="utf-8")
        result = alto_to_json(str(p))
        assert all(r["id"] != "r_empty" for r in result)

    def test_handles_missing_type(self, tmp_path):
        alto_no_type = """<?xml version='1.0' encoding='utf-8'?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout><Page ID="p" WIDTH="100" HEIGHT="100">
    <PrintSpace HPOS="0" VPOS="0" WIDTH="100" HEIGHT="100">
      <TextBlock ID="r_no_type" HPOS="0" VPOS="0" WIDTH="50" HEIGHT="50">
        <TextLine ID="l1" HPOS="0" VPOS="0" WIDTH="50" HEIGHT="50">
          <String ID="s1" CONTENT="Text" HPOS="0" VPOS="0" WIDTH="50" HEIGHT="50" />
        </TextLine>
      </TextBlock>
    </PrintSpace>
  </Page></Layout>
</alto>"""
        p = tmp_path / "test.xml"
        p.write_text(alto_no_type, encoding="utf-8")
        result = alto_to_json(str(p))
        assert len(result) == 1
        assert result[0]["id"] == "r_no_type"
        assert result[0]["type"] is None


# ─── load_fragments ──────────────────────────────────────────────────────────


class TestLoadFragments:
    def test_loads_xml_file(self, tmp_path):
        p = tmp_path / "page.xml"
        p.write_text(ALTO_SAMPLE, encoding="utf-8")
        result = load_fragments(str(p))
        assert len(result) == 2
        assert result[0]["id"] in ("r_text1", "r_text2")

    def test_loads_json_file(self, tmp_path):
        data = [{"id": "r_1", "text": "hello", "type": "text"}]
        p = tmp_path / "page.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_fragments(str(p))
        assert result == data

    def test_loads_directory(self, tmp_path):
        d = tmp_path / "pages"
        d.mkdir()
        (d / "page1.xml").write_text(ALTO_SAMPLE, encoding="utf-8")
        data = [{"id": "r_99", "text": "json page", "type": "text"}]
        (d / "page2.json").write_text(json.dumps(data), encoding="utf-8")
        result = load_fragments(str(d))
        assert "page1" in result
        assert "page2" in result
        assert len(result["page1"]) == 2
        assert len(result["page2"]) == 1


# ─── load_fragments_cached ───────────────────────────────────────────────────


class TestLoadFragmentsCached:
    def test_caches_xml_to_interim(self, tmp_path):
        src = tmp_path / "page.xml"
        src.write_text(ALTO_SAMPLE, encoding="utf-8")
        interim = tmp_path / "interim"

        result = load_fragments_cached(str(src), str(interim), force=False)
        assert len(result) == 2
        assert (interim / "fragments" / "page.json").exists()
        cached = json.loads((interim / "fragments" / "page.json").read_text())
        assert len(cached) == 2

    def test_loads_from_cache_when_exists(self, tmp_path):
        src = tmp_path / "page.xml"
        src.write_text(ALTO_SAMPLE, encoding="utf-8")
        interim = tmp_path / "interim"
        interim.mkdir()
        (interim / "fragments").mkdir()
        cached_data = [{"id": "cached", "text": "cached text", "type": "text"}]
        (interim / "fragments" / "page.json").write_text(
            json.dumps(cached_data), encoding="utf-8"
        )

        result = load_fragments_cached(str(src), str(interim), force=False)
        assert result == cached_data

    def test_force_reparses_even_with_cache(self, tmp_path):
        src = tmp_path / "page.xml"
        src.write_text(ALTO_SAMPLE, encoding="utf-8")
        interim = tmp_path / "interim"
        interim.mkdir()
        (interim / "fragments").mkdir()
        (interim / "fragments" / "page.json").write_text(
            json.dumps([{"id": "stale", "text": "stale", "type": "text"}]),
            encoding="utf-8",
        )

        result = load_fragments_cached(str(src), str(interim), force=True)
        assert len(result) == 2
        assert result[0]["id"] in ("r_text1", "r_text2")

    def test_json_files_pass_through_uncached(self, tmp_path):
        data = [{"id": "r_1", "text": "hello", "type": "text"}]
        src = tmp_path / "page.json"
        src.write_text(json.dumps(data), encoding="utf-8")
        interim = tmp_path / "interim"

        result = load_fragments_cached(str(src), str(interim), force=False)
        assert result == data
        assert not (interim / "fragments" / "page.json").exists()

    def test_caches_directory_of_xml_files(self, tmp_path):
        src = tmp_path / "pages"
        src.mkdir()
        (src / "page1.xml").write_text(ALTO_SAMPLE, encoding="utf-8")
        interim = tmp_path / "interim"

        result = load_fragments_cached(str(src), str(interim), force=False)
        assert "page1" in result
        assert len(result["page1"]) == 2
        assert (interim / "fragments" / "page1.json").exists()

    def test_second_run_uses_cache(self, tmp_path):
        src = tmp_path / "page.xml"
        src.write_text(ALTO_SAMPLE, encoding="utf-8")
        interim = tmp_path / "interim"

        first = load_fragments_cached(str(src), str(interim), force=False)
        assert len(first) == 2

        src.unlink()
        second = load_fragments_cached(str(src), str(interim), force=False)
        assert second == first

    def test_no_interim_dir_falls_back_to_load_fragments(self, tmp_path):
        src = tmp_path / "page.xml"
        src.write_text(ALTO_SAMPLE, encoding="utf-8")
        result = load_fragments_cached(str(src), None, force=False)
        assert len(result) == 2


# ─── make_client ─────────────────────────────────────────────────────────────


class TestMakeClient:
    def test_returns_client_with_env_vars(self):
        with patch.dict(
            os.environ, {"LLM_API_KEY": "test-key", "LLM_MODEL": "test-model"}
        ):
            client = make_client()
            assert client is not None
            assert client.model == "test-model"

    def test_raises_without_model(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="Model not set"),
        ):
            make_client()

    def test_api_key_defaults_to_none(self):
        with patch.dict(os.environ, {"LLM_MODEL": "test-model"}, clear=True):
            client = make_client()
            assert client is not None
            assert client.model == "test-model"

    def test_accepts_explicit_args(self):
        client = make_client(
            api_key="explicit", model="my-model", base_url="http://localhost:8080"
        )
        assert client is not None
        assert client.model == "my-model"
        assert client.base_url == "http://localhost:8080"


# ─── reconstruct_articles ─────────────────────────────────────────────────────

VALID_RESPONSE = json.dumps(
    [
        {"fragment_ids": ["r_text1"], "title": "Greeting", "class": "article"},
        {"fragment_ids": ["r_text2"], "title": "Solo ad", "class": "advertisement"},
    ]
)

FENCED_RESPONSE = """Here are the results:
```json
[{"fragment_ids": ["r_text1"], "title": "Greeting", "class": "article"}]
```
Hope this helps!"""

INVALID_RESPONSE = "I cannot help with that."


class TestReconstructArticles:
    def test_parses_valid_json(self):
        client = MagicMock()
        client.complete.return_value = VALID_RESPONSE
        fragments = [
            {"id": "r_text1", "text": "Hello World"},
            {"id": "r_text2", "text": "Solo"},
        ]
        result = reconstruct_articles(
            fragments, client, TEST_SYSTEM_PROMPT, TEST_USER_PROMPT_TEMPLATE
        )
        assert len(result) == 2
        assert result[0]["fragment_ids"] == ["r_text1"]
        assert result[1]["class"] == "advertisement"

    def test_parses_fenced_json(self):
        client = MagicMock()
        client.complete.return_value = FENCED_RESPONSE
        fragments = [{"id": "r_text1", "text": "Hello World"}]
        result = reconstruct_articles(
            fragments, client, TEST_SYSTEM_PROMPT, TEST_USER_PROMPT_TEMPLATE
        )
        assert len(result) == 1
        assert result[0]["title"] == "Greeting"

    def test_returns_none_on_invalid_json(self):
        client = MagicMock()
        client.complete.return_value = INVALID_RESPONSE
        fragments = [{"id": "r_text1", "text": "Hello"}]
        result = reconstruct_articles(
            fragments, client, TEST_SYSTEM_PROMPT, TEST_USER_PROMPT_TEMPLATE
        )
        assert result is None

    def test_all_fragment_ids_from_input(self):
        client = MagicMock()
        client.complete.return_value = VALID_RESPONSE
        fragments = [
            {"id": "r_text1", "text": "Hello World"},
            {"id": "r_text2", "text": "Solo"},
        ]
        result = reconstruct_articles(
            fragments, client, TEST_SYSTEM_PROMPT, TEST_USER_PROMPT_TEMPLATE
        )
        all_ids = set()
        for item in result:
            all_ids.update(item["fragment_ids"])
        assert all_ids == {"r_text1", "r_text2"}

    def test_single_fragment_items(self):
        client = MagicMock()
        client.complete.return_value = json.dumps(
            [
                {"fragment_ids": ["r_1", "r_2"], "title": "Multi", "class": "article"},
                {"fragment_ids": ["r_3"], "title": "Single", "class": "advertisement"},
            ]
        )
        fragments = [
            {"id": "r_1", "text": "a"},
            {"id": "r_2", "text": "b"},
            {"id": "r_3", "text": "c"},
        ]
        result = reconstruct_articles(
            fragments, client, TEST_SYSTEM_PROMPT, TEST_USER_PROMPT_TEMPLATE
        )
        assert any(len(i["fragment_ids"]) == 1 for i in result)

    def test_retries_on_first_failure(self):
        client = MagicMock()
        client.complete.side_effect = ["not json", VALID_RESPONSE]
        fragments = [
            {"id": "r_text1", "text": "Hello World"},
            {"id": "r_text2", "text": "Ad text"},
        ]
        result = reconstruct_articles(
            fragments, client, TEST_SYSTEM_PROMPT, TEST_USER_PROMPT_TEMPLATE
        )
        assert len(result) == 2
        assert client.complete.call_count == 2

    def test_custom_prompts_passed_to_client(self):
        client = MagicMock()
        client.complete.return_value = VALID_RESPONSE
        fragments = [{"id": "r_text1", "text": "Hello World"}]
        reconstruct_articles(
            fragments,
            client,
            system_prompt="custom sys",
            user_prompt_template="custom user: {fragments}",
        )
        call_args = client.complete.call_args
        assert call_args[0][0] == "custom sys"
        assert call_args[0][1].startswith("custom user:")

    def test_fragments_sent_as_json(self):
        client = MagicMock()
        client.complete.return_value = VALID_RESPONSE
        fragments = [
            {
                "id": "r_1",
                "text": "hello",
                "hpos": 20,
                "vpos": 30,
                "width": 200,
                "height": 50,
            },
        ]
        reconstruct_articles(
            fragments, client, TEST_SYSTEM_PROMPT, TEST_USER_PROMPT_TEMPLATE
        )
        user_prompt = client.complete.call_args[0][1]
        assert '"id": "r_1"' in user_prompt
        assert '"hpos": 20' in user_prompt
        assert '"vpos": 30' in user_prompt
        assert '"text": "hello"' in user_prompt

    def test_retries_on_api_error(self):
        from openai import APIError

        client = MagicMock()
        client.complete.side_effect = [
            APIError(message="500", request=None, body=None),
            VALID_RESPONSE,
        ]
        fragments = [
            {"id": "r_text1", "text": "Hello World"},
            {"id": "r_text2", "text": "Ad text"},
        ]
        with patch("reconstruct.time.sleep"):
            result = reconstruct_articles(
                fragments,
                client,
                TEST_SYSTEM_PROMPT,
                TEST_USER_PROMPT_TEMPLATE,
                max_retries=3,
            )
        assert len(result) == 2
        assert client.complete.call_count == 2

    def test_returns_none_after_all_retries_fail(self):
        from openai import APIError

        client = MagicMock()
        client.complete.side_effect = APIError(message="500", request=None, body=None)
        fragments = [{"id": "r_text1", "text": "Hello World"}]
        with patch("reconstruct.time.sleep"):
            result = reconstruct_articles(
                fragments,
                client,
                TEST_SYSTEM_PROMPT,
                TEST_USER_PROMPT_TEMPLATE,
                max_retries=2,
            )
        assert result is None
        assert client.complete.call_count == 2

    def test_retries_on_timeout_error(self):
        from openai import APITimeoutError

        client = MagicMock()
        mock_req = MagicMock()
        client.complete.side_effect = APITimeoutError(request=mock_req)
        fragments = [
            {"id": "r_text1", "text": "Hello World"},
            {"id": "r_text2", "text": "Ad text"},
        ]
        result = reconstruct_articles(
            fragments,
            client,
            TEST_SYSTEM_PROMPT,
            TEST_USER_PROMPT_TEMPLATE,
            max_retries=3,
        )
        assert result is None
        assert client.complete.call_count == 1

    def test_returns_none_after_all_timeouts(self):
        from openai import APITimeoutError

        client = MagicMock()
        mock_req = MagicMock()
        client.complete.side_effect = APITimeoutError(request=mock_req)
        fragments = [{"id": "r_text1", "text": "Hello World"}]
        result = reconstruct_articles(
            fragments,
            client,
            TEST_SYSTEM_PROMPT,
            TEST_USER_PROMPT_TEMPLATE,
            max_retries=2,
        )
        assert result is None
        assert client.complete.call_count == 1


# ─── reconstruct_articles_cached ──────────────────────────────────────────────

SAMPLE_FRAGMENTS = [{"id": "r_1", "text": "hello world", "type": "text"}]
SAMPLE_RESULT = [
    {"fragment_ids": ["r_1"], "title": "Hello", "class": "article"},
]


class TestReconstructArticlesCached:
    def test_cache_miss_calls_llm_and_saves(self, tmp_path):
        client = MagicMock()
        client.complete.return_value = json.dumps(SAMPLE_RESULT)

        result = reconstruct_articles_cached(
            SAMPLE_FRAGMENTS,
            client,
            TEST_SYSTEM_PROMPT,
            TEST_USER_PROMPT_TEMPLATE,
            page_id="test_page",
            interim_dir=str(tmp_path / "interim"),
            prompt_name="default",
            model="test-model",
            force=False,
        )
        assert result == SAMPLE_RESULT
        assert client.complete.call_count == 1

        cache_path = (
            tmp_path
            / "interim"
            / "reconstructions"
            / "default"
            / "test-model"
            / "test_page.json"
        )
        assert cache_path.exists()
        assert json.loads(cache_path.read_text()) == SAMPLE_RESULT

    def test_cache_hit_skips_llm(self, tmp_path):
        cache_dir = tmp_path / "interim" / "reconstructions" / "default" / "test-model"
        cache_dir.mkdir(parents=True)
        cache_path = cache_dir / "test_page.json"
        cache_path.write_text(json.dumps(SAMPLE_RESULT), encoding="utf-8")

        client = MagicMock()

        result = reconstruct_articles_cached(
            SAMPLE_FRAGMENTS,
            client,
            TEST_SYSTEM_PROMPT,
            TEST_USER_PROMPT_TEMPLATE,
            page_id="test_page",
            interim_dir=str(tmp_path / "interim"),
            prompt_name="default",
            model="test-model",
            force=False,
        )
        assert result == SAMPLE_RESULT
        assert client.complete.call_count == 0

    def test_force_bypasses_cache(self, tmp_path):
        cache_dir = tmp_path / "interim" / "reconstructions" / "default" / "test-model"
        cache_dir.mkdir(parents=True)
        cache_path = cache_dir / "test_page.json"
        cache_path.write_text(json.dumps(SAMPLE_RESULT), encoding="utf-8")

        client = MagicMock()
        client.complete.return_value = json.dumps(
            [{"fragment_ids": ["r_1"], "title": "Changed", "class": "miscellaneous"}]
        )

        result = reconstruct_articles_cached(
            SAMPLE_FRAGMENTS,
            client,
            TEST_SYSTEM_PROMPT,
            TEST_USER_PROMPT_TEMPLATE,
            page_id="test_page",
            interim_dir=str(tmp_path / "interim"),
            prompt_name="default",
            model="test-model",
            force=True,
        )
        assert result[0]["title"] == "Changed"
        assert client.complete.call_count == 1
        assert json.loads(cache_path.read_text()) == result

    def test_returns_none_on_llm_failure(self, tmp_path):
        from openai import APIError

        client = MagicMock()
        client.complete.side_effect = APIError(message="504", request=None, body=None)

        result = reconstruct_articles_cached(
            SAMPLE_FRAGMENTS,
            client,
            TEST_SYSTEM_PROMPT,
            TEST_USER_PROMPT_TEMPLATE,
            page_id="test_page",
            interim_dir=str(tmp_path / "interim"),
            prompt_name="default",
            model="test-model",
            force=False,
        )
        assert result is None
        # No cache file created on failure
        cache_path = (
            tmp_path
            / "interim"
            / "reconstructions"
            / "default"
            / "test-model"
            / "test_page.json"
        )
        assert not cache_path.exists()

    def test_different_prompt_names_have_separate_caches(self, tmp_path):
        v1_result = [{"fragment_ids": ["r_1"], "title": "v1", "class": "article"}]
        v2_result = [{"fragment_ids": ["r_1"], "title": "v2", "class": "article"}]

        client_v1 = MagicMock()
        client_v1.complete.return_value = json.dumps(v1_result)
        client_v2 = MagicMock()
        client_v2.complete.return_value = json.dumps(v2_result)

        # Cache with v1 prompt
        result1 = reconstruct_articles_cached(
            SAMPLE_FRAGMENTS,
            client_v1,
            TEST_SYSTEM_PROMPT,
            TEST_USER_PROMPT_TEMPLATE,
            page_id="test_page",
            interim_dir=str(tmp_path / "interim"),
            prompt_name="v1",
            model="test-model",
            force=False,
        )
        assert result1[0]["title"] == "v1"

        # Cache with v2 prompt should be separate
        result2 = reconstruct_articles_cached(
            SAMPLE_FRAGMENTS,
            client_v2,
            TEST_SYSTEM_PROMPT,
            TEST_USER_PROMPT_TEMPLATE,
            page_id="test_page",
            interim_dir=str(tmp_path / "interim"),
            prompt_name="v2",
            model="test-model",
            force=False,
        )
        assert result2[0]["title"] == "v2"
        assert result1 != result2
