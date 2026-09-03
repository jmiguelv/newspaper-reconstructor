"""Tests for the jawi-pipeline module: adapter, process, bulk_process, CLI."""

import json
from unittest.mock import MagicMock, patch

import pytest
from jawi_pipeline.types import (
    ArticleReconstructionInput,
    BBox,
    Coordinate,
    ImageRegion,
    Line,
    LineOcr,
    Page,
    PageMetadata,
    TextImageRegionWithOcr,
    TextRegionWithOcr,
)
from jawi_pipeline.types.module import BaseLine

from src.newspaper_reconstructor.module import (
    ArticleReconstructionConfig,
    ArticleReconstructionModule,
    input_to_fragments,
)


def make_bbox(x=10.0, y=20.0, w=200.0, h=100.0) -> BBox:
    return BBox(x1=x, y1=y, x2=x + w, y2=y, x3=x + w, y3=y + h, x4=x, y4=y + h)


def make_line_ocr(i: int, text: str, conf: float = 0.9) -> LineOcr:
    return LineOcr(
        line=Line(
            id=f"l{i}",
            baseline=BaseLine(x1=0.0, y1=0.0, x2=10.0, y2=0.0),
            boundaries=[Coordinate(x=0.0, y=0.0), Coordinate(x=10.0, y=0.0)],
        ),
        script="jawi",
        text=text,
        conf=conf,
        glyph_loc=[],
    )


def make_text_region(rid, texts, t="text", bbox=None) -> TextRegionWithOcr:
    return TextRegionWithOcr(
        t=t,
        id=rid,
        bbox=bbox or make_bbox(),
        line_ocr=[make_line_ocr(i, text) for i, text in enumerate(texts)],
    )


def make_text_image_region(rid, texts, bbox=None) -> TextImageRegionWithOcr:
    return TextImageRegionWithOcr(
        t="text-image",
        id=rid,
        bbox=bbox or make_bbox(),
        line_ocr=[make_line_ocr(i, text) for i, text in enumerate(texts)],
    )


def make_image_region(rid, bbox=None) -> ImageRegion:
    return ImageRegion(t="image", id=rid, bbox=bbox or make_bbox())


def make_page(pid="p1") -> Page:
    return Page(
        id=pid,
        metadata=PageMetadata(height=2000, width=1000),
        url=f"https://example.com/{pid}.png",
    )


def make_input(regions, pid="p1") -> ArticleReconstructionInput:
    return ArticleReconstructionInput(page=make_page(pid), regions=regions)


class TestInputToFragments:
    def test_joins_line_texts_in_order(self):
        region = make_text_region("r_1", ["Hello", "World"])
        fragments = input_to_fragments(make_input([region]))
        assert fragments == [
            {
                "id": "r_1",
                "text": "Hello World",
                "type": "text",
                "hpos": 10,
                "vpos": 20,
                "width": 200,
                "height": 100,
            }
        ]

    def test_whitespace_normalized(self):
        region = make_text_region("r_1", ["Hello \n world", "  foo   bar  "])
        fragments = input_to_fragments(make_input([region]))
        assert fragments[0]["text"] == "Hello world foo bar"

    def test_image_region_skipped(self):
        regions = [make_image_region("r_img"), make_text_region("r_1", ["text"])]
        fragments = input_to_fragments(make_input(regions))
        assert [f["id"] for f in fragments] == ["r_1"]

    def test_empty_text_region_skipped(self):
        regions = [
            make_text_region("r_empty", ["", "   "]),
            make_text_region("r_1", ["text"]),
        ]
        fragments = input_to_fragments(make_input(regions))
        assert [f["id"] for f in fragments] == ["r_1"]

    def test_region_without_line_ocr_skipped(self):
        regions = [make_text_region("r_none", []), make_text_region("r_1", ["text"])]
        fragments = input_to_fragments(make_input(regions))
        assert [f["id"] for f in fragments] == ["r_1"]

    def test_bbox_min_max_derivation(self):
        bbox = BBox(
            x1=110.0, y1=30.0, x2=210.0, y2=30.0, x3=210.0, y3=130.0, x4=100.0, y4=140.0
        )
        region = make_text_region("r_1", ["text"], bbox=bbox)
        fragments = input_to_fragments(make_input([region]))
        assert fragments[0]["hpos"] == 100
        assert fragments[0]["vpos"] == 30
        assert fragments[0]["width"] == 110
        assert fragments[0]["height"] == 110

    def test_float_coords_truncated_to_int(self):
        bbox = BBox(
            x1=10.9, y1=20.4, x2=50.5, y2=20.4, x3=50.5, y3=80.7, x4=10.9, y4=80.7
        )
        region = make_text_region("r_1", ["text"], bbox=bbox)
        fragment = input_to_fragments(make_input([region]))[0]
        assert fragment["hpos"] == 10
        assert fragment["vpos"] == 20
        assert fragment["width"] == 39
        assert fragment["height"] == 60

    def test_type_mapping(self):
        regions = [
            make_text_region("r_1", ["a"], t="header"),
            make_text_region("r_2", ["b"], t="footer"),
            make_text_region("r_3", ["c"], t="headline"),
            make_text_image_region("r_4", ["d"]),
        ]
        fragments = input_to_fragments(make_input(regions))
        assert [f["type"] for f in fragments] == [
            "header",
            "footer",
            "headline",
            "text-image",
        ]

    def test_region_order_preserved(self):
        regions = [make_text_region("r_2", ["b"]), make_text_region("r_1", ["a"])]
        fragments = input_to_fragments(make_input(regions))
        assert [f["id"] for f in fragments] == ["r_2", "r_1"]

    def test_no_text_regions_yields_empty_list(self):
        regions = [make_image_region("r_img")]
        assert input_to_fragments(make_input(regions)) == []


class TestProcess:
    @staticmethod
    def make_prompt_file(tmp_path):
        prompt = tmp_path / "prompt.md"
        prompt.write_text(
            "# System Prompt\n\nsys prompt\n\n"
            "# User Prompt Template\n\nFragments:\n\n{fragments}\n"
        )
        return str(prompt)

    @staticmethod
    def make_module(client, prompt_file):
        from src.newspaper_reconstructor.module import (
            ArticleReconstructionConfig,
            ArticleReconstructionModule,
        )

        config = ArticleReconstructionConfig(
            prompt_file=prompt_file, model="test-model"
        )
        with patch(
            "src.newspaper_reconstructor.module.make_client", return_value=client
        ):
            return ArticleReconstructionModule(config=config)

    def test_returns_articles_with_sequential_ids(self, tmp_path):
        client = MagicMock()
        client.complete.return_value = (
            '[{"fragment_ids": ["r_1", "r_2"], "title": "t", "class": "article"},'
            ' {"fragment_ids": ["r_3"], "title": "u", "class": "advertisement"}]'
        )
        module = self.make_module(client, self.make_prompt_file(tmp_path))
        regions = [
            make_text_region("r_1", ["a"]),
            make_text_region("r_2", ["b"]),
            make_text_region("r_3", ["c"]),
        ]
        out = module.process(make_input(regions))
        assert out.articles == {"article_1": ["r_1", "r_2"], "article_2": ["r_3"]}

    def test_empty_llm_items_yield_empty_articles(self, tmp_path):
        client = MagicMock()
        client.complete.return_value = "[]"
        module = self.make_module(client, self.make_prompt_file(tmp_path))
        out = module.process(make_input([make_text_region("r_1", ["a"])]))
        assert out.articles == {}

    def test_page_without_text_regions_skips_llm(self, tmp_path):
        client = MagicMock()
        module = self.make_module(client, self.make_prompt_file(tmp_path))
        out = module.process(make_input([make_image_region("r_img")]))
        assert out.articles == {}
        client.complete.assert_not_called()

    def test_unknown_fragment_ids_filtered(self, tmp_path):
        client = MagicMock()
        client.complete.return_value = (
            '[{"fragment_ids": ["r_1", "r_unknown"], "title": "t", "class": "article"}]'
        )
        module = self.make_module(client, self.make_prompt_file(tmp_path))
        out = module.process(make_input([make_text_region("r_1", ["a"])]))
        assert out.articles == {"article_1": ["r_1"]}

    def test_llm_failure_raises_with_page_id(self, tmp_path):
        from openai import APIError

        client = MagicMock()
        client.complete.side_effect = APIError("boom", request=None, body=None)
        config = ArticleReconstructionConfig(
            prompt_file=self.make_prompt_file(tmp_path),
            model="test-model",
            max_retries=1,
        )
        with patch(
            "src.newspaper_reconstructor.module.make_client", return_value=client
        ):
            module = ArticleReconstructionModule(config=config)
        with pytest.raises(RuntimeError, match="p1"):
            module.process(make_input([make_text_region("r_1", ["a"])]))

    def test_article_id_prefix_config(self, tmp_path):
        client = MagicMock()
        client.complete.return_value = (
            '[{"fragment_ids": ["r_1"], "title": "t", "class": "article"}]'
        )
        config = ArticleReconstructionConfig(
            prompt_file=self.make_prompt_file(tmp_path),
            model="test-model",
            article_id_prefix="art_",
        )
        with patch(
            "src.newspaper_reconstructor.module.make_client", return_value=client
        ):
            module = ArticleReconstructionModule(config=config)
        out = module.process(make_input([make_text_region("r_1", ["a"])]))
        assert out.articles == {"art_1": ["r_1"]}

    def test_prompt_sent_to_client(self, tmp_path):
        client = MagicMock()
        client.complete.return_value = "[]"
        module = self.make_module(client, self.make_prompt_file(tmp_path))
        module.process(make_input([make_text_region("r_1", ["a"])]))
        args, _ = client.complete.call_args
        assert args[0] == "sys prompt"
        assert "Fragments:" in args[1]
        assert '"id": "r_1"' in args[1]


class TestModuleContract:
    def test_config_defaults(self):
        config = ArticleReconstructionConfig()
        assert config.model is None
        assert config.base_url is None
        assert config.api_key is None
        assert config.provider is None
        assert config.timeout == 300.0
        assert config.prompt_file == "prompts/v01.md"
        assert config.max_retries == 3
        assert config.max_workers == 1
        assert config.article_id_prefix == "article_"

    def test_input_rows_placeholder_empty(self, tmp_path):
        client = MagicMock()
        module = TestProcess.make_module(client, TestProcess.make_prompt_file(tmp_path))
        assert list(module._input_rows()) == []

    def test_resolve_input_type(self):
        from jawi_pipeline.cli import resolve_input_type

        assert (
            resolve_input_type(ArticleReconstructionModule)
            is ArticleReconstructionInput
        )


class TestBulkProcess:
    @staticmethod
    def make_config(prompt_file, **overrides):
        return ArticleReconstructionConfig(
            prompt_file=prompt_file, model="test-model", **overrides
        )

    @staticmethod
    def page_complete(system, user):
        import re
        import time

        match = re.search(r'"id": "(r_\w+)"', user)
        rid = match.group(1)
        if rid == "r_p1":
            time.sleep(0.15)
        return f'[{{"fragment_ids": ["{rid}"], "title": "t", "class": "article"}}]'

    def test_preserves_input_order_with_concurrency(self, tmp_path):
        client = MagicMock()
        client.complete.side_effect = self.page_complete
        prompt = TestProcess.make_prompt_file(tmp_path)
        with patch(
            "src.newspaper_reconstructor.module.make_client", return_value=client
        ):
            module = ArticleReconstructionModule(
                config=self.make_config(prompt, max_workers=3)
            )
        pages = [
            make_input([make_text_region("r_p1", ["a"])], pid="p1"),
            make_input([make_text_region("r_p2", ["b"])], pid="p2"),
            make_input([make_text_region("r_p3", ["c"])], pid="p3"),
        ]
        results = list(module.bulk_process(pages))
        assert [r.articles["article_1"] for r in results] == [
            ["r_p1"],
            ["r_p2"],
            ["r_p3"],
        ]

    def test_failed_page_yields_none_others_succeed(self, tmp_path, capsys):
        from openai import APIError

        client = MagicMock()

        def complete(system, user):
            if '"r_p2"' in user:
                raise APIError("boom", request=None, body=None)
            return '[{"fragment_ids": ["r_ok"], "title": "t", "class": "article"}]'

        client.complete.side_effect = complete
        prompt = TestProcess.make_prompt_file(tmp_path)
        with patch(
            "src.newspaper_reconstructor.module.make_client", return_value=client
        ):
            module = ArticleReconstructionModule(
                config=self.make_config(prompt, max_retries=1)
            )
        pages = [
            make_input([make_text_region("r_ok", ["a"])], pid="p1"),
            make_input([make_text_region("r_p2", ["b"])], pid="p2"),
            make_input([make_text_region("r_ok", ["c"])], pid="p3"),
        ]
        results = list(module.bulk_process(pages))
        assert results[0] is not None and results[0].articles == {"article_1": ["r_ok"]}
        assert results[1] is None
        assert results[2] is not None
        assert "p2" in capsys.readouterr().err

    def test_sequential_with_default_max_workers(self, tmp_path):
        client = MagicMock()
        client.complete.return_value = "[]"
        prompt = TestProcess.make_prompt_file(tmp_path)
        with patch(
            "src.newspaper_reconstructor.module.make_client", return_value=client
        ):
            module = ArticleReconstructionModule(config=self.make_config(prompt))
        pages = [
            make_input([make_text_region("r_1", ["a"])], pid="p1"),
            make_input([make_text_region("r_2", ["b"])], pid="p2"),
        ]
        results = list(module.bulk_process(pages))
        assert all(r is not None and r.articles == {} for r in results)

    def test_empty_input(self, tmp_path):
        client = MagicMock()
        prompt = TestProcess.make_prompt_file(tmp_path)
        with patch(
            "src.newspaper_reconstructor.module.make_client", return_value=client
        ):
            module = ArticleReconstructionModule(config=self.make_config(prompt))
        assert list(module.bulk_process([])) == []


class TestCli:
    @staticmethod
    def invoke(app, args):
        from typer.testing import CliRunner

        return CliRunner().invoke(app, args)

    @staticmethod
    def write_page(path, regions, pid):
        page = make_input(regions, pid=pid)
        path.write_text(page.model_dump_json(indent=2))

    @staticmethod
    def cli_config(prompt_file, **overrides):
        payload = {"model": "test-model", "prompt_file": str(prompt_file)}
        payload.update(overrides)
        return json.dumps(payload)

    def test_process_command_writes_output(self, tmp_path):
        prompt = TestProcess.make_prompt_file(tmp_path)
        input_file = tmp_path / "page.json"
        self.write_page(input_file, [make_text_region("r_1", ["a"])], "p1")
        output_file = tmp_path / "out.json"
        client = MagicMock()
        client.complete.return_value = (
            '[{"fragment_ids": ["r_1"], "title": "t", "class": "article"}]'
        )
        app = ArticleReconstructionModule.make_cli()
        with patch(
            "src.newspaper_reconstructor.module.make_client", return_value=client
        ):
            result = self.invoke(
                app,
                [
                    "process",
                    "--input",
                    str(input_file),
                    "--output",
                    str(output_file),
                    "--config",
                    self.cli_config(prompt),
                ],
            )
        assert result.exit_code == 0, result.output
        assert json.loads(output_file.read_text())["articles"] == {"article_1": ["r_1"]}

    def test_bulk_process_command_writes_outputs_and_checkpoint(self, tmp_path):
        prompt = TestProcess.make_prompt_file(tmp_path)
        input_dir = tmp_path / "pages"
        input_dir.mkdir()
        self.write_page(
            input_dir / "page1.json", [make_text_region("r_1", ["a"])], "p1"
        )
        self.write_page(
            input_dir / "page2.json", [make_text_region("r_2", ["b"])], "p2"
        )
        output_dir = tmp_path / "out"
        client = MagicMock()
        client.complete.return_value = "[]"
        app = ArticleReconstructionModule.make_cli()
        with patch(
            "src.newspaper_reconstructor.module.make_client", return_value=client
        ):
            result = self.invoke(
                app,
                [
                    "bulk-process",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output_dir),
                    "--config",
                    self.cli_config(prompt),
                ],
            )
        assert result.exit_code == 0, result.output
        assert json.loads((output_dir / "page1.json").read_text())["articles"] == {}
        assert json.loads((output_dir / "page2.json").read_text())["articles"] == {}
        checkpoint = json.loads((output_dir / ".checkpoint.json").read_text())
        assert checkpoint["completed_files"] == ["page1.json", "page2.json"]

    def test_missing_model_fails_fast(self, tmp_path, monkeypatch):
        for var in ("LLM_MODEL", "LLM_BASE_URL", "LLM_PROVIDER", "LLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        input_file = tmp_path / "page.json"
        self.write_page(input_file, [make_text_region("r_1", ["a"])], "p1")
        app = ArticleReconstructionModule.make_cli()
        result = self.invoke(
            app,
            [
                "process",
                "--input",
                str(input_file),
                "--output",
                str(tmp_path / "out.json"),
                "--config",
                "{}",
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)

    def test_help_lists_framework_commands(self):
        app = ArticleReconstructionModule.make_cli()
        result = self.invoke(app, ["--help"])
        assert "process" in result.output
        assert "bulk-process" in result.output
