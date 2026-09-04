"""Tests for the LLM client factory and the local transformers backend."""

import importlib.util
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.newspaper_reconstructor import llm
from src.newspaper_reconstructor.llm import (
    CompletionClient,
    LLMClient,
    LLMError,
    LocalLLMClient,
    make_client,
)

TRANSFORMERS_INSTALLED = importlib.util.find_spec("transformers") is not None


class FakeTensor:
    def __init__(self, last_dim: int = 25):
        self._shape = (1, last_dim)

    @property
    def shape(self):
        return self._shape


class FakeInputs(dict):
    def __init__(self, input_len: int = 25):
        super().__init__(input_ids=FakeTensor(input_len))
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeGeneration:
    """Supports generation[0][input_len:] and records the slice bounds."""

    def __init__(self):
        self.first_index = None
        self.slice_start = None

    def __getitem__(self, index):
        if self.first_index is None:
            self.first_index = index
            return self
        self.slice_start = index.start
        return ["token"]


class FakeProcessor:
    def __init__(self, output='[{"fragment_ids": ["r1"]}]', input_len=25):
        self.output = output
        self.inputs = FakeInputs(input_len)
        self.template_calls = []
        self.decode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return self.inputs

    def decode(self, tokens, **kwargs):
        self.decode_calls.append((tokens, kwargs))
        return self.output


class FakeModel:
    pass


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    for var in (
        "LLM_BACKEND",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_PROVIDER",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def local_stack(monkeypatch):
    """Patch the torch/transformers seams and record every interaction."""
    recorded = {}

    def fake_load(model_name, device):
        recorded["loaded"] = (model_name, device)
        recorded["model"] = FakeModel()
        recorded["processor"] = FakeProcessor()
        return recorded["model"], recorded["processor"]

    def fake_generate(model, inputs, gen_kwargs):
        recorded["generated"] = (model, inputs, gen_kwargs)
        generation = FakeGeneration()
        recorded["generation"] = generation
        return generation

    monkeypatch.setattr(llm, "_detect_device", lambda: "cpu")
    monkeypatch.setattr(llm, "_load_local_model", fake_load)
    monkeypatch.setattr(llm, "_generate_local", fake_generate)
    return recorded


class TestMakeClientDispatch:
    def test_default_backend_is_api_client(self):
        client = make_client(model="gpt-test", api_key="none")
        assert isinstance(client, LLMClient)

    def test_local_backend_returns_local_client(self, local_stack):
        client = make_client(backend="local", model="test-model")
        assert isinstance(client, LocalLLMClient)
        assert local_stack["loaded"] == ("test-model", "cpu")

    def test_backend_env_var_fallback(self, local_stack, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "Local")
        client = make_client(model="test-model")
        assert isinstance(client, LocalLLMClient)

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="api, local"):
            make_client(model="m", backend="banana")

    def test_local_backend_ignores_provider_and_base_url(self, local_stack):
        client = make_client(
            model="m",
            backend="local",
            provider="no-such-provider",
            base_url="http://localhost:1",
        )
        assert isinstance(client, LocalLLMClient)

    def test_local_backend_requires_model(self, local_stack):
        with pytest.raises(ValueError, match="Model not set"):
            make_client(backend="local")


class TestCompletionClientContract:
    def test_both_backends_satisfy_completion_client(self, local_stack):
        api_client = make_client(model="m", api_key="none")
        local_client = make_client(backend="local", model="m")
        assert isinstance(api_client, CompletionClient)
        assert isinstance(local_client, CompletionClient)


class TestLocalLLMClient:
    def test_complete_builds_messages_and_template_flags(self, local_stack):
        client = make_client(backend="local", model="m")
        output = client.complete("SYS", "USER")
        messages, kwargs = local_stack["processor"].template_calls[0]
        assert messages == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USER"},
        ]
        assert kwargs == {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        assert local_stack["processor"].inputs.moved_to == "cpu"
        assert output == '[{"fragment_ids": ["r1"]}]'

    def test_default_generation_kwargs(self, local_stack):
        client = make_client(backend="local", model="m")
        client.complete("s", "u")
        _, _, gen_kwargs = local_stack["generated"]
        assert gen_kwargs == {"max_new_tokens": 2048, "do_sample": False}

    def test_model_kwargs_merge_with_defaults(self, local_stack):
        client = make_client(
            backend="local", model="m", model_kwargs={"max_new_tokens": 8}
        )
        client.complete("s", "u")
        _, _, gen_kwargs = local_stack["generated"]
        assert gen_kwargs == {"max_new_tokens": 8, "do_sample": False}

    def test_max_tokens_remapped_to_max_new_tokens(self, local_stack):
        client = make_client(
            backend="local", model="m", model_kwargs={"max_tokens": 128}
        )
        client.complete("s", "u")
        _, _, gen_kwargs = local_stack["generated"]
        assert gen_kwargs == {"max_new_tokens": 128, "do_sample": False}

    def test_complete_slices_input_tokens_and_decodes(self, local_stack):
        client = make_client(backend="local", model="m")
        output = client.complete("s", "u")
        generation = local_stack["generation"]
        assert generation.first_index == 0
        assert generation.slice_start == 25
        tokens, decode_kwargs = local_stack["processor"].decode_calls[0]
        assert tokens == ["token"]
        assert decode_kwargs == {"skip_special_tokens": True}
        assert output == '[{"fragment_ids": ["r1"]}]'

    def test_generate_failure_wrapped_in_llm_error(self, local_stack, monkeypatch):
        def failing_generate(model, inputs, gen_kwargs):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(llm, "_generate_local", failing_generate)
        client = make_client(backend="local", model="m")
        with pytest.raises(LLMError, match="'m'") as exc_info:
            client.complete("s", "u")
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_load_failure_wrapped_in_llm_error(self, monkeypatch):
        monkeypatch.setattr(llm, "_detect_device", lambda: "cpu")

        def failing_load(model_name, device):
            raise OSError("connection error")

        monkeypatch.setattr(llm, "_load_local_model", failing_load)
        with pytest.raises(LLMError, match="'m'") as exc_info:
            make_client(backend="local", model="m")
        assert isinstance(exc_info.value.__cause__, OSError)

    def test_llm_error_from_loader_not_double_wrapped(self, monkeypatch):
        monkeypatch.setattr(llm, "_detect_device", lambda: "cpu")
        original = LLMError("torch missing")

        def failing_load(model_name, device):
            raise original

        monkeypatch.setattr(llm, "_load_local_model", failing_load)
        with pytest.raises(LLMError) as exc_info:
            make_client(backend="local", model="m")
        assert exc_info.value is original

    @pytest.mark.skipif(
        TRANSFORMERS_INSTALLED, reason="requires torch/transformers to be absent"
    )
    def test_missing_deps_raises_llm_error_with_remedy(self):
        with pytest.raises(LLMError, match="uv sync --group local"):
            make_client(backend="local", model="m")

    def test_generation_serialized_across_threads(self, local_stack, monkeypatch):
        intervals = []
        interval_lock = threading.Lock()

        def slow_generate(model, inputs, gen_kwargs):
            start = time.monotonic()
            time.sleep(0.05)
            with interval_lock:
                intervals.append((start, time.monotonic()))
            return FakeGeneration()

        monkeypatch.setattr(llm, "_generate_local", slow_generate)
        client = make_client(backend="local", model="m")

        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(lambda _: client.complete("s", "u"), range(3)))

        intervals.sort()
        assert len(intervals) == 3
        for (_, previous_end), (next_start, _) in itertools.pairwise(intervals):
            assert next_start >= previous_end
