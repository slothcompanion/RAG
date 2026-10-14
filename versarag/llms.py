"""LLM backends. Every backend takes OpenAI-style chat messages
(``[{"role": "system"|"user"|"assistant", "content": ...}]``) and lets the
model's own chat template do the formatting, so Llama, Qwen, Mistral, Gemma,
Phi, DeepSeek, etc. all work without prompt-format changes.

Backends:

* ``ollama``            – Ollama native API (exposes ``num_ctx`` and ``think``)
* ``openai_compatible`` – vLLM, llama.cpp server, LM Studio, SGLang, TGI,
                          LocalAI, Together/Groq/OpenRouter, ...
* ``huggingface``       – transformers, in-process (optional 4/8-bit)
* ``llama_cpp``         – llama-cpp-python, in-process GGUF
* ``echo``              – no model; for tests and dry runs
"""

from __future__ import annotations

import json
import os
import re
import threading
from abc import ABC, abstractmethod
from typing import Any, Iterator

import httpx

from .registry import register

Messages = list[dict[str, str]]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks (Qwen3, QwQ, DeepSeek-R1...)."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # template pre-opened <think> in the prompt
        text = text.split("</think>", 1)[1]
    return text.strip()


class ThinkingFilter:
    """Streaming counterpart of :func:`strip_thinking`; handles tags split across chunks."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self, starts_inside: bool = False) -> None:
        self.buf = ""
        self.inside = starts_inside
        self.lstrip = starts_inside
        self.started = False

    @staticmethod
    def _partial_suffix(buf: str, tag: str) -> int:
        for n in range(min(len(tag) - 1, len(buf)), 0, -1):
            if buf.endswith(tag[:n]):
                return n
        return 0

    def feed(self, text: str) -> str:
        self.buf += text
        out: list[str] = []
        while True:
            if self.inside:
                i = self.buf.find(self.CLOSE)
                if i == -1:
                    keep = self._partial_suffix(self.buf, self.CLOSE)
                    self.buf = self.buf[len(self.buf) - keep :]
                    break
                self.buf = self.buf[i + len(self.CLOSE) :]
                self.inside, self.lstrip = False, True
            else:
                i = self.buf.find(self.OPEN)
                if i == -1:
                    keep = self._partial_suffix(self.buf, self.OPEN)
                    out.append(self.buf[: len(self.buf) - keep])
                    self.buf = self.buf[len(self.buf) - keep :]
                    break
                out.append(self.buf[:i])
                self.buf = self.buf[i + len(self.OPEN) :]
                self.inside = True
        return self._emit("".join(out))

    def _emit(self, s: str) -> str:
        if self.lstrip or not self.started:
            s = s.lstrip()
            if s:
                self.lstrip, self.started = False, True
        return s

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return self._emit(rest)


class LLM(ABC):
    strip_thinking: bool = True
    thinking_prefilled: bool = False

    @abstractmethod
    def _generate(self, messages: Messages, **kwargs: Any) -> str: ...

    @abstractmethod
    def _stream(self, messages: Messages, **kwargs: Any) -> Iterator[str]: ...

    def generate(self, messages: Messages, **kwargs: Any) -> str:
        text = self._generate(messages, **kwargs)
        return strip_thinking(text) if self.strip_thinking else text

    def stream(self, messages: Messages, **kwargs: Any) -> Iterator[str]:
        if not self.strip_thinking:
            yield from self._stream(messages, **kwargs)
            return
        f = ThinkingFilter(starts_inside=self.thinking_prefilled)
        for piece in self._stream(messages, **kwargs):
            out = f.feed(piece)
            if out:
                yield out
        tail = f.flush()
        if tail:
            yield tail


@register("llm", "openai_compatible")
class OpenAICompatibleLLM(LLM):
    """Any ``/v1/chat/completions`` server.

    Examples: vLLM (``vllm serve Qwen/Qwen3-8B`` -> ``http://localhost:8000/v1``),
    llama.cpp (``llama-server -m model.gguf`` -> ``http://localhost:8080/v1``),
    LM Studio (``http://localhost:1234/v1``), Ollama (``http://localhost:11434/v1``).

    ``extra_body`` is merged into the request — e.g. for Qwen3 on vLLM/SGLang:
    ``{"chat_template_kwargs": {"enable_thinking": false}}``.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
        top_p: float | None = None,
        extra_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 300.0,
        strip_thinking: bool = True,
        thinking_prefilled: bool = False,
    ) -> None:
        self.model = model
        self.params = {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
        self.extra_body = extra_body or {}
        self.strip_thinking = strip_thinking
        self.thinking_prefilled = thinking_prefilled
        key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {key}", **(headers or {})},
            timeout=timeout,
        )

    def _payload(self, messages: Messages, stream: bool, **kwargs: Any) -> dict[str, Any]:
        params = {k: v for k, v in {**self.params, **kwargs}.items() if v is not None}
        return {"model": self.model, "messages": messages, "stream": stream, **params, **self.extra_body}

    def _generate(self, messages: Messages, **kwargs: Any) -> str:
        r = self.client.post("/chat/completions", json=self._payload(messages, False, **kwargs))
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""

    def _stream(self, messages: Messages, **kwargs: Any) -> Iterator[str]:
        with self.client.stream("POST", "/chat/completions", json=self._payload(messages, True, **kwargs)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or []
                if choices:
                    piece = (choices[0].get("delta") or {}).get("content")
                    if piece:
                        yield piece


@register("llm", "ollama")
class OllamaLLM(LLM):
    """Ollama's native ``/api/chat``.

    Prefer this over Ollama's OpenAI endpoint for RAG: it lets you raise
    ``num_ctx`` (Ollama's default context is small and silently truncates
    long prompts) and toggle ``think`` for reasoning models like Qwen3.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        num_ctx: int | None = 8192,
        num_predict: int | None = None,
        top_p: float | None = None,
        think: bool | None = None,
        keep_alive: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: float = 300.0,
        strip_thinking: bool = True,
    ) -> None:
        self.model = model
        opts = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict, "top_p": top_p, **(options or {})}
        self.options = {k: v for k, v in opts.items() if v is not None}
        self.think = think
        self.keep_alive = keep_alive
        self.strip_thinking = strip_thinking
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _payload(self, messages: Messages, stream: bool, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream, "options": {**self.options, **kwargs}}
        if self.think is not None:
            body["think"] = self.think
        if self.keep_alive is not None:
            body["keep_alive"] = self.keep_alive
        return body

    def _generate(self, messages: Messages, **kwargs: Any) -> str:
        r = self.client.post("/api/chat", json=self._payload(messages, False, **kwargs))
        r.raise_for_status()
        return r.json()["message"].get("content") or ""

    def _stream(self, messages: Messages, **kwargs: Any) -> Iterator[str]:
        with self.client.stream("POST", "/api/chat", json=self._payload(messages, True, **kwargs)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise RuntimeError(f"Ollama error: {obj['error']}")
                piece = (obj.get("message") or {}).get("content")
                if piece:
                    yield piece
                if obj.get("done"):
                    break


@register("llm", "huggingface")
class HuggingFaceLLM(LLM):
    """Run any HF chat model in-process with ``transformers``.

    e.g. ``meta-llama/Llama-3.1-8B-Instruct`` (gated: set ``HF_TOKEN``),
    ``Qwen/Qwen3-8B`` (pass ``chat_template_kwargs: {enable_thinking: false}``),
    ``Qwen/Qwen2.5-7B-Instruct``.
    """

    def __init__(
        self,
        model: str,
        device_map: str | dict[str, Any] | None = "auto",
        dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float | None = None,
        trust_remote_code: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        strip_thinking: bool = True,
        thinking_prefilled: bool = False,
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("huggingface LLM needs `pip install versarag[transformers]`") from e
        self.torch = torch
        self.strip_thinking = strip_thinking
        self.thinking_prefilled = thinking_prefilled
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        kwargs: dict[str, Any] = {"device_map": device_map, "trust_remote_code": trust_remote_code, **(model_kwargs or {})}
        torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)
        # transformers >= 4.56 renamed torch_dtype -> dtype
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kwargs["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = torch_dtype
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if load_in_4bit else None,
                bnb_4bit_quant_type="nf4" if load_in_4bit else "fp4",
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.model.eval()

    def _prepare(self, messages: Messages, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, **self.chat_template_kwargs
        ).to(self.model.device)
        gen = {k: v for k, v in {**self.gen_kwargs, **kwargs}.items() if v is not None}
        if "max_tokens" in gen:
            gen["max_new_tokens"] = gen.pop("max_tokens")
        if gen.get("temperature", 0) <= 0:
            gen.pop("temperature", None)
            gen.pop("top_p", None)
            gen["do_sample"] = False
        else:
            gen["do_sample"] = True
        gen.setdefault("pad_token_id", self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return dict(inputs), gen

    def _generate(self, messages: Messages, **kwargs: Any) -> str:
        inputs, gen = self._prepare(messages, **kwargs)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **gen)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _stream(self, messages: Messages, **kwargs: Any) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        inputs, gen = self._prepare(messages, **kwargs)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def run() -> None:
            with self.torch.inference_mode():
                self.model.generate(**inputs, **gen, streamer=streamer)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        yield from streamer
        thread.join()


@register("llm", "llama_cpp")
class LlamaCppLLM(LLM):
    """GGUF models in-process via ``llama-cpp-python`` (CPU, CUDA, Metal).

    Use ``model_path`` for a local file, or ``repo_id`` + ``filename`` (glob ok)
    to download from the HF Hub, e.g. ``Qwen/Qwen3-8B-GGUF`` / ``*Q4_K_M.gguf``.
    """

    def __init__(
        self,
        model_path: str | None = None,
        repo_id: str | None = None,
        filename: str | None = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float | None = None,
        chat_format: str | None = None,
        verbose: bool = False,
        strip_thinking: bool = True,
        thinking_prefilled: bool = False,
        **llama_kwargs: Any,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover
            raise ImportError("llama_cpp LLM needs `pip install llama-cpp-python`") from e
        common = {"n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers, "chat_format": chat_format, "verbose": verbose, **llama_kwargs}
        if model_path:
            self.llm = Llama(model_path=model_path, **common)
        elif repo_id and filename:
            self.llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, **common)
        else:
            raise ValueError("llama_cpp needs `model_path` or `repo_id` + `filename`")
        self.params = {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
        self.strip_thinking = strip_thinking
        self.thinking_prefilled = thinking_prefilled

    def _params(self, **kwargs: Any) -> dict[str, Any]:
        return {k: v for k, v in {**self.params, **kwargs}.items() if v is not None}

    def _generate(self, messages: Messages, **kwargs: Any) -> str:
        out = self.llm.create_chat_completion(messages=messages, **self._params(**kwargs))
        return out["choices"][0]["message"].get("content") or ""

    def _stream(self, messages: Messages, **kwargs: Any) -> Iterator[str]:
        for part in self.llm.create_chat_completion(messages=messages, stream=True, **self._params(**kwargs)):
            piece = part["choices"][0]["delta"].get("content")
            if piece:
                yield piece


@register("llm", "echo")
class EchoLLM(LLM):
    """Returns the retrieved context instead of calling a model (tests / dry runs)."""

    def __init__(self, max_chars: int = 600, **_: Any) -> None:
        self.max_chars = max_chars
        self.strip_thinking = False

    def _generate(self, messages: Messages, **kwargs: Any) -> str:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return "[echo] " + last[: self.max_chars]

    def _stream(self, messages: Messages, **kwargs: Any) -> Iterator[str]:
        yield from re.findall(r"\S+\s*", self._generate(messages))
