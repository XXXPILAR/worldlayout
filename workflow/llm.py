"""One LLM request through the official OpenAI Responses API, with retries and a usage record.

Settings come from ``credentials.official_api`` (its own file, never the process environment) unless the caller
passes them (``credentials.api_settings``, e.g. the relay); a base URL whose host is not in
``credentials.ALLOWED_HOSTS`` is refused. ``WORKFLOW_RPM`` (with ``WORKFLOW_RPM_FILE``, a file shared by the worker
processes) keeps every request, retries included, under that many per rolling minute.

The relay (``credentials.RELAY_HOSTS``) answers its Responses endpoint with its own ~3.3k-token Codex system prompt
in front (it replaces the ``instructions`` field: the model never sees ours; checked 2026-10-28), but its chat
completions endpoint adds nothing (a 129-token request is billed 129 tokens). For a relay host the call therefore
goes to ``/chat/completions`` with the instructions as the system message, as ``fastfill.v2.ops.llm_probe`` (the
pure-GPT baseline) calls it; the official API keeps the Responses endpoint. A rejected key (401/403), an exhausted quota or a missing setting raises
``credentials.FatalAPIError``, which the planner and drafter do not catch, so a run stops instead of falling back
to rules. Transient failures (429 rate limit, 5xx, timeouts, an incomplete answer) are retried with backoff.
Usage is reported in chat-completions field names, the format the cost ledgers read, plus ``cost_usd``.

With ``WORKFLOW_LLM_LOG_DIR`` set, every call (answered or failed) is appended as one JSON line to
``<dir>/<WORKFLOW_LLM_TAG or "calls">.jsonl``: time, tag, model, instructions, prompt, the answer's text, usage, cost,
latency, failed attempts and the last problem. The key and the request headers are never written.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .credentials import ALLOWED_HOSTS, RELAY_HOSTS, FatalAPIError, cost_usd, official_api

USER_AGENT = "fastfill-workflow/1.0"
_LOG_LOCK = threading.Lock()


def _throttle() -> None:
    """Wait until one more request fits under WORKFLOW_RPM per rolling 60 s, counted across processes in
    WORKFLOW_RPM_FILE (a JSON list of request times, guarded by an exclusive file lock)."""
    rpm, state = os.environ.get("WORKFLOW_RPM"), os.environ.get("WORKFLOW_RPM_FILE")
    if not rpm or not state:
        return
    import fcntl
    limit = int(rpm)
    while True:
        with open(state, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            try:
                stamps = [t for t in json.loads(f.read() or "[]") if time.time() - t < 60.]
            except ValueError:
                stamps = []
            if len(stamps) < limit:
                stamps.append(time.time())
                f.seek(0)
                f.truncate()
                f.write(json.dumps(stamps))
                return
            wait = 60. - (time.time() - min(stamps)) + .05
        time.sleep(max(wait, .05))


def _log(record: dict) -> None:
    folder = os.environ.get("WORKFLOW_LLM_LOG_DIR")
    if not folder:
        return
    tag = os.environ.get("WORKFLOW_LLM_TAG") or "calls"
    path = Path(folder) / f"{tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, **record}, ensure_ascii=False)
    with _LOG_LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _usage(raw: dict | None) -> dict:
    raw = raw or {}
    return {"prompt_tokens": raw.get("input_tokens"), "completion_tokens": raw.get("output_tokens"),
            "total_tokens": raw.get("total_tokens"),
            "prompt_tokens_details": {"cached_tokens": (raw.get("input_tokens_details") or {}).get("cached_tokens")},
            "completion_tokens_details": {"reasoning_tokens": (raw.get("output_tokens_details") or {}).get("reasoning_tokens")}}


def read_chat(response) -> tuple[str, str | None, dict | None, object]:
    """(text, finish_reason, usage, error) of a chat completions answer, streamed (server-sent events) or not. The
    relay's gateway cuts a request off after 60 s without a byte (HTTP 504; 2026-10-29), so a large room's answer,
    which takes minutes, only arrives streamed: the model and what it writes are the same, only the transport differs.
    A busy relay may answer with one JSON error object instead of events."""
    kind = (response.headers.get("Content-Type") or "") if hasattr(response, "headers") else ""
    if "event-stream" not in kind:
        answer = json.loads(response.read())
        choice = (answer.get("choices") or [{}])[0]
        return ((choice.get("message") or {}).get("content") or "", choice.get("finish_reason"),
                answer.get("usage"), answer.get("error") if not answer.get("choices") else None)
    parts, finish, usage, error = [], None, None, None
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("error") or event.get("type") == "error":
            error = event.get("error") or event
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            parts.append((choice.get("delta") or {}).get("content") or "")
            finish = choice.get("finish_reason") or finish
    return "".join(parts), finish, usage, error


def _chat_usage(raw: dict | None) -> dict:
    raw = raw or {}
    return {"prompt_tokens": raw.get("prompt_tokens"), "completion_tokens": raw.get("completion_tokens"),
            "total_tokens": raw.get("total_tokens"),
            "prompt_tokens_details": {"cached_tokens": (raw.get("prompt_tokens_details") or {}).get("cached_tokens")},
            "completion_tokens_details": {"reasoning_tokens": (raw.get("completion_tokens_details") or {}).get("reasoning_tokens")}}


def _text(answer: dict) -> str:
    return "".join(part.get("text", "") for item in answer.get("output") or [] if item.get("type") == "message"
                   for part in item.get("content") or [] if part.get("type") == "output_text")


def ask(instructions: str, prompt: str, *, model: str | None = None, base_url: str | None = None,
        api_key: str | None = None, reasoning_effort: str | None = None,
        attempts: int | None = None, timeout_s: float = 600., images: list[str] | None = None) -> tuple[str, dict]:
    """Returns (text, call) with call = {model, usage, cost_usd, latency_s, failed_attempts}.

    ``images``: PNG data URLs, sent as image content; logs retain only their count and SHA256 fingerprints.
    ``attempts``: requests tried before giving up on transient problems (busy / rate limited / 5xx / dropped
    connection / an answer without text); default ``WORKFLOW_LLM_ATTEMPTS`` (0 = until it succeeds), else 6. A
    rejected key, missing access, an unknown model or an exhausted quota raise ``FatalAPIError`` at once, whatever
    ``attempts`` says. Retries wait 5, 10, 20, ... s, at most 120 s, and every fifth one is printed."""
    images = [] if images is None else images
    if not isinstance(images, list):
        raise ValueError("images must be a list of PNG data URLs")
    hashes = []
    for image in images:
        if not isinstance(image, str) or not image.startswith("data:image/png;base64,"):
            raise ValueError("images must contain PNG data URLs only")
        try:
            raw = base64.b64decode(image.split(",", 1)[1], validate=True)
        except ValueError:
            raise ValueError("image data is not valid base64") from None
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("image data is not PNG")
        hashes.append(hashlib.sha256(raw).hexdigest())
    image_info = {"image_count": len(images), "image_sha256": hashes}
    if attempts is None:
        attempts = int(os.environ.get("WORKFLOW_LLM_ATTEMPTS", "6"))
    if not (model and base_url and api_key):
        settings = official_api(model=model)
        model, base_url, api_key = model or settings["model"], base_url or settings["base_url"], \
            api_key or settings["api_key"]
    if urlparse(base_url).hostname not in ALLOWED_HOSTS:
        raise FatalAPIError("the workflow runs on the official OpenAI API or the configured relay only")
    effort = reasoning_effort or os.environ.get("WORKFLOW_REASONING_EFFORT", "medium")
    relay = urlparse(base_url).hostname in RELAY_HOSTS
    if relay:  # chat completions: no hidden system prompt on the relay
        content = ([{"type": "text", "text": prompt}] +
                   [{"type": "image_url", "image_url": {"url": image}} for image in images]) if images else prompt
        body = {"model": model, "reasoning_effort": effort, "stream": True, "stream_options": {"include_usage": True},
                "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": content}]}
        url = base_url.rstrip("/") + "/chat/completions"
    else:
        content = ([{"type": "input_text", "text": prompt}] +
                   [{"type": "input_image", "image_url": image} for image in images]) if images else prompt
        body = {"model": model, "instructions": instructions, "store": False,
                "input": [{"role": "user", "content": content}], "reasoning": {"effort": effort}}
        url = base_url.rstrip("/") + "/responses"
    data = json.dumps(body).encode("utf-8")
    start, problem, attempt = time.perf_counter(), None, -1
    while True:
        attempt += 1
        if attempts and attempt >= attempts:
            break
        if attempt and attempt % 5 == 0:
            print(f"LLM retry {attempt}: {str(problem)[:160]}", flush=True)
        try:
            _throttle()
            request = urllib.request.Request(url, data, {"Authorization": f"Bearer {api_key}",
                                                         "Content-Type": "application/json", "User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                if relay:  # streamed: see read_chat
                    text, finish, raw_usage, error = read_chat(response)
                    answer = {"error": error} if error else {"incomplete_details": f"finish_reason {finish}"}
                else:
                    answer = json.load(response)
            if relay:
                usage = _chat_usage(raw_usage)
                done = finish == "stop" and not error
            else:
                text = _text(answer)
                usage = _usage(answer.get("usage"))
                done = answer.get("status") == "completed"
            if done and text:
                call = {"model": model, "reasoning_effort": effort, "host": urlparse(base_url).hostname, "usage": usage,
                        "cost_usd": cost_usd(usage, model),
                        "latency_s": round(time.perf_counter() - start, 3), "failed_attempts": attempt, **image_info}
                _log({**call, "instructions": instructions, "prompt": prompt, "response": text, "ok": True})
                return text, call
            problem = answer.get("error") or answer.get("incomplete_details") or "empty response"
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300]
            problem = f"HTTP {exc.code}: {detail!r}"
            if exc.code in (401, 403, 404) or b"insufficient_quota" in detail:  # key, access, unknown model, quota
                _log({"model": model, "reasoning_effort": effort, "instructions": instructions,
                      "prompt": prompt, "ok": False, **image_info,
                      "failed_attempts": attempt + 1, "problem": str(problem)[:300]})
                raise FatalAPIError(f"OpenAI rejected the request: HTTP {exc.code} {detail[:160]!r}") from None
            if exc.code == 400 and not any(w in detail.lower() for w in (b"busy", b"overload", b"upstream", b"try again")):
                break  # this request is malformed (e.g. too long): retrying will not help
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, http.client.HTTPException) as exc:
            # HTTPException: the relay dropped the connection while the answer was being read (IncompleteRead)
            problem = f"{type(exc).__name__}: {exc}"
        time.sleep(min(120., 5. * 2 ** min(attempt, 5)))
    _log({"model": model, "reasoning_effort": effort, "instructions": instructions,
          "prompt": prompt, "ok": False, **image_info,
          "latency_s": round(time.perf_counter() - start, 3), "problem": str(problem)[:300]})
    raise RuntimeError(f"LLM request failed: {str(problem)[:300]}")
