"""Downloads Adaptea performs itself, and the cancellation that was the point of them."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from adaptea.downloads import (
    DownloadError,
    DownloadProgress,
    RepositoryFile,
    download_repository,
    lmstudio_models_root,
    parse_repository,
    pull_ollama_model,
    select_files,
)


class RecordingStream(httpx.AsyncByteStream):
    """A response body that reports whether the client hung up before it finished."""

    def __init__(self, chunks: list[bytes], *, block_after: int | None = None) -> None:
        self.chunks = chunks
        self.block_after = block_after
        self.closed = False
        self.delivered = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            for index, chunk in enumerate(self.chunks):
                if self.block_after is not None and index == self.block_after:
                    # Stand in for a transfer still in flight: the only way out is the
                    # caller closing the connection.
                    await asyncio.Event().wait()
                self.delivered += 1
                yield chunk
        finally:
            self.closed = True

    async def aclose(self) -> None:
        self.closed = True


def ndjson(*events: dict[str, object]) -> list[bytes]:
    return [json.dumps(event).encode() + b"\n" for event in events]


@pytest.mark.asyncio
async def test_ollama_pull_reports_progress_across_every_layer() -> None:
    stream = RecordingStream(
        ndjson(
            {"status": "pulling manifest"},
            {"status": "downloading", "digest": "a", "total": 100, "completed": 50},
            {"status": "downloading", "digest": "b", "total": 100, "completed": 25},
            {"status": "downloading", "digest": "a", "total": 100, "completed": 100},
            {"status": "success"},
        )
    )
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
    seen: list[DownloadProgress] = []

    async with httpx.AsyncClient(transport=transport, base_url="http://ollama") as client:
        await pull_ollama_model("coder:7b", on_progress=seen.append, client=client)

    # Ollama reports one blob at a time; the user is downloading a model, so the total is
    # the sum of the blobs rather than whichever one happens to be moving.
    assert seen[-2].completed == 125
    assert seen[-2].total == 200
    assert seen[-1].detail == "success"


@pytest.mark.asyncio
async def test_ollama_pull_surfaces_the_daemon_error_rather_than_finishing_quietly() -> None:
    stream = RecordingStream(ndjson({"status": "pulling manifest"}, {"error": "model not found"}))
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))

    async with httpx.AsyncClient(transport=transport, base_url="http://ollama") as client:
        with pytest.raises(DownloadError, match="model not found"):
            await pull_ollama_model("nope", on_progress=None, client=client)


@pytest.mark.asyncio
async def test_cancelling_an_ollama_pull_closes_the_connection_that_drives_it() -> None:
    """Closing the request is how the daemon learns to stop; nothing else reaches it."""
    stream = RecordingStream(
        ndjson(
            {"status": "downloading", "digest": "a", "total": 100, "completed": 10},
            {"status": "downloading", "digest": "a", "total": 100, "completed": 20},
        ),
        block_after=1,
    )
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
    first = asyncio.Event()

    async with httpx.AsyncClient(transport=transport, base_url="http://ollama") as client:
        pull = asyncio.create_task(
            pull_ollama_model("coder:7b", on_progress=lambda _progress: first.set(), client=client)
        )
        await asyncio.wait_for(first.wait(), timeout=5)
        pull.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pull

    assert stream.closed


def hub_transport(
    files: list[dict[str, object]], bodies: dict[str, bytes], seen: list[httpx.Request]
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "/tree/" in request.url.path:
            return httpx.Response(200, json=files)
        name = request.url.path.rsplit("/", 1)[-1]
        body = bodies[name]
        span = request.headers.get("Range")
        if span:
            start = int(span.removeprefix("bytes=").rstrip("-"))
            return httpx.Response(206, content=body[start:])
        return httpx.Response(200, content=body)

    return httpx.MockTransport(handle)


@pytest.mark.asyncio
async def test_download_repository_mirrors_the_repository_into_lmstudios_layout(
    tmp_path: Path,
) -> None:
    seen: list[httpx.Request] = []
    transport = hub_transport(
        [
            {"type": "file", "path": "model.safetensors", "size": 5},
            {"type": "file", "path": "config.json", "size": 2},
            {"type": "directory", "path": "nested"},
        ],
        {"model.safetensors": b"12345", "config.json": b"{}"},
        seen,
    )

    async with httpx.AsyncClient(transport=transport) as client:
        target = await download_repository(
            "publisher/coder-MLX-4bit",
            tmp_path,
            client=client,
            endpoint="https://hub",
        )

    # `<root>/<owner>/<repository>` is the layout LM Studio indexes, so no import step
    # is needed for the model to show up in `lms ls`.
    assert target == tmp_path / "publisher" / "coder-MLX-4bit"
    assert (target / "model.safetensors").read_bytes() == b"12345"
    assert (target / "config.json").read_bytes() == b"{}"
    assert not list(target.glob("*.part"))


@pytest.mark.asyncio
async def test_cancelling_a_repository_download_leaves_a_resumable_part_and_no_model(
    tmp_path: Path,
) -> None:
    body = RecordingStream([b"aaaa", b"bbbb"], block_after=1)

    def handle(request: httpx.Request) -> httpx.Response:
        if "/tree/" in request.url.path:
            return httpx.Response(200, json=[{"type": "file", "path": "model.gguf", "size": 8}])
        return httpx.Response(200, stream=body)

    started = asyncio.Event()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        download = asyncio.create_task(
            download_repository(
                "publisher/coder",
                tmp_path,
                on_progress=lambda progress: started.set() if progress.completed else None,
                client=client,
                endpoint="https://hub",
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        download.cancel()
        with pytest.raises(asyncio.CancelledError):
            await download

    target = tmp_path / "publisher" / "coder"
    # A half-written file must never be mistaken for a model, and must not be thrown away
    # either: the next attempt continues from it.
    assert not (target / "model.gguf").exists()
    assert (target / "model.gguf.part").read_bytes() == b"aaaa"


@pytest.mark.asyncio
async def test_a_resumed_download_asks_only_for_the_bytes_it_is_missing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "publisher" / "coder"
    target.mkdir(parents=True)
    (target / "model.gguf.part").write_bytes(b"aaaa")
    seen: list[httpx.Request] = []
    transport = hub_transport(
        [{"type": "file", "path": "model.gguf", "size": 8}], {"model.gguf": b"aaaabbbb"}, seen
    )

    async with httpx.AsyncClient(transport=transport) as client:
        await download_repository(
            "publisher/coder", tmp_path, client=client, endpoint="https://hub"
        )

    assert seen[-1].headers["Range"] == "bytes=4-"
    assert (target / "model.gguf").read_bytes() == b"aaaabbbb"


def test_a_multi_quantization_repository_asks_which_one_rather_than_fetching_all() -> None:
    files = [
        RepositoryFile("coder-Q4_K_M.gguf", 10),
        RepositoryFile("coder-Q8_0.gguf", 20),
    ]
    with pytest.raises(DownloadError, match="Q4_K_M, Q8_0"):
        select_files(files, None)
    assert select_files(files, "Q8_0") == [files[1]]


def test_weight_sets_are_taken_whole_because_a_partial_one_does_not_load() -> None:
    files = [
        RepositoryFile("model.safetensors", 10),
        RepositoryFile("config.json", 1),
        RepositoryFile(".gitattributes", 1),
    ]
    assert select_files(files, None) == files[:2]


def test_repository_names_are_validated_before_anything_is_fetched() -> None:
    assert parse_repository("publisher/coder-GGUF:Q4_K_M") == ("publisher/coder-GGUF", "Q4_K_M")
    with pytest.raises(ValueError, match="owner/repository"):
        parse_repository("qwen2.5-coder-14b")


def test_lmstudio_models_root_follows_the_folder_lm_studio_was_told_to_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADAPTEA_LMSTUDIO_MODELS_DIR", str(tmp_path / "elsewhere"))
    assert lmstudio_models_root() == tmp_path / "elsewhere"
