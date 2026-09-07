"""Model downloads Adaptea performs itself.

Adaptea used to download models by shelling out to `lms get` or `ollama pull`. Neither
of those programs downloads anything: they hand the job to a daemon — LM Studio's
service, or `ollama serve` — that Adaptea did not start and cannot control. Killing the
client therefore stopped the *reporting* and nothing else. Pause and Stop were untrue,
and closing the app left a transfer running that nothing in the interface could reach.

Everything here runs inside this process. A download is an ordinary coroutine reading an
HTTP stream, so cancelling it is cancelling the thing doing the work: the socket closes,
the writes stop, and the bytes stop arriving. That is the whole point of the module, and
the reason none of it is allowed to delegate to a subprocess.
"""

from __future__ import annotations

import json
import os
import platform
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx

HUGGINGFACE_ENDPOINT = "https://huggingface.co"

#: Long transfers must not be cut off by a read timeout, but a server that never answers
#: at all still has to fail quickly. Connect and write get a bound; reading does not.
DOWNLOAD_TIMEOUT = httpx.Timeout(connect=15.0, read=None, write=60.0, pool=15.0)

_HF_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_HF_QUANTIZATION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class DownloadError(RuntimeError):
    """A download could not be completed. The message is shown to the user verbatim."""


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    """How far a download has got, in bytes rather than in scraped console text."""

    detail: str
    completed: int = 0
    total: int | None = None

    @property
    def percent(self) -> float | None:
        if not self.total or self.total <= 0:
            return None
        return min(100.0, max(0.0, self.completed / self.total * 100))


ProgressCallback = Callable[[DownloadProgress], None]


def _noop(_progress: DownloadProgress) -> None:
    return None


def human_bytes(value: int | None) -> str:
    if not value or value <= 0:
        return "unknown size"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"  # pragma: no cover - the loop returns first.


# ---------------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------------


async def pull_ollama_model(
    model: str,
    *,
    base_url: str = "http://127.0.0.1:11434",
    on_progress: ProgressCallback | None = None,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Pull a model by holding Ollama's own streaming pull open ourselves.

    `ollama pull` is a client for this exact endpoint. Running it as a subprocess put a
    process we could kill in front of a transfer we could not, and the daemon kept the
    pull alive after the client died. Owning the request instead means cancelling this
    coroutine closes the connection, which is the signal Ollama uses to abandon the pull.
    """
    report = on_progress or _noop
    owns_client = client is None
    http = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=DOWNLOAD_TIMEOUT)
    # Ollama reports one layer at a time. Summing what each digest has reported so far is
    # the only way to show progress for the download rather than for its current blob.
    completed_by_digest: dict[str, int] = {}
    total_by_digest: dict[str, int] = {}
    try:
        # `model` is current; `name` is the field older daemons read. Sending both keeps
        # one code path working across the versions people actually have installed.
        payload = {"model": model, "name": model, "stream": True}
        async with http.stream("POST", "/api/pull", json=payload) as response:
            if response.status_code >= 400:
                await response.aread()
                raise DownloadError(_ollama_error(response) or f"Ollama refused to pull {model}.")
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                error = event.get("error")
                if isinstance(error, str) and error:
                    raise DownloadError(error)
                digest = event.get("digest")
                if isinstance(digest, str) and digest:
                    if isinstance(event.get("total"), int):
                        total_by_digest[digest] = int(event["total"])
                    if isinstance(event.get("completed"), int):
                        completed_by_digest[digest] = int(event["completed"])
                status = str(event.get("status") or "Downloading")
                total = sum(total_by_digest.values()) or None
                completed = sum(completed_by_digest.values())
                report(DownloadProgress(detail=status, completed=completed, total=total))
    except httpx.HTTPError as exc:
        raise DownloadError(f"Could not reach Ollama at {base_url}: {exc}") from exc
    finally:
        if owns_client:
            await http.aclose()


def _ollama_error(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or None
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        return str(body["error"])
    return None


# ---------------------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------------------


def lmstudio_models_root() -> Path:
    """Where LM Studio keeps models, which is a plain mirror of Hugging Face repos.

    `<root>/<owner>/<repository>/<files…>` is exactly the layout `lms get` produces, so
    writing there ourselves needs no import step: LM Studio indexes the folder and the
    model appears in `lms ls` like any other.
    """
    override = os.environ.get("ADAPTEA_LMSTUDIO_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    for settings in _lmstudio_settings_paths():
        try:
            data = json.loads(settings.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        folder = data.get("downloadsFolder") if isinstance(data, dict) else None
        if isinstance(folder, str) and folder.strip():
            return Path(folder).expanduser()
    home = Path.home()
    for fallback in (home / ".cache" / "lm-studio" / "models", home / ".lmstudio" / "models"):
        if fallback.is_dir():
            return fallback
    return home / ".lmstudio" / "models"


def _lmstudio_settings_paths() -> Iterable[Path]:
    home = Path.home()
    system = platform.system()
    if system == "Darwin":
        yield home / "Library" / "Application Support" / "LM Studio" / "settings.json"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            yield Path(appdata) / "LM Studio" / "settings.json"
    else:
        yield home / ".config" / "LM Studio" / "settings.json"
    yield home / ".lmstudio" / "settings.json"


def parse_repository(value: str) -> tuple[str, str | None]:
    """Split `owner/repository` from an optional `:quantization` suffix."""
    repo_id = value.strip()
    quantization: str | None = None
    if ":" in repo_id:
        repo_id, quantization = repo_id.rsplit(":", 1)
    if not _HF_REPO.fullmatch(repo_id):
        raise ValueError(
            "Enter a Hugging Face model ID in owner/repository format, "
            "such as lmstudio-community/Qwen2.5-Coder-14B-Instruct-GGUF."
        )
    if quantization is not None and not _HF_QUANTIZATION.fullmatch(quantization):
        raise ValueError("Enter a quantization such as Q4_K_M after the model ID.")
    return repo_id, quantization


@dataclass(frozen=True, slots=True)
class RepositoryFile:
    path: str
    size: int


async def repository_files(
    repo_id: str,
    *,
    client: httpx.AsyncClient,
    endpoint: str = HUGGINGFACE_ENDPOINT,
) -> list[RepositoryFile]:
    """Every file in a repository, with its size, in one request."""
    url = f"{endpoint}/api/models/{repo_id}/tree/main"
    try:
        response = await client.get(url, params={"recursive": "1"}, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise DownloadError(f"Could not reach Hugging Face: {exc}") from exc
    if response.status_code == 404:
        raise DownloadError(f"{repo_id} was not found on Hugging Face.")
    if response.status_code >= 400:
        raise DownloadError(f"Hugging Face refused to list {repo_id} ({response.status_code}).")
    try:
        entries = response.json()
    except ValueError as exc:
        raise DownloadError(f"Hugging Face sent an unreadable file list for {repo_id}.") from exc
    files: list[RepositoryFile] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or entry.get("type") != "file":
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path or path.startswith("."):
            continue
        lfs = entry.get("lfs")
        size = lfs.get("size") if isinstance(lfs, dict) else entry.get("size")
        files.append(RepositoryFile(path=path, size=int(size) if isinstance(size, int) else 0))
    return files


def select_files(files: list[RepositoryFile], quantization: str | None) -> list[RepositoryFile]:
    """Pick the files a runtime actually needs out of a repository.

    A GGUF repository ships one file per quantization and downloading all of them would
    fetch the same model five times over, so one has to be named. Anything else — MLX and
    safetensors weights — is a set of files that only works complete.
    """
    gguf = [item for item in files if item.path.lower().endswith(".gguf")]
    if not gguf:
        return [item for item in files if not item.path.lower().endswith(".gitattributes")]
    if quantization is None and len(gguf) == 1:
        # Nothing to choose between. Asking anyway would turn the common single-build
        # repository into an error the user has no way to answer.
        return gguf
    if quantization is None:
        available = sorted({_quantization_of(item.path) for item in gguf} - {""})
        listed = ", ".join(available) if available else "the quantizations it lists"
        raise DownloadError(
            f"This repository ships several quantizations. Add one after a colon — {listed}."
        )
    wanted = quantization.lower()
    matching = [item for item in gguf if wanted in item.path.lower()]
    if not matching:
        offered = ", ".join(sorted({_quantization_of(item.path) for item in gguf} - {""}))
        raise DownloadError(
            f"No file matching {quantization} in this repository."
            + (f" It offers {offered}." if offered else "")
        )
    return matching


def _quantization_of(path: str) -> str:
    stem = Path(path).stem
    match = re.search(r"((?:IQ|Q)\d+[A-Za-z0-9_]*|[Ff]\d{2}|BF16|\d+bit)", stem)
    return match.group(1) if match else ""


async def download_repository(
    repo_id: str,
    destination_root: Path | str,
    *,
    quantization: str | None = None,
    on_progress: ProgressCallback | None = None,
    client: httpx.AsyncClient | None = None,
    endpoint: str = HUGGINGFACE_ENDPOINT,
) -> Path:
    """Fetch a Hugging Face repository into `<root>/<owner>/<repository>`.

    Each file lands beside its final name as `.part` while it is in flight and is only
    renamed once the bytes are all there, so an interrupted download can never be mistaken
    for a complete model. The parts are deliberately left behind on cancellation: a
    resumed download continues from them with a range request rather than starting again.
    """
    report = on_progress or _noop
    owner, _, name = repo_id.partition("/")
    destination = Path(destination_root).expanduser() / owner / name
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True)
    try:
        report(DownloadProgress(detail=f"Reading {repo_id}…"))
        wanted = select_files(
            await repository_files(repo_id, client=http, endpoint=endpoint), quantization
        )
        total = sum(item.size for item in wanted) or None
        # Bytes already on disk from an earlier attempt count as progress; without this a
        # resumed download would report 0% while writing at the end of a full-size file.
        done = 0
        for item in wanted:
            final = destination / item.path
            if final.is_file():
                done += final.stat().st_size
        report(
            DownloadProgress(
                detail=f"Downloading {len(wanted)} file{'' if len(wanted) == 1 else 's'} "
                f"({human_bytes(total)})",
                completed=done,
                total=total,
            )
        )
        for item in wanted:
            done = await _download_file(
                f"{endpoint}/{repo_id}/resolve/main/{item.path}",
                destination / item.path,
                client=http,
                already_done=done,
                total=total,
                label=Path(item.path).name,
                on_progress=report,
            )
        report(DownloadProgress(detail="Download complete", completed=done, total=total))
        return destination
    finally:
        if owns_client:
            await http.aclose()


async def _download_file(
    url: str,
    destination: Path,
    *,
    client: httpx.AsyncClient,
    already_done: int,
    total: int | None,
    label: str,
    on_progress: ProgressCallback,
) -> int:
    """Stream one file to disk, resuming a `.part` left by an earlier attempt."""
    if destination.is_file():
        return already_done
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    resume_from = part.stat().st_size if part.is_file() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    completed = already_done + resume_from
    try:
        async with client.stream("GET", url, headers=headers) as response:
            if resume_from and response.status_code == 200:
                # The server ignored the range and is sending the file from the start.
                # Honouring that means discarding what we had rather than appending to it.
                resume_from = 0
                completed = already_done
            elif response.status_code >= 400:
                await response.aread()
                raise DownloadError(f"Could not download {label} ({response.status_code}).")
            mode = "ab" if resume_from else "wb"
            handle = part.open(mode)
            try:
                async for chunk in response.aiter_bytes():
                    handle.write(chunk)
                    completed += len(chunk)
                    on_progress(
                        DownloadProgress(
                            detail=f"Downloading {label}", completed=completed, total=total
                        )
                    )
            finally:
                handle.close()
    except httpx.HTTPError as exc:
        raise DownloadError(f"Could not download {label}: {exc}") from exc
    part.replace(destination)
    return completed
