"""Which directory ``./workspace_repo`` means for the job running now.

The agent names its checkout ``./workspace_repo`` everywhere -- in prompts, in
tool arguments the model writes, in tool output it reads back -- so one process
could run only one job at a time. To run several at once (``--max-concurrency``)
each job gets its own checkout, and this module redirects only the filesystem
calls to it. Every string the model sees stays byte-identical to a serial run:
paths in arguments and messages keep their ``./workspace_repo`` spelling, and
absolute paths in tool output are mapped back to ``<cwd>/workspace_repo``. That
is what lets a serially recorded trace replay unchanged under concurrency.

The job's checkout is a context variable, set by the harness before the job's
graph runs. asyncio tasks and LangChain's executors copy the context, so the
agent's sync nodes and tools, which run in worker threads, see the job that
started them. With nothing set every function here is the identity, which is
the serial harness's behaviour.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# What the agent's literal resolves to: abspath, not resolve(), because the
# search tool prints os.path.abspath() of the files it matched.
CANONICAL = os.path.abspath("./workspace_repo")

_JOB_ROOT: ContextVar[str | None] = ContextVar("swe_workspace_root", default=None)


def set_job_root(root: Path | str | None) -> None:
    """Point this context's ``./workspace_repo`` at ``root`` (None: the shared one)."""
    _JOB_ROOT.set(None if root is None else os.path.abspath(root))


def job_root() -> str | None:
    return _JOB_ROOT.get()


def _swap(path: str, old: str, new: str) -> str | None:
    if path == old:
        return new
    if path.startswith(old + os.sep):
        return new + path[len(old):]
    return None


def fs_path(path: Any) -> Any:
    """``path`` as this job's filesystem sees it. Anything outside the checkout is unchanged."""
    root = _JOB_ROOT.get()
    if root is None or not isinstance(path, (str, os.PathLike)):
        return path
    text = os.fspath(path)
    swapped = _swap(os.path.abspath(text), CANONICAL, root)
    return text if swapped is None else swapped


def display_path(path: str) -> str:
    """An absolute path inside this job's checkout, spelled as the serial run spelled it."""
    root = _JOB_ROOT.get()
    if root is None:
        return path
    swapped = _swap(path, root, CANONICAL)
    return path if swapped is None else swapped


def _respell(exc: OSError, real: str, shown: Any) -> OSError:
    # OSError formats its message from these attributes when printed, so the
    # error a tool returns to the model names the path the model asked for.
    if exc.filename == real:
        exc.filename = shown
    return exc


def ws_open(path: Any, *args: Any, **kwargs: Any):
    """open() on this job's checkout; errors name the path as given."""
    real = fs_path(path)
    try:
        return open(real, *args, **kwargs)
    except OSError as exc:
        raise _respell(exc, real, path) from None


def ws_exists(path: Any) -> bool:
    return os.path.exists(fs_path(path))


def ws_makedirs(path: Any, exist_ok: bool = False) -> None:
    real = fs_path(path)
    try:
        os.makedirs(real, exist_ok=exist_ok)
    except OSError as exc:
        raise _respell(exc, real, path) from None
