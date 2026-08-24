"""SWE-bench instances in, unified diffs out.

The three harnesses share this module. It does the four things that turn a
LangGraph agent into a SWE-bench run:

1. **Load** instances from a SWE-bench dataset (HF hub, or a local
   .json/.jsonl/.parquet), using the official loader so aliases and splits
   behave exactly as ``swebench eval`` expects.
2. **Materialise** one instance into the working tree the agent reads. The
   agent hard-codes ``./workspace_repo`` in four places
   (architect/graph.py, developer/graph.py, tools/write.py), so rather than
   rewrite those, each job wipes and re-creates that directory at the
   instance's ``base_commit``. Jobs run one at a time, so a single shared
   workspace is not a race -- and one workspace is what keeps the harness's
   determinism story simple.
3. **Collect** the patch: whatever the agent wrote to disk, as a diff against
   ``base_commit``.
4. **Emit** ``predictions.jsonl`` in the schema ``swebench eval`` reads.

Nothing here talks to vLLM, LangGraph or Docker. Evaluation of the patches is
a separate step on a machine with Docker (see plan/03-running.md); keeping it
out of the benchmark process means a scoring failure cannot corrupt a latency
measurement, and the GPU box never needs a Docker daemon.

Determinism note: unlike ODR -- whose Tavily results had to be frozen into a
cache to make two runs comparable -- every tool this agent has reads the local
checkout. Pinned to ``base_commit``, those reads are already reproducible, so
there is no cache to maintain and no cached-search caveat to attach to the
results. The remaining non-determinism is the server's (continuous batching,
APC, quantised kernels), which is the thing being measured.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

# The directory the agent reads. Kept as the literal the agent already uses;
# override only if you also change the agent's own references.
WORKSPACE_DIR = Path(os.getenv("SWE_WORKSPACE_DIR", "./workspace_repo")).resolve()

# Bare mirrors live here so a repo is cloned from GitHub once per machine and
# every later instance of the same repo is a local, hard-linked clone. Twelve
# django instances should not mean twelve network clones of django.
REPO_CACHE_DIR = Path(
    os.getenv("SWE_REPO_CACHE", Path.home() / ".cache" / "swe-agent-bench" / "repos")
).resolve()

DEFAULT_DATASET = os.getenv("SWE_DATASET", "SWE-bench/SWE-bench_Verified")
DEFAULT_SPLIT = os.getenv("SWE_SPLIT", "test")

GIT_TIMEOUT_S = int(os.getenv("SWE_GIT_TIMEOUT_S", "900"))


class InstanceSetupError(RuntimeError):
    """The workspace could not be prepared, so this instance cannot be run."""


def _git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
        env={
            **os.environ,
            # A prompt for credentials inside a benchmark run hangs forever.
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "echo",
        },
    )
    if check and result.returncode != 0:
        raise InstanceSetupError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip()[:2000]}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def load_instances(
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    *,
    instance_ids: list[str] | None = None,
    max_instances: int = 0,
    seed: int = 62,
) -> list[dict[str, Any]]:
    """Load, then sample deterministically.

    Sampling is a seeded `random.Random(seed).sample`, the same construction
    ODR uses to pick benchmark queries, so two arms of an experiment see the
    same instances in the same order without pinning ids by hand.
    """
    from swebench.harness.utils import load_swebench_dataset

    dataset = list(load_swebench_dataset(dataset_name, split, instance_ids))
    if max_instances and 0 < max_instances < len(dataset):
        dataset = random.Random(seed).sample(dataset, max_instances)
        # Sample order is arbitrary; sort so the run order is a property of the
        # set rather than of the sampler's internals.
        dataset.sort(key=lambda item: item["instance_id"])
    return dataset


def load_instances_file(path: Path) -> list[dict[str, Any]]:
    """Instances from a local JSONL, for hand-built sets.

    The analogue of ODR's ``tests/questions/*.jsonl``: a set chosen to share
    content (same repo, neighbouring modules) is what makes cross-instance KV
    reuse measurable on purpose, instead of hoping the benchmark's unrelated
    instances happen to overlap.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            missing = {"instance_id", "repo", "base_commit", "problem_statement"} - set(record)
            if missing:
                raise ValueError(f"{path}:{number} missing keys: {sorted(missing)}")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no instances")
    return records


# ---------------------------------------------------------------------------
# 2. Materialising the workspace
# ---------------------------------------------------------------------------
def _mirror_path(repo: str) -> Path:
    return REPO_CACHE_DIR / f"{repo.replace('/', '__')}.git"


def _ensure_mirror(repo: str, base_commit: str) -> Path:
    """A bare mirror of ``repo`` that contains ``base_commit``."""
    mirror = _mirror_path(repo)
    if not mirror.exists():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        url = os.getenv("SWE_REPO_URL_TEMPLATE", "https://github.com/{repo}.git").format(
            repo=repo
        )
        _git("clone", "--mirror", "--quiet", url, str(mirror))
        return mirror

    # The mirror exists but may predate the commit (a repo cached during an
    # earlier run against a different split). Fetch only when it is actually
    # missing -- an unconditional fetch per instance is a network round trip
    # inside the measured section of the benchmark.
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{base_commit}^{{commit}}"],
        cwd=str(mirror),
        capture_output=True,
    )
    if probe.returncode != 0:
        _git("fetch", "--quiet", "--all", cwd=mirror)
    return mirror


def materialize_workspace(instance: dict[str, Any]) -> Path:
    """Put ``instance`` on disk at ``WORKSPACE_DIR``, at its base commit.

    Destructive by design: the previous instance's edits must not survive into
    this one, and a `git checkout .` would leave untracked files the agent
    created behind.
    """
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    mirror = _ensure_mirror(repo, base_commit)

    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)
    WORKSPACE_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Local clone: git hard-links objects, so this costs almost nothing after
    # the first instance of a repo.
    _git("clone", "--quiet", "--no-checkout", str(mirror), str(WORKSPACE_DIR))
    _git("checkout", "--quiet", "--detach", base_commit, cwd=WORKSPACE_DIR)
    # Identity is needed only so `git stash`/`commit` style operations do not
    # fail if something downstream tries one; the harness itself only diffs.
    _git("config", "user.email", "bench@localhost", cwd=WORKSPACE_DIR)
    _git("config", "user.name", "swe-agent-bench", cwd=WORKSPACE_DIR)
    return WORKSPACE_DIR


# ---------------------------------------------------------------------------
# 3. Collecting the patch
# ---------------------------------------------------------------------------
def collect_patch() -> str:
    """Everything the agent changed, as a diff against the base commit.

    Staged first so files the agent *created* are included -- a plain
    `git diff` would silently drop new files, which is a large share of what
    an implementation agent produces.
    """
    if not WORKSPACE_DIR.exists():
        return ""
    _git("add", "-A", cwd=WORKSPACE_DIR)
    return _git("diff", "--cached", "--no-color", cwd=WORKSPACE_DIR, check=False)


def patch_stats(patch: str) -> dict[str, int]:
    files = sum(1 for line in patch.splitlines() if line.startswith("diff --git "))
    added = sum(
        1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")
    )
    return {"patch_files": files, "patch_added_lines": added, "patch_removed_lines": removed}


# ---------------------------------------------------------------------------
# 4. Predictions
# ---------------------------------------------------------------------------
def append_prediction(
    path: Path,
    instance: dict[str, Any],
    patch: str,
    model_name: str,
) -> None:
    """One line of the file ``swebench eval -p`` consumes.

    An empty patch is written rather than skipped: SWE-bench scores a missing
    instance and an unresolved instance identically, but only the written line
    proves the agent ran and produced nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "instance_id": instance["instance_id"],
                    "model_name_or_path": model_name,
                    "model_patch": patch,
                },
                ensure_ascii=True,
            )
            + "\n"
        )


def problem_statement_message(instance: dict[str, Any]) -> dict[str, str]:
    """The instance as the root graph's first message.

    The root state's only input key is ``implementation_research_scratchpad``
    (agent/graph.py::AgentState), so the issue text enters the graph as the
    architect's first scratchpad entry.
    """
    return {"role": "user", "content": instance["problem_statement"]}
