"""SWE-bench run for the prefix-prefetch experiment: cold phase, then measured warm phase.

The counterpart of ``open_deep_research/tests/run_evaluate.py``. Same two-phase
shape, same metric files, same ablation switch -- the workload underneath is
this repo's architect/developer graph solving SWE-bench instances instead of a
deep-research graph answering questions.

Run it with the LangGraph fork installed (branch Prediction-SWEbench) and a
vLLM exposing ``/v1/agents/*`` if you want the treatment arm; see
plan/03-running.md.
"""

import argparse
import csv
import itertools
import json
import os
import random
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
import asyncio
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import swe_agent_builder
from agent.common import trace_store
from agent.common.model import MODEL_NAME, MODEL_PROVIDER

try:
    # `python tests/run_evaluate.py` puts tests/ on sys.path...
    from kv_metrics_reset import reset_kv_metrics
    from swebench_instances import (
        append_prediction,
        collect_patch,
        load_instances,
        load_instances_file,
        materialize_workspace,
        patch_stats,
        problem_statement_message,
    )
except ImportError:  # ...`python -m tests.run_evaluate` does not.
    from tests.kv_metrics_reset import reset_kv_metrics
    from tests.swebench_instances import (
        append_prediction,
        collect_patch,
        load_instances,
        load_instances_file,
        materialize_workspace,
        patch_stats,
        problem_statement_message,
    )

# override=True so values in .env take precedence over any variables already
# exported in the process environment. Without this, load_dotenv leaves
# pre-set vars (e.g. the *_MODEL_MAX_TOKENS knobs) untouched, and editing .env
# silently has no effect.
load_dotenv(find_dotenv(), override=True)

RANDOM_SEED = 62
TARGET_COMPLETIONS_PER_INSTANCE = 3

# NOTE: Configure the experiment here; these are logged in the run metadata.
seed = 0
max_researcher_iterations = int(os.getenv("MAX_RESEARCHER_ITERATIONS", "6"))
max_react_tool_calls = int(os.getenv("MAX_REACT_TOOL_CALLS", "8"))
max_atomic_tasks = int(os.getenv("MAX_ATOMIC_TASKS", "20"))
# The root graph ships with recursion_limit 200; keep it explicit here so a run
# that hits it says so in its own config rather than in a library default.
recursion_limit = int(os.getenv("GRAPH_RECURSION_LIMIT", "200"))

job_counter = itertools.count(1)
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
METRICS_DIR = Path(__file__).resolve().parent / "timing_logs" / RUN_ID
NODE_METRICS_DIR = METRICS_DIR / "node_metrics"
QUERY_SUMMARY_PATH = METRICS_DIR / "instance_summary.jsonl"
E2E_LATENCY_CSV_PATH = METRICS_DIR / "job_instance_e2e_latency.csv"
PREDICTIONS_PATH = METRICS_DIR / "predictions.jsonl"
# The server's answer to the end-of-cold marker, kept with the run it belongs
# to: it carries the discarded cold-phase `kv_hbm` line and the epoch number
# every warm-phase line is stamped with.
KV_METRICS_RESET_PATH = METRICS_DIR / "kv_metrics_reset.json"
PREDICTION_DIR = Path(__file__).resolve().parents[1] / "SWE_predictions" / RUN_ID
os.environ["LANGGRAPH_PREDICTION_LOG_DIR"] = str(PREDICTION_DIR)

# ---------------------------------------------------------------------------
# Cold phase vs warm phase
# ---------------------------------------------------------------------------
# Every run is two phases with a signal between them:
#
#   cold phase  -> POST /v1/kv_metrics/reset -> warm phase
#
# The cold phase fills the caches (vLLM's prefix cache, any KV connector's CPU
# tier); the warm phase is the one being measured. The reset in between is what
# makes the server's `kv_hbm` numbers describe the warm phase alone -- see
# tests/kv_metrics_reset.py.
#
# Files directly under METRICS_DIR are the run's report and hold **warm phase
# rows only**. Cold-phase output is not discarded -- a cold pass that failed
# halfway explains a warm phase that never hit -- it goes to a parallel tree
# under `cold_phase/` so nothing aggregating the report can pick it up.
COLD_METRICS_DIR = METRICS_DIR / "cold_phase"
COLD_PREDICTION_DIR = PREDICTION_DIR / "cold_phase"
PHASE_COLD = "cold"
PHASE_WARM = "warm"


def _phase_metrics_dir(phase: str) -> Path:
    return METRICS_DIR if phase == PHASE_WARM else COLD_METRICS_DIR


def _phase_node_metrics_dir(phase: str) -> Path:
    return _phase_metrics_dir(phase) / "node_metrics"


def _phase_summary_path(phase: str) -> Path:
    return _phase_metrics_dir(phase) / "instance_summary.jsonl"


def _phase_e2e_csv_path(phase: str) -> Path:
    return _phase_metrics_dir(phase) / "job_instance_e2e_latency.csv"


def _phase_predictions_path(phase: str) -> Path:
    return _phase_metrics_dir(phase) / "predictions.jsonl"


ABLATION_MODE = os.getenv("LANGGRAPH_ABLATION_MODE", "full").strip().lower() or "full"
VALID_BENCHMARK_ABLATION_MODES = {
    "warmup_minimal",
    "warmup_extended",
    "prediction_minimal",
    "full",
}
# Values that mean "run the no-ablation baseline" (ablation study disabled).
ABLATION_DISABLED_MODES = {"baseline", "off", "none", "disabled"}


def _slugify_model(name: str) -> str:
    """Make a model identifier safe for use in a LangSmith project name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "model"


def _pseudo_dynamic_enabled() -> bool:
    """Mirror of langgraph's own reading of LANGGRAPH_PROMPT_PSEUDO_DYNAMIC."""
    return os.getenv("LANGGRAPH_PROMPT_PSEUDO_DYNAMIC", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


# ---------------------------------------------------------------------------
# LangSmith log naming
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = os.getenv(
    "LANGSMITH_EXPERIMENT",
    f"swe-{MODEL_PROVIDER}-{_slugify_model(MODEL_NAME)}-{ABLATION_MODE}-{RUN_ID}",
)
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_PROJECT"] = EXPERIMENT_NAME
os.environ["LANGCHAIN_PROJECT"] = EXPERIMENT_NAME  # legacy alias for older SDKs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWE-bench through the LangGraph SWE agent.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.getenv("SWE_DATASET", "SWE-bench/SWE-bench_Verified"),
        help="HF dataset name or a local .json/.jsonl/.parquet path.",
    )
    parser.add_argument("--split", type=str, default=os.getenv("SWE_SPLIT", "test"))
    parser.add_argument(
        "--max-instances",
        type=int,
        default=6,
        help="Number of benchmark instances to sample (0 = all).",
    )
    parser.add_argument(
        "--instance-ids",
        type=str,
        nargs="*",
        default=None,
        help="Run exactly these instance ids instead of sampling.",
    )
    parser.add_argument(
        "--instances-file",
        type=Path,
        default=None,
        help=(
            "Run instances from a local JSONL instead of the dataset. One "
            "object per line with instance_id / repo / base_commit / "
            "problem_statement. See tests/instances/."
        ),
    )
    parser.add_argument(
        "--completions-per-instance",
        type=int,
        default=TARGET_COMPLETIONS_PER_INSTANCE,
        help="Number of runs (instances) to execute per benchmark instance.",
    )
    parser.add_argument(
        "--ablation-mode",
        type=str,
        default=None,
        help=(
            "Override the LANGGRAPH_ABLATION_MODE env var. One of: "
            "warmup_minimal, warmup_extended, prediction_minimal, full; or "
            "baseline/off/none/disabled to DISABLE the ablation study."
        ),
    )
    parser.add_argument(
        "--pseudo-dynamic",
        choices=("on", "off"),
        default=None,
        help=(
            "Whether the prefix builder fills pseudo-dynamic prompt segments "
            "when warming a predicted node. Unset leaves "
            "LANGGRAPH_PROMPT_PSEUDO_DYNAMIC as the environment has it "
            "(default on). No effect when the prefix prefetch is disabled."
        ),
    )
    parser.add_argument(
        "--cold-instances-per-query",
        type=int,
        default=1,
        help=(
            "Runs of each sampled instance in the cold phase, before the warm "
            "phase is measured. Their output goes to cold_phase/ and is not "
            "counted towards --completions-per-instance."
        ),
    )
    parser.add_argument(
        "--skip-cold-phase",
        action="store_true",
        help=(
            "Run no cold phase in this process. For when the caches were "
            "already filled by a separate invocation -- the reset still fires "
            "before the warm phase, so the boundary is still drawn."
        ),
    )
    parser.add_argument(
        "--no-hbm-flush",
        action="store_true",
        help=(
            "Keep the cold phase's blocks resident in GPU HBM instead of "
            "evicting them at the phase boundary."
        ),
    )
    parser.add_argument(
        "--trace-mode",
        choices=("off", "record", "pinned", "offline"),
        default=None,
        help=(
            "Fixed-trace study (agent/common/trace_store.py). record: dump every "
            "chat-completion request/response to --trace-path. pinned: issue the "
            "live call for timing but hand the agent the recorded response. "
            "offline: replay without touching the server. Overrides "
            "SWE_TRACE_MODE / ODR_TRACE_MODE."
        ),
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Trace JSONL, written in record mode and read in pinned/offline. "
             "Overrides SWE_TRACE_PATH / ODR_TRACE_PATH.",
    )
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help=(
            "Clear LANGGRAPH_VLLM_AGENT_ENABLE / _BASE_URL for this run, after "
            ".env has been loaded. The baseline arm -- and what a `record` run "
            "should use, so the trace is a stock-stack trajectory."
        ),
    )
    parser.add_argument(
        "--no-kv-metrics-reset",
        action="store_true",
        help="Do not POST /v1/kv_metrics/reset between the phases.",
    )
    return parser.parse_args()


def _validate_benchmark_ablation_mode(mode: str) -> str:
    if mode in ABLATION_DISABLED_MODES:
        return mode
    if mode not in VALID_BENCHMARK_ABLATION_MODES:
        valid = ", ".join(sorted(VALID_BENCHMARK_ABLATION_MODES))
        disabled = ", ".join(sorted(ABLATION_DISABLED_MODES))
        raise ValueError(
            "run_evaluate.py expects LANGGRAPH_ABLATION_MODE to be one of "
            f"{valid} (or {disabled} to disable the ablation study)."
        )
    return mode


class TimingCollector:
    """Collect per-task timings from LangGraph task stream events."""

    def __init__(self) -> None:
        self._active: dict[str, dict[str, Any]] = {}
        self._invocation_counts: dict[str, int] = defaultdict(int)
        self.series: list[dict[str, Any]] = []
        self.totals_by_name_ns: dict[str, int] = defaultdict(int)

    def start(self, namespace: tuple[str, ...], payload: dict[str, Any]) -> None:
        node_name = payload["name"]
        self._invocation_counts[node_name] += 1
        self._active[payload["id"]] = {
            "node_name": node_name,
            "namespace": list(namespace),
            "invocation_index": self._invocation_counts[node_name],
            "started_at_ns": time.perf_counter_ns(),
        }

    def finish(self, payload: dict[str, Any]) -> None:
        started = self._active.pop(payload["id"], None)
        if started is None:
            return

        duration_ns = time.perf_counter_ns() - started["started_at_ns"]
        event = {
            "task_id": payload["id"],
            "node_name": started["node_name"],
            "namespace": started["namespace"],
            "invocation_index": started["invocation_index"],
            "duration_ns": duration_ns,
            "duration_seconds": duration_ns / 1_000_000_000,
            "error": _normalize_json_error(payload.get("error")),
            "interrupt_count": len(payload.get("interrupts", [])),
        }
        self.series.append(event)
        self.totals_by_name_ns[started["node_name"]] += duration_ns

    def build_summary(self) -> dict[str, Any]:
        now_ns = time.perf_counter_ns()
        unfinished_tasks = []
        for task_id, started in self._active.items():
            elapsed_ns = now_ns - started["started_at_ns"]
            unfinished_tasks.append(
                {
                    "task_id": task_id,
                    "node_name": started["node_name"],
                    "namespace": started["namespace"],
                    "invocation_index": started["invocation_index"],
                    "elapsed_ns_so_far": elapsed_ns,
                    "elapsed_seconds_so_far": elapsed_ns / 1_000_000_000,
                }
            )

        totals_by_node_name_ns = dict(sorted(self.totals_by_name_ns.items()))
        totals_by_node_name_seconds = {
            key: value / 1_000_000_000 for key, value in totals_by_node_name_ns.items()
        }

        return {
            "time_series": self.series,
            "totals_by_node_name_ns": totals_by_node_name_ns,
            "totals_by_node_name_seconds": totals_by_node_name_seconds,
            "unfinished_tasks": unfinished_tasks,
        }


def _normalize_json_error(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _select_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.instances_file is not None:
        instances = load_instances_file(args.instances_file)
        if 0 < args.max_instances < len(instances):
            instances = instances[: args.max_instances]
        print(f"Instances: {len(instances)} from {args.instances_file} (dataset not used)")
        return instances
    return load_instances(
        args.dataset,
        args.split,
        instance_ids=args.instance_ids,
        max_instances=args.max_instances,
        seed=RANDOM_SEED,
    )


def _job_config(
    *,
    job_id: str,
    thread_id: str,
    instance: dict[str, Any],
    logical_job_index: int,
    instance_index: int,
    phase: str,
) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": thread_id,
            # The KV-eviction / provenance key. Every LLM call this job makes
            # carries it to vLLM in extra_body -- see
            # agent/common/llm_request_metadata.py.
            "job_id": job_id,
            "seed": seed,
            "max_researcher_iterations": max_researcher_iterations,
            "max_react_tool_calls": max_react_tool_calls,
            "max_atomic_tasks": max_atomic_tasks,
        },
        "recursion_limit": recursion_limit,
        "run_name": f"swe-job{job_id}-{instance['instance_id']}-inst{instance_index}",
        "metadata": {
            "run_id": RUN_ID,
            "experiment": EXPERIMENT_NAME,
            "model": MODEL_NAME,
            "model_provider": MODEL_PROVIDER,
            "ablation_mode": ABLATION_MODE,
            "swe_instance_id": instance["instance_id"],
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "job_id": job_id,
            "logical_job_index": logical_job_index,
            "instance_index": instance_index,
            "phase": phase,
            "seed": seed,
        },
        "tags": [
            f"provider:{MODEL_PROVIDER}",
            f"model:{_slugify_model(MODEL_NAME)}",
            f"ablation:{ABLATION_MODE}",
            f"run:{RUN_ID}",
            f"phase:{phase}",
        ],
    }


async def target(
    instance: dict[str, Any],
    *,
    logical_job_index: int,
    instance_index: int,
    phase: str = PHASE_WARM,
):
    graph = swe_agent_builder.compile(
        checkpointer=MemorySaver(),
        _is_root=True,
    )
    job_id = str(next(job_counter))
    thread_id = str(uuid.uuid4())
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started_at_ns = time.perf_counter_ns()

    collector = TimingCollector()
    setup_error: str | None = None
    stream_error: str | None = None
    state_error: str | None = None
    final_state: dict[str, Any] | None = None
    patch = ""

    config = _job_config(
        job_id=job_id,
        thread_id=thread_id,
        instance=instance,
        logical_job_index=logical_job_index,
        instance_index=instance_index,
        phase=phase,
    )

    # Workspace setup is outside the measured stream but inside the job: a
    # clone that fails is an instance that did not run, and it has to be
    # reported as such rather than as a zero-node success.
    try:
        materialize_workspace(instance)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal to the run
        setup_error = f"{type(exc).__name__}: {exc}"

    if setup_error is None:
        try:
            async for part in graph.astream(
                {"implementation_research_scratchpad": [problem_statement_message(instance)]},
                config,
                stream_mode="tasks",
                subgraphs=True,
                version="v2",
            ):
                if part["type"] != "tasks":
                    continue
                payload = part["data"]
                if "triggers" in payload:
                    collector.start(tuple(part["ns"]), payload)
                else:
                    collector.finish(payload)
        except Exception as exc:
            stream_error = str(exc)
        finally:
            try:
                final_state_snapshot = await graph.aget_state(config)
                final_state = final_state_snapshot.values
            except Exception as exc:
                state_error = str(exc)
            # Collected even when the stream failed: a partial edit is still
            # what the agent left on disk, and scoring it is how a crash mid-run
            # is told apart from a crash before any edit.
            try:
                patch = collect_patch()
            except Exception as exc:  # noqa: BLE001
                state_error = state_error or f"patch_collect: {exc}"

    finished_at_ns = time.perf_counter_ns()
    finished_at_utc = datetime.now(timezone.utc).isoformat()
    query_duration_ns = finished_at_ns - started_at_ns
    summary = collector.build_summary()
    node_metrics_path = (
        _phase_node_metrics_dir(phase) / f"job_{job_id}_thread_{thread_id}.jsonl"
    )

    append_prediction(
        _phase_predictions_path(phase),
        instance,
        patch,
        f"{EXPERIMENT_NAME}-inst{instance_index}",
    )

    base_record = {
        "run_id": RUN_ID,
        "swe_instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "logical_job_index": logical_job_index,
        "instance_index": instance_index,
        # Every row still says which phase it came from, even though the two
        # phases are already in separate trees. The label is what survives when
        # rows are concatenated, and a cold row averaged into a warm aggregate
        # is exactly the mistake this split exists to prevent.
        "phase": phase,
        "thread_id": thread_id,
        "job_id": job_id,
        "started_at": started_at_utc,
        "finished_at": finished_at_utc,
    }

    for event in summary["time_series"]:
        _append_jsonl(node_metrics_path, {**base_record, **event})

    _append_jsonl(
        _phase_summary_path(phase),
        {
            **base_record,
            **patch_stats(patch),
            "node_metrics_jsonl": str(node_metrics_path.resolve()),
            "query_duration_ns": query_duration_ns,
            "query_duration_seconds": query_duration_ns / 1_000_000_000,
            "node_execution_count": len(summary["time_series"]),
            "totals_by_node_name_ns": summary["totals_by_node_name_ns"],
            "totals_by_node_name_seconds": summary["totals_by_node_name_seconds"],
            "unfinished_tasks": summary["unfinished_tasks"],
            "setup_error": setup_error,
            "stream_error": stream_error,
            "state_error": state_error,
            "final_state_present": final_state is not None,
            "empty_patch": not patch.strip(),
        },
    )

    success = setup_error is None and stream_error is None and state_error is None
    return {
        "success": success,
        "run_id": RUN_ID,
        "swe_instance_id": instance["instance_id"],
        "logical_job_index": logical_job_index,
        "instance_index": instance_index,
        "phase": phase,
        "job_id": job_id,
        "thread_id": thread_id,
        "started_at": started_at_utc,
        "finished_at": finished_at_utc,
        "query_duration_ns": query_duration_ns,
        "query_duration_seconds": query_duration_ns / 1_000_000_000,
        "setup_error": setup_error,
        "stream_error": stream_error,
        "state_error": state_error,
        "empty_patch": not patch.strip(),
        **patch_stats(patch),
    }


def _e2e_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "swe_instance_id": result["swe_instance_id"],
        "logical_job_index": result["logical_job_index"],
        "instance_index": result["instance_index"],
        # Redundant with the file this row is written to, and kept anyway: it
        # is the only thing that still identifies the phase once someone
        # concatenates two CSVs.
        "phase": result["phase"],
        "job_id": result["job_id"],
        "thread_id": result["thread_id"],
        "started_at": result["started_at"],
        "finished_at": result["finished_at"],
        "query_duration_ns": result["query_duration_ns"],
        "query_duration_seconds": result["query_duration_seconds"],
        "success": result["success"],
        "empty_patch": result["empty_patch"],
        "patch_files": result["patch_files"],
        "setup_error": result["setup_error"],
        "stream_error": result["stream_error"],
        "state_error": result["state_error"],
    }


async def _run_cold_phase(
    selected: list[dict[str, Any]],
    runs_per_instance: int,
) -> list[dict[str, Any]]:
    """Fill the caches. Nothing here is part of the reported result.

    A full pass over every sampled instance before any instance is repeated:
    the warm phase runs all of them, so warming only the first would leave the
    rest cold and turn "warm phase" into a mix of both.
    """
    rows: list[dict[str, Any]] = []
    total = runs_per_instance * len(selected)
    position = 0
    for instance_index in range(1, runs_per_instance + 1):
        for job_index, instance in enumerate(selected, start=1):
            position += 1
            print(
                f"[cold {position}/{total}] {instance['instance_id']} "
                f"run {instance_index} -- filling caches, not measured"
            )
            result = await target(
                instance,
                logical_job_index=job_index,
                instance_index=instance_index,
                phase=PHASE_COLD,
            )
            rows.append(result)
            _append_csv_row(_phase_e2e_csv_path(PHASE_COLD), _e2e_csv_row(result))
    return rows


async def _run_cold_phase_isolated(
    selected: list[dict[str, Any]],
    runs_per_instance: int,
) -> list[dict[str, Any]]:
    """Run the cold phase with its prediction logs redirected too.

    `LANGGRAPH_PREDICTION_LOG_DIR` is read with `os.getenv` on every write
    (`langgraph/pregel/_prediction.py::_default_log_path` and friends), so
    repointing it for the duration of this call keeps the cold jobs' prediction,
    trace and prefix logs out of the reported set. Safe because the phases are
    strictly sequential.
    """
    previous = os.environ.get("LANGGRAPH_PREDICTION_LOG_DIR")
    os.environ["LANGGRAPH_PREDICTION_LOG_DIR"] = str(COLD_PREDICTION_DIR)
    try:
        return await _run_cold_phase(selected, runs_per_instance)
    finally:
        if previous is None:
            os.environ.pop("LANGGRAPH_PREDICTION_LOG_DIR", None)
        else:
            os.environ["LANGGRAPH_PREDICTION_LOG_DIR"] = previous


async def _run_warm_phase(
    selected: list[dict[str, Any]],
    *,
    completions_per_instance: int,
) -> dict[str, Any]:
    """The measured pass: every instance, `completions_per_instance` times.

    Strictly sequential, and unlike ODR that is not a choice: all jobs share
    one `./workspace_repo`, because the agent hard-codes that path. Two jobs at
    once would edit each other's checkout. Concurrency here needs per-job
    workspaces first -- see plan/02-required-changes.md.
    """
    completion_rows: list[dict[str, Any]] = []
    total = completions_per_instance * len(selected)
    position = 0
    for instance_index in range(1, completions_per_instance + 1):
        for job_index, instance in enumerate(selected, start=1):
            position += 1
            print(f"[warm {position}/{total}] {instance['instance_id']} run {instance_index}")
            result = await target(
                instance,
                logical_job_index=job_index,
                instance_index=instance_index,
                phase=PHASE_WARM,
            )
            completion_rows.append(result)
            _append_csv_row(_phase_e2e_csv_path(PHASE_WARM), _e2e_csv_row(result))

    failed = [row for row in completion_rows if not row["success"]]
    empty = [row for row in completion_rows if row["empty_patch"]]
    return {
        "run_id": RUN_ID,
        "sampled_instances": len(selected),
        "completions_per_instance": completions_per_instance,
        "completed_runs": len(completion_rows),
        "failed_runs": len(failed),
        "empty_patch_runs": len(empty),
        "timing_metrics_dir": str(METRICS_DIR),
        "prediction_logs_dir": str(PREDICTION_DIR),
        "instance_summary_jsonl": str(QUERY_SUMMARY_PATH),
        "e2e_latency_csv": str(E2E_LATENCY_CSV_PATH),
        "predictions_jsonl": str(PREDICTIONS_PATH),
    }


async def main():
    global ABLATION_MODE, EXPERIMENT_NAME
    args = parse_args()

    # Both after load_dotenv(override=True), which is why they cannot be done
    # by exporting in the shell: a value in .env would win over the export.
    if args.trace_mode:
        os.environ["SWE_TRACE_MODE"] = args.trace_mode
    if args.trace_path:
        os.environ["SWE_TRACE_PATH"] = str(args.trace_path)
    if args.no_prefetch:
        # `vllm_agent_enabled()` is true if _ENABLE == "1" OR _BASE_URL is set,
        # so both have to go.
        os.environ["LANGGRAPH_VLLM_AGENT_ENABLE"] = "0"
        os.environ.pop("LANGGRAPH_VLLM_AGENT_BASE_URL", None)

    # Resolves and validates the trace config once, here, rather than inside the
    # first model call where a bad path surfaces as a fake connection error.
    print(trace_store.describe())

    if args.pseudo_dynamic:
        os.environ["LANGGRAPH_PROMPT_PSEUDO_DYNAMIC"] = (
            "1" if args.pseudo_dynamic == "on" else "0"
        )

    if args.ablation_mode:
        ABLATION_MODE = args.ablation_mode.strip().lower()
        env_value = "baseline" if ABLATION_MODE in ABLATION_DISABLED_MODES else ABLATION_MODE
        os.environ["LANGGRAPH_ABLATION_MODE"] = env_value
        if not os.getenv("LANGSMITH_EXPERIMENT"):
            EXPERIMENT_NAME = (
                f"swe-{MODEL_PROVIDER}-{_slugify_model(MODEL_NAME)}-{ABLATION_MODE}-{RUN_ID}"
            )
            os.environ["LANGSMITH_PROJECT"] = EXPERIMENT_NAME
            os.environ["LANGCHAIN_PROJECT"] = EXPERIMENT_NAME

    _validate_benchmark_ablation_mode(ABLATION_MODE)
    if ABLATION_MODE in ABLATION_DISABLED_MODES:
        print("Ablation study: DISABLED (baseline / no-ablation run)")

    selected = _select_instances(args)

    print(f"Timing metrics directory: {METRICS_DIR}")
    print(f"Instance summary JSONL: {QUERY_SUMMARY_PATH}")
    print(f"E2E latency CSV: {E2E_LATENCY_CSV_PATH}")
    print(f"Predictions JSONL: {PREDICTIONS_PATH}")
    print(f"Prediction logs directory: {PREDICTION_DIR}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Model provider: {MODEL_PROVIDER}")
    print(f"Model: {MODEL_NAME}")
    print(f"LangSmith project: {EXPERIMENT_NAME}")
    print(f"Ablation mode: {ABLATION_MODE}")
    print(
        "Pseudo-dynamic prefix fill: "
        f"{'enabled' if _pseudo_dynamic_enabled() else 'DISABLED'}"
    )
    print(
        "Loop bounds: "
        f"max_researcher_iterations={max_researcher_iterations}, "
        f"max_react_tool_calls={max_react_tool_calls}, "
        f"max_atomic_tasks={max_atomic_tasks}, "
        f"recursion_limit={recursion_limit}"
    )
    print(
        f"Running {len(selected)} instances sequentially, "
        f"{args.completions_per_instance} run(s) each"
    )
    print(
        "Phases: cold "
        f"({0 if args.skip_cold_phase else args.cold_instances_per_query} "
        f"run(s) per instance, output under {COLD_METRICS_DIR}) "
        "-> POST /v1/kv_metrics/reset"
        f"{'' if args.no_hbm_flush else ' (+ HBM flush)'} "
        "-> warm (measured, reported above)"
    )

    # ---- cold phase -------------------------------------------------------
    cold_rows: list[dict[str, Any]] = []
    if args.skip_cold_phase:
        print("Cold phase: SKIPPED (--skip-cold-phase)")
    elif args.cold_instances_per_query > 0:
        cold_rows = await _run_cold_phase_isolated(selected, args.cold_instances_per_query)
        failed_cold = [row for row in cold_rows if not row["success"]]
        print(f"Cold phase complete: {len(cold_rows)} runs, {len(failed_cold)} failed")
        if failed_cold:
            print(
                "  WARNING: some cold runs failed, so the caches are only "
                "partly filled for the warm phase"
            )

    # ---- the boundary -----------------------------------------------------
    kv_metrics_reset: dict[str, Any] | None = None
    if not args.no_kv_metrics_reset:
        kv_metrics_reset = await reset_kv_metrics(
            f"cold_done_{RUN_ID}",
            flush_hbm=not args.no_hbm_flush,
            record_to=KV_METRICS_RESET_PATH,
        )

    # ---- warm phase -------------------------------------------------------
    print(f"Warm phase: measuring, report goes to {METRICS_DIR}")
    results = await _run_warm_phase(
        selected,
        completions_per_instance=args.completions_per_instance,
    )
    results["cold_runs"] = len(cold_rows)
    results["cold_runs_failed"] = sum(1 for row in cold_rows if not row["success"])
    results["cold_phase_metrics_dir"] = str(COLD_METRICS_DIR)
    results["cold_phase_prediction_logs_dir"] = str(COLD_PREDICTION_DIR)
    results["kv_metrics_reset_ok"] = bool(kv_metrics_reset and kv_metrics_reset.get("ok"))
    results["kv_metrics_epoch"] = kv_metrics_reset.get("epoch") if kv_metrics_reset else None
    results["hbm_flushed"] = kv_metrics_reset.get("hbm_flushed") if kv_metrics_reset else None
    results["kv_metrics_reset_json"] = str(KV_METRICS_RESET_PATH) if kv_metrics_reset else None
    print(
        "\nScore the patches on a machine with Docker:\n"
        f"  swebench eval -p {PREDICTIONS_PATH} -d {args.dataset} --run-id {RUN_ID}"
    )
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)
