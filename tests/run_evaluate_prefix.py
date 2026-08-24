"""SWE-bench run instrumented for longest-repeating-prefix analysis.

The counterpart of ``open_deep_research/tests/run_evaluate_prefix.py``: a copy
of ``run_evaluate.py`` with the cold/warm phase split removed -- every job runs
once, in one phase, and its output lands directly under the run's metrics
directory. What is added instead is the prefix analyzer in
``tests/prefix_analysis.py``, which records how much of every prompt the node
issuing it had already sent.

This is the one harness that does **not** depend on the fork's compile-time
prompt analysis: it measures the prompts the agent actually sends, client side.
So it produces a real answer for this agent today, and it is the right way to
find out how much prefix there is to warm before investing in warming it.
"""

import argparse
import csv
import itertools
import json
import os
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
from agent.common.model import MODEL_NAME, MODEL_PROVIDER

try:
    # `python tests/run_evaluate_prefix.py` puts tests/ on sys.path...
    from kv_reuse_snapshot import snapshot_kv_metrics
    from prefix_analysis import PrefixAnalyzer, PrefixCaptureHandler
    from swebench_instances import (
        append_prediction,
        collect_patch,
        load_instances,
        load_instances_file,
        materialize_workspace,
        patch_stats,
        problem_statement_message,
    )
except ImportError:  # ...`python -m tests.run_evaluate_prefix` does not.
    from tests.kv_reuse_snapshot import snapshot_kv_metrics
    from tests.prefix_analysis import PrefixAnalyzer, PrefixCaptureHandler
    from tests.swebench_instances import (
        append_prediction,
        collect_patch,
        load_instances,
        load_instances_file,
        materialize_workspace,
        patch_stats,
        problem_statement_message,
    )

load_dotenv(find_dotenv(), override=True)

RANDOM_SEED = 62
TARGET_COMPLETIONS_PER_INSTANCE = 3

seed = 0
max_researcher_iterations = int(os.getenv("MAX_RESEARCHER_ITERATIONS", "6"))
max_react_tool_calls = int(os.getenv("MAX_REACT_TOOL_CALLS", "8"))
max_atomic_tasks = int(os.getenv("MAX_ATOMIC_TASKS", "20"))
recursion_limit = int(os.getenv("GRAPH_RECURSION_LIMIT", "200"))

job_counter = itertools.count(1)
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
# Its own tree, not run_evaluate.py's timing_logs/: these runs measure prompt
# reuse rather than latency, and mixing them into the same directory makes any
# tool that globs timing_logs/*/ silently average two different experiments.
METRICS_DIR = Path(__file__).resolve().parent / "prefix_logs" / RUN_ID
NODE_METRICS_DIR = METRICS_DIR / "node_metrics"
SUMMARY_PATH = METRICS_DIR / "instance_summary.jsonl"
E2E_LATENCY_CSV_PATH = METRICS_DIR / "job_instance_e2e_latency.csv"
PREDICTIONS_PATH = METRICS_DIR / "predictions.jsonl"
PREFIX_ANALYSIS_DIR = METRICS_DIR / "prefix_analysis"
PREDICTION_DIR = Path(__file__).resolve().parents[1] / "SWE_predictions" / RUN_ID
os.environ["LANGGRAPH_PREDICTION_LOG_DIR"] = str(PREDICTION_DIR)

ABLATION_MODE = os.getenv("LANGGRAPH_ABLATION_MODE", "full").strip().lower() or "full"
VALID_BENCHMARK_ABLATION_MODES = {
    "warmup_minimal",
    "warmup_extended",
    "prediction_minimal",
    "full",
}
ABLATION_DISABLED_MODES = {"baseline", "off", "none", "disabled"}


def _slugify_model(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "model"


def _pseudo_dynamic_enabled() -> bool:
    return os.getenv("LANGGRAPH_PROMPT_PSEUDO_DYNAMIC", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


EXPERIMENT_NAME = os.getenv(
    "LANGSMITH_EXPERIMENT",
    f"swe-prefix-{MODEL_PROVIDER}-{_slugify_model(MODEL_NAME)}-{ABLATION_MODE}-{RUN_ID}",
)
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_PROJECT"] = EXPERIMENT_NAME
os.environ["LANGCHAIN_PROJECT"] = EXPERIMENT_NAME

PREFIX_ANALYZER = PrefixAnalyzer(
    run_id=RUN_ID,
    analysis_dir=PREFIX_ANALYSIS_DIR,
    model_name=MODEL_NAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench with longest-repeating-prefix analysis."
    )
    parser.add_argument("--dataset", type=str, default=os.getenv("SWE_DATASET", "SWE-bench/SWE-bench_Verified"))
    parser.add_argument("--split", type=str, default=os.getenv("SWE_SPLIT", "test"))
    parser.add_argument("--max-instances", type=int, default=6)
    parser.add_argument("--instance-ids", type=str, nargs="*", default=None)
    parser.add_argument(
        "--instances-file",
        type=Path,
        default=None,
        help=(
            "Run instances from a local JSONL instead of the dataset. The "
            "analogue of ODR's tests/questions/: a set chosen to share content "
            "(same repo, neighbouring modules) is what makes cross-instance "
            "reuse measurable on purpose."
        ),
    )
    parser.add_argument(
        "--completions-per-instance",
        type=int,
        default=TARGET_COMPLETIONS_PER_INSTANCE,
    )
    parser.add_argument("--ablation-mode", type=str, default=None)
    parser.add_argument("--pseudo-dynamic", choices=("on", "off"), default=None)
    return parser.parse_args()


def _validate_benchmark_ablation_mode(mode: str) -> str:
    if mode in ABLATION_DISABLED_MODES:
        return mode
    if mode not in VALID_BENCHMARK_ABLATION_MODES:
        valid = ", ".join(sorted(VALID_BENCHMARK_ABLATION_MODES))
        disabled = ", ".join(sorted(ABLATION_DISABLED_MODES))
        raise ValueError(
            "run_evaluate_prefix.py expects LANGGRAPH_ABLATION_MODE to be one "
            f"of {valid} (or {disabled} to disable the ablation study)."
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
        self.series.append(
            {
                "task_id": payload["id"],
                "node_name": started["node_name"],
                "namespace": started["namespace"],
                "invocation_index": started["invocation_index"],
                "duration_ns": duration_ns,
                "duration_seconds": duration_ns / 1_000_000_000,
                "error": _normalize_json_error(payload.get("error")),
                "interrupt_count": len(payload.get("interrupts", [])),
            }
        )
        self.totals_by_name_ns[started["node_name"]] += duration_ns

    def build_summary(self) -> dict[str, Any]:
        now_ns = time.perf_counter_ns()
        unfinished_tasks = [
            {
                "task_id": task_id,
                "node_name": started["node_name"],
                "namespace": started["namespace"],
                "invocation_index": started["invocation_index"],
                "elapsed_ns_so_far": now_ns - started["started_at_ns"],
                "elapsed_seconds_so_far": (now_ns - started["started_at_ns"]) / 1_000_000_000,
            }
            for task_id, started in self._active.items()
        ]
        totals_by_node_name_ns = dict(sorted(self.totals_by_name_ns.items()))
        return {
            "time_series": self.series,
            "totals_by_node_name_ns": totals_by_node_name_ns,
            "totals_by_node_name_seconds": {
                key: value / 1_000_000_000 for key, value in totals_by_node_name_ns.items()
            },
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


async def target(
    instance: dict[str, Any],
    *,
    logical_job_index: int,
    instance_index: int,
):
    graph = swe_agent_builder.compile(checkpointer=MemorySaver(), _is_root=True)
    job_id = str(next(job_counter))
    thread_id = str(uuid.uuid4())
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started_at_ns = time.perf_counter_ns()

    config = {
        "configurable": {
            "thread_id": thread_id,
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
            "seed": seed,
        },
        "tags": [
            f"provider:{MODEL_PROVIDER}",
            f"model:{_slugify_model(MODEL_NAME)}",
            f"ablation:{ABLATION_MODE}",
            f"run:{RUN_ID}",
        ],
        # One handler for the whole job. The callback manager carries it into
        # both subgraphs, so a single attachment here sees every prompt the job
        # sends.
        "callbacks": [PrefixCaptureHandler(PREFIX_ANALYZER, job_id=job_id)],
    }

    collector = TimingCollector()
    setup_error: str | None = None
    stream_error: str | None = None
    state_error: str | None = None
    final_state: dict[str, Any] | None = None
    patch = ""

    try:
        materialize_workspace(instance)
    except Exception as exc:  # noqa: BLE001
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
            try:
                patch = collect_patch()
            except Exception as exc:  # noqa: BLE001
                state_error = state_error or f"patch_collect: {exc}"

    finished_at_ns = time.perf_counter_ns()
    finished_at_utc = datetime.now(timezone.utc).isoformat()
    query_duration_ns = finished_at_ns - started_at_ns
    summary = collector.build_summary()
    node_metrics_path = NODE_METRICS_DIR / f"job_{job_id}_thread_{thread_id}.jsonl"

    append_prediction(PREDICTIONS_PATH, instance, patch, f"{EXPERIMENT_NAME}-inst{instance_index}")

    base_record = {
        "run_id": RUN_ID,
        "swe_instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "logical_job_index": logical_job_index,
        "instance_index": instance_index,
        "thread_id": thread_id,
        "job_id": job_id,
        "started_at": started_at_utc,
        "finished_at": finished_at_utc,
    }

    for event in summary["time_series"]:
        _append_jsonl(node_metrics_path, {**base_record, **event})

    _append_jsonl(
        SUMMARY_PATH,
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

    # Free this job's prompt pool, then re-emit the aggregate files. The dump
    # is a full rewrite, which is why it happens once per job and not once per
    # turn -- and why a run killed halfway still leaves the totals for every
    # job that finished.
    PREFIX_ANALYZER.close_job(job_id)
    PREFIX_ANALYZER.dump_job(job_id)
    PREFIX_ANALYZER.dump_run()

    success = setup_error is None and stream_error is None and state_error is None
    return {
        "success": success,
        "run_id": RUN_ID,
        "swe_instance_id": instance["instance_id"],
        "logical_job_index": logical_job_index,
        "instance_index": instance_index,
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


async def _run_benchmark(
    selected: list[dict[str, Any]],
    *,
    completions_per_instance: int,
) -> dict[str, Any]:
    """Sequential, and here that is forced rather than chosen.

    All jobs share one `./workspace_repo` because the agent hard-codes that
    path, so two jobs at once would edit each other's checkout. ODR's
    `--max-concurrency` has no equivalent until the workspace is per job -- see
    plan/02-required-changes.md.
    """
    rows: list[dict[str, Any]] = []
    total = completions_per_instance * len(selected)
    position = 0
    for instance_index in range(1, completions_per_instance + 1):
        for job_index, instance in enumerate(selected, start=1):
            position += 1
            print(f"[{position}/{total}] {instance['instance_id']} run {instance_index}")
            result = await target(
                instance,
                logical_job_index=job_index,
                instance_index=instance_index,
            )
            rows.append(result)
            _append_csv_row(E2E_LATENCY_CSV_PATH, _e2e_csv_row(result))

    failed = [row for row in rows if not row["success"]]
    return {
        "run_id": RUN_ID,
        "sampled_instances": len(selected),
        "completions_per_instance": completions_per_instance,
        "completed_runs": len(rows),
        "failed_runs": len(failed),
        "empty_patch_runs": sum(1 for row in rows if row["empty_patch"]),
        "metrics_dir": str(METRICS_DIR),
        "prediction_logs_dir": str(PREDICTION_DIR),
        "instance_summary_jsonl": str(SUMMARY_PATH),
        "e2e_latency_csv": str(E2E_LATENCY_CSV_PATH),
        "predictions_jsonl": str(PREDICTIONS_PATH),
    }


async def main():
    global ABLATION_MODE, EXPERIMENT_NAME
    args = parse_args()

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
                f"swe-prefix-{MODEL_PROVIDER}-{_slugify_model(MODEL_NAME)}-"
                f"{ABLATION_MODE}-{RUN_ID}"
            )
            os.environ["LANGSMITH_PROJECT"] = EXPERIMENT_NAME
            os.environ["LANGCHAIN_PROJECT"] = EXPERIMENT_NAME

    _validate_benchmark_ablation_mode(ABLATION_MODE)
    selected = _select_instances(args)

    print(f"Metrics directory: {METRICS_DIR}")
    print(f"Prefix analysis directory: {PREFIX_ANALYSIS_DIR}")
    print(f"Prefix tokenizer: {PREFIX_ANALYZER.tokenizer_label}")
    print(f"Predictions JSONL: {PREDICTIONS_PATH}")
    print(f"Model: {MODEL_PROVIDER}:{MODEL_NAME}")
    print(f"LangSmith project: {EXPERIMENT_NAME}")
    print(f"Ablation mode: {ABLATION_MODE}")
    print(
        "Pseudo-dynamic prefix fill: "
        f"{'enabled' if _pseudo_dynamic_enabled() else 'DISABLED'}"
    )
    print(
        f"Running {len(selected)} instances sequentially, "
        f"{args.completions_per_instance} run(s) each"
    )

    results = await _run_benchmark(
        selected,
        completions_per_instance=args.completions_per_instance,
    )
    results.update(PREFIX_ANALYZER.dump_run())

    # The engine's own account of what this run reused and from which job. Read
    # rather than reset, so it can also be polled mid-run. Never fatal: a
    # server without VLLM_KV_PROVENANCE=1 (or no server at all) leaves the rest
    # of the run's outputs untouched and says so in the snapshot file.
    kv_reuse_snapshot_path = METRICS_DIR / "kv_reuse_snapshot.json"
    kv_reuse_snapshot = await snapshot_kv_metrics(record_to=kv_reuse_snapshot_path)
    results["kv_reuse_snapshot_json"] = str(kv_reuse_snapshot_path)
    results["kv_reuse_snapshot_ok"] = bool(kv_reuse_snapshot.get("ok"))
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)
