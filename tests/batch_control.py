"""How a batch of instances is driven: resume, subsets, isolation, interrupts.

Shared by ``run_evaluate.py`` and ``run_evaluate_node_eviction.py``, and
spelled the way ``open_deep_research``'s harness spells it, because pitlane
drives both workflows with the same flags:

``--completed-log``              record finished instances; skip them on a resume
``--only-questions``             run a 1-based subset of the sample, keeping each
                                 instance's full-run job id
``--hbm-flush-between-queries``  flush GPU HBM after every instance, not only
                                 before the first

An interrupt lands on an instance boundary: the first SIGINT/SIGTERM lets the
instance in flight finish and stops there, so a resume picks up at the next
one; a second exits at once.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from pathlib import Path
from typing import Any, Awaitable, Callable

try:
    from kv_metrics_reset import reset_kv_metrics
except ImportError:
    from tests.kv_metrics_reset import reset_kv_metrics


def read_completed_log(path: Path | None) -> tuple[set[int], int]:
    """``(instance indices already finished, highest job id used)`` from ``path``.

    The job id matters as much as the index. Ids come from a process-local
    counter, so a resumed run would restart it at 1 and relabel its instances
    with ids the earlier attempt already used -- and the id is what every
    per-request stats row carries, so two instances would be grouped as one.

    One JSON object per line, so a file cut off by the interrupt it exists to
    survive loses at most its last line, and losing a line only means running
    that instance again. An unreadable file means running everything.
    """
    if path is None or not path.exists():
        return set(), 0
    done: set[int] = set()
    highest = 0
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line
            if isinstance(entry, int):
                done.add(entry)
                continue
            index = entry.get("question")
            if isinstance(index, int):
                done.add(index)
            for job_id in entry.get("job_ids") or []:
                if str(job_id).isdigit():
                    highest = max(highest, int(job_id))
    except OSError:
        return set(), 0
    return done, highest


def append_completed_log(path: Path | None, index: int, job_ids: list[str]) -> None:
    """Note that instance ``index`` is finished, durably, right now."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps({"question": index, "job_ids": list(job_ids)}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        print(f"WARNING: could not record instance {index} as done: {exc}", flush=True)


def parse_only_questions(
    value: str | None, sample_size: int, completions_per_instance: int
) -> set[int] | None:
    """Indices for ``--only-questions``, checked against the sample they index."""
    if not value:
        return None
    try:
        picked = {int(part) for part in value.split(",") if part.strip()}
    except ValueError:
        raise SystemExit(f"--only-questions {value!r}: not a list of integers")
    outside = sorted(i for i in picked if not 1 <= i <= sample_size)
    if outside:
        raise SystemExit(f"--only-questions: {outside} outside the {sample_size}-instance sample")
    if completions_per_instance != 1:
        # An instance's job id is its index only when each runs once.
        raise SystemExit("--only-questions needs --completions-per-instance 1")
    return picked


RunOne = Callable[..., Awaitable[dict[str, Any]]]


async def run_batch(
    selected: list[dict[str, Any]],
    *,
    completions_per_instance: int,
    run_one: RunOne,
    on_row: Callable[[dict[str, Any]], None] = lambda row: None,
    completed_log: Path | None = None,
    only_questions: set[int] | None = None,
    flush_between: bool = False,
    set_next_job_id: Callable[[int], None] | None = None,
    label: str = "warm",
) -> dict[str, Any]:
    """Run every selected instance ``completions_per_instance`` times, in order.

    ``run_one(instance, logical_job_index=, instance_index=, fixed_job_id=)``
    runs one completion; ``on_row`` gets each result as it lands. Skipped instances -- finished on an earlier attempt,
    or outside ``only_questions`` -- keep their place, so every index (and with
    it every job id) is what an uninterrupted full run would have had.
    """
    already_done, highest_job_id = read_completed_log(completed_log)
    if highest_job_id and set_next_job_id is not None:
        # Continue the numbering rather than restarting it: reusing an id
        # would merge two instances' stats rows into one.
        set_next_job_id(highest_job_id + 1)
        print(f"resuming: job ids continue from {highest_job_id + 1}", flush=True)

    pending = [
        (index, instance)
        for index, instance in enumerate(selected, start=1)
        if index not in already_done
        and (only_questions is None or index in only_questions)
    ]
    if already_done:
        print(f"resuming: {len(already_done)} of {len(selected)} instance(s) "
              f"already done, running {len(pending)}", flush=True)

    stop_requested = False

    def request_stop() -> None:
        nonlocal stop_requested
        if stop_requested:
            print("\nsecond interrupt: exiting now, the instance in flight is lost", flush=True)
            raise KeyboardInterrupt
        stop_requested = True
        print("\ninterrupt received: finishing the instance in flight, then stopping. "
              "Finished instances are recorded and skipped on a resume. Press "
              "Ctrl-C again to stop immediately.", flush=True)

    loop = asyncio.get_running_loop()
    installed: list[int] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_stop)
            installed.append(signum)
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    rows: list[dict[str, Any]] = []
    flushes: list[dict[str, Any]] = []
    job_ids: dict[int, list[str]] = {index: [] for index, _ in pending}
    total = len(pending) * completions_per_instance
    position = 0
    try:
        # Round-robin, as the harness always ran: every instance once, then
        # every instance again. An instance counts as done after its last
        # round, so a resume repeats the rounds of one that had not finished.
        for instance_index in range(1, completions_per_instance + 1):
            for index, instance in pending:
                if stop_requested:
                    break
                position += 1
                print(f"[{label} {position}/{total}] {instance['instance_id']} run {instance_index}")
                row = await run_one(
                    instance,
                    logical_job_index=index,
                    instance_index=instance_index,
                    # The full run's id, so pinned replay finds the recording;
                    # the shared counter would number a subset 1..k.
                    fixed_job_id=str(index) if only_questions is not None else None,
                )
                rows.append(row)
                on_row(row)
                job_ids[index].append(str(row.get("job_id", "")))
                # Recorded before the flush: an instance marked done must need
                # no further work, or a resume would skip an unfinished one.
                if instance_index == completions_per_instance:
                    append_completed_log(completed_log, index, job_ids[index])
                if flush_between and position < total and not stop_requested:
                    flushes.append(await reset_kv_metrics(
                        f"between_queries_after_job_{row.get('job_id', index)}", flush_hbm=True,
                    ))
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)

    return {
        "rows": rows,
        "drained": stop_requested and position < total,
        "between_query_flushes": len(flushes),
        "between_query_flushes_ok": sum(1 for f in flushes if f and f.get("ok")),
        "skipped_already_done": len(already_done),
    }
