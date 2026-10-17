"""Check tests/batch_control.py without a server or a SWE-bench checkout.

Drives `run_batch` with a fake instance runner and a fake HBM flush, and
checks the four properties pitlane relies on: run order, resume from the
completed log, `--only-questions` keeping full-run job ids, and an interrupt
stopping at an instance boundary.

    python tests/check_batch_control.py
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import batch_control  # noqa: E402

FAILURES: list[str] = []
FLUSHES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {label}{f' -- {detail}' if detail and not ok else ''}")
    if not ok:
        FAILURES.append(label)


async def _fake_flush(label: str, *, flush_hbm: bool = True, **_) -> dict:
    FLUSHES.append(label)
    return {"ok": True, "hbm_flushed": flush_hbm}


batch_control.reset_kv_metrics = _fake_flush


class Harness:
    """Stands in for a run_evaluate script: a job counter and a target."""

    def __init__(self, interrupt_after: int | None = None) -> None:
        self.counter = itertools.count(1)
        self.calls: list[tuple[int, int, str]] = []
        self.interrupt_after = interrupt_after

    def continue_from(self, first: int) -> None:
        self.counter = itertools.count(first)

    async def run_one(self, instance, *, logical_job_index, instance_index, fixed_job_id):
        job_id = fixed_job_id or str(next(self.counter))
        self.calls.append((logical_job_index, instance_index, job_id))
        if self.interrupt_after is not None and len(self.calls) == self.interrupt_after:
            os.kill(os.getpid(), signal.SIGINT)
        # Long enough for the loop to run the signal handler, which a real
        # instance -- seconds of model calls -- always is.
        await asyncio.sleep(0.05)
        return {"job_id": job_id, "instance": instance["instance_id"]}


SAMPLE = [{"instance_id": f"repo__{i}"} for i in range(1, 6)]


async def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="batch-control-"))

    # Round-robin order and per-instance flushes.
    h = Harness()
    out = await batch_control.run_batch(
        SAMPLE[:3], completions_per_instance=2, run_one=h.run_one,
        flush_between=True, set_next_job_id=h.continue_from,
    )
    check("round-robin: every instance once, then again",
          [(i, r) for i, r, _ in h.calls] == [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)])
    check("a flush after every run but the last", len(FLUSHES) == 5, f"{len(FLUSHES)} flushes")
    check("not drained", out["drained"] is False)

    # Interrupt at an instance boundary, then resume.
    log = work / "completed.jsonl"
    h = Harness(interrupt_after=2)
    out = await batch_control.run_batch(
        SAMPLE, completions_per_instance=1, run_one=h.run_one,
        completed_log=log, set_next_job_id=h.continue_from,
    )
    check("interrupt finishes the instance in flight and stops",
          [i for i, _, _ in h.calls] == [1, 2], str(h.calls))
    check("drained is reported", out["drained"] is True)
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    check("finished instances are logged with their job ids",
          entries == [{"question": 1, "job_ids": ["1"]}, {"question": 2, "job_ids": ["2"]}],
          str(entries))

    h = Harness()
    out = await batch_control.run_batch(
        SAMPLE, completions_per_instance=1, run_one=h.run_one,
        completed_log=log, set_next_job_id=h.continue_from,
    )
    check("resume runs only the unfinished instances", [i for i, _, _ in h.calls] == [3, 4, 5])
    check("resume continues the job ids", [j for _, _, j in h.calls] == ["3", "4", "5"],
          str(h.calls))
    check("resume reports what it skipped", out["skipped_already_done"] == 2)

    # A subset keeps each instance's full-run job id.
    h = Harness()
    await batch_control.run_batch(
        SAMPLE, completions_per_instance=1, run_one=h.run_one,
        only_questions=batch_control.parse_only_questions("2,5", len(SAMPLE), 1),
    )
    check("--only-questions runs the subset with full-run ids",
          h.calls == [(2, 1, "2"), (5, 1, "5")], str(h.calls))
    for bad, why in (("0", "outside"), ("x", "not integers")):
        try:
            batch_control.parse_only_questions(bad, len(SAMPLE), 1)
            check(f"--only-questions {bad!r} is rejected ({why})", False)
        except SystemExit:
            check(f"--only-questions {bad!r} is rejected ({why})", True)
    try:
        batch_control.parse_only_questions("1", len(SAMPLE), 2)
        check("--only-questions needs one completion per instance", False)
    except SystemExit:
        check("--only-questions needs one completion per instance", True)

    print(f"\n{'all checks passed' if not FAILURES else f'{len(FAILURES)} FAILED'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
