"""Per-run knobs, read off the LangGraph ``RunnableConfig``.

Mirrors ``open_deep_research.configuration.Configuration``. Two reasons it
exists rather than the values being module constants:

1. The benchmark harnesses set them per job (``config["configurable"]``), so a
   run can sweep a limit without editing code.
2. The LangGraph fork's compile-time transition analyser recognises a loop
   bound written as ``state_key <op> configurable.<config_key>`` -- see
   ``langgraph/graph/state.py::_has_iteration_limit_condition``. Reading the
   bound off a ``configurable`` object is what turns "this react loop ends
   eventually" into a *predictable* transition the prefetch can act on. A bare
   module constant compiles to nothing.

The two names below are deliberately the ones ODR uses. The recogniser matches
them literally today, so reusing them buys working prediction rules with no
fork change; see plan/02-required-changes.md for the generalisation that would
remove that coupling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """Everything a job can vary without touching the graph."""

    # Architect: how many research turns (`conduct_research` visits) before the
    # plan is extracted with whatever has been found. Without a bound the
    # architect loop is open-ended -- the upstream agent relies solely on the
    # model deciding to stop, plus the root graph's recursion_limit=200.
    max_researcher_iterations: int = 6

    # Both react loops: how many consecutive tool round-trips one visit may
    # make before it is forced to produce its answer.
    max_react_tool_calls: int = 8

    # Developer: cap on atomic tasks executed per instance. A plan with 40
    # atomic tasks is a runaway, not a solution, and on a benchmark it burns
    # the run's wall clock on one instance.
    max_atomic_tasks: int = 20

    # Sampling seed, forwarded to vLLM when set. None means "do not send one".
    seed: int | None = None

    # Opaque per-instance id, echoed to vLLM in extra_body as the KV-eviction
    # and provenance key. The harnesses set it; nothing in the graph reads it.
    job_id: str | None = None

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
        """Build from ``config["configurable"]``, falling back to env then default.

        Env fallback uses the upper-cased field name, so
        ``MAX_REACT_TOOL_CALLS=4`` works for a one-off run without a harness.
        """
        configurable: dict[str, Any] = (
            config.get("configurable", {}) if isinstance(config, dict) else {}
        )
        values: dict[str, Any] = {}
        for field in fields(cls):
            raw = configurable.get(field.name, os.environ.get(field.name.upper()))
            if raw is None or raw == "":
                continue
            if field.type in ("int", "int | None") and isinstance(raw, str):
                try:
                    raw = int(raw)
                except ValueError:
                    continue
            values[field.name] = raw
        return cls(**values)
