"""Helpers for forwarding LangGraph execution metadata to model requests."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.pregel._vllm_agent import (
    AGENT_UNIT_METADATA_KEY,
    derive_agent_id,
    derive_call_type,
    graph_path_from_checkpoint_ns,
    vllm_agent_enabled,
)


def inject_langgraph_request_metadata(
    config: RunnableConfig | None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge LangGraph request metadata into an OpenAI-compatible extra_body payload."""
    payload = dict(extra_body or {})

    if not isinstance(config, dict):
        return payload

    job_id: str | None = None
    configurable = config.get("configurable")
    if isinstance(configurable, dict):
        raw_job_id = configurable.get("job_id")
        if isinstance(raw_job_id, str) and raw_job_id:
            job_id = raw_job_id
            if "job_id" not in payload:
                payload["job_id"] = raw_job_id

    metadata = config.get("metadata")
    if isinstance(metadata, dict):
        node_name = metadata.get("langgraph_node")
        if isinstance(node_name, str) and node_name and "langgraph_node" not in payload:
            payload["langgraph_node"] = node_name
        if "call_type" not in payload:
            # Third element of vLLM's KV-eviction index key, alongside job_id
            # and langgraph_node. Sent unconditionally: it costs one short
            # string per request, and a request that arrives without it is
            # indexed under the empty label, which silently fails to join the
            # PROB/HISTORY rows that do carry one.
            #
            # These ride to vLLM as top-level extra_body fields, which its
            # OpenAI layer allows (extra="allow") and folds into
            # sampling_params.extra_args via model_extra — the same path
            # job_id and langgraph_node already take. No vLLM-side change.
            call_type = derive_call_type(cast(str | None, node_name))
            if isinstance(call_type, str) and call_type:
                payload["call_type"] = call_type
        if vllm_agent_enabled() and "agent_id" not in payload:
            # The prefetch registry key: one bucket per (job, graph path,
            # parallel unit), so this call's recorded prefix is looked up again
            # only by a prefetch for *this* place in *this* run — not by every
            # node that happens to share the name `researcher`. vLLM's
            # `auto_register.py` records under this id verbatim; the eviction
            # index key below (job_id / langgraph_node / call_type) is
            # unchanged and stays keyed on the bare node.
            agent_id = derive_agent_id(
                graph_path_from_checkpoint_ns(
                    cast(str | None, metadata.get("langgraph_checkpoint_ns")),
                    cast(str | None, node_name),
                ),
                job_id=job_id,
                unit=cast(
                    "int | str | None", metadata.get(AGENT_UNIT_METADATA_KEY)
                ),
            )
            if isinstance(agent_id, str) and agent_id:
                payload["agent_id"] = agent_id
                payload.setdefault("record_in_registry", True)

    return payload
