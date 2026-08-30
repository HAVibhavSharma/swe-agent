"""The single place a model call's kwargs are decided.

Adapted from ``open_deep_research.utils.get_model_config``. The upstream agent
constructed ``ChatAnthropic(model="claude-sonnet-4-20250514")`` at import time
in eight places, which makes three things impossible:

* pointing the workload at a local vLLM (the whole reason this repo is a
  benchmark workload rather than a product),
* attaching ``extra_body`` -- the channel that carries ``job_id`` /
  ``langgraph_node`` / ``call_type`` / ``agent_id`` to vLLM, and
* varying max_tokens per call site.

So the model is built per call from the node's ``RunnableConfig``. That config
is what carries ``metadata["langgraph_node"]``, so it must reach this function
or the request arrives at vLLM unlabelled and is indexed under the empty
label -- silently failing to join anything.

``MODEL_PROVIDER=openai`` (default) covers the local vLLM via
``OPENAI_BASE_URL``; ``anthropic`` keeps the original Claude path for
debugging, minus the vLLM-only knobs the Anthropic client rejects.
"""

from __future__ import annotations

import logging
import os

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

from agent.common import trace_store
from agent.common.configuration import Configuration
from agent.common.llm_request_metadata import inject_langgraph_request_metadata

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv(
    "OPENAI_MODEL", "Qwen/Qwen2.5-72B-Instruct-AWQ"
)

# Per-call-site output budgets. Named after the node that spends them so a run
# that truncates says which prompt to look at.
PLAN_MAX_TOKENS = int(os.getenv("PLAN_MODEL_MAX_TOKENS", "4096"))
RESEARCH_MAX_TOKENS = int(os.getenv("RESEARCH_MODEL_MAX_TOKENS", "8192"))
EXTRACT_PLAN_MAX_TOKENS = int(os.getenv("EXTRACT_PLAN_MODEL_MAX_TOKENS", "16384"))
DIFF_MAX_TOKENS = int(os.getenv("DIFF_MODEL_MAX_TOKENS", "16384"))
NEW_FILE_MAX_TOKENS = int(os.getenv("NEW_FILE_MODEL_MAX_TOKENS", "16384"))


def _disable_vllm_agent_for_closed_source() -> None:
    """Keep the prefetch path off whenever the model is not served by vLLM.

    ``vllm_agent_enabled()`` in the fork is purely env-driven and has no
    provider awareness -- it would happily fire prefetch at localhost while the
    traffic goes to Anthropic, warming a cache nothing reads. Same guard as
    ``open_deep_research.utils``.
    """
    if MODEL_PROVIDER == "openai":
        return
    cleared = [
        var
        for var in ("LANGGRAPH_VLLM_AGENT_ENABLE", "LANGGRAPH_VLLM_AGENT_BASE_URL")
        if os.environ.pop(var, None) is not None
    ]
    if cleared:
        logging.warning(
            "MODEL_PROVIDER=%s is not vLLM; disabling prefix prefetch (cleared %s).",
            MODEL_PROVIDER,
            ", ".join(cleared),
        )


_disable_vllm_agent_for_closed_source()

# One configurable model, re-configured per call. Declaring the fields here is
# what lets `.with_config()` change them later without rebuilding the client --
# the same pattern ODR uses (deep_researcher.py:74).
configurable_model = init_chat_model(
    configurable_fields=(
        "model",
        "model_provider",
        "max_tokens",
        "temperature",
        "api_key",
        "extra_body",
        "stream_usage",
        "seed",
        # Fixed-trace study: lets get_model_config() swap in the record/replay
        # httpx client per call without rebuilding the model.
        "http_async_client",
    ),
)


def _read_float_env(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        logging.warning("Invalid float for %s=%r; ignoring", name, raw)
        return None


def _temperature() -> float:
    value = _read_float_env("MODEL_TEMPERATURE")
    return 0.0 if value is None else value


def _extra_body(config: RunnableConfig | None) -> dict:
    """The prefetch/eviction hint channel.

    ``inject_langgraph_request_metadata`` adds job_id, langgraph_node,
    call_type and (when the prefetch is on) agent_id. The OpenAI client
    flattens this dict into the top level of the request body, and vLLM's
    OpenAI layer folds unknown top-level fields into
    ``sampling_params.extra_args`` -- so no server-side change is needed.
    """
    extra_body: dict = {}
    repetition_penalty = _read_float_env("OPENAI_REPETITION_PENALTY")
    if repetition_penalty is not None:
        extra_body["repetition_penalty"] = repetition_penalty
    return inject_langgraph_request_metadata(config, extra_body)


def get_model_config(
    config: RunnableConfig | None,
    max_tokens: int,
    *,
    provider: str | None = None,
) -> dict:
    """Provider-aware kwargs for one model call."""
    provider = provider or MODEL_PROVIDER
    model_config: dict = {
        "model": MODEL_NAME,
        "model_provider": provider,
        "max_tokens": max_tokens,
        "temperature": _temperature(),
        "tags": ["langsmith:nostream"],
    }
    if provider == "openai":
        # The fork forces stream_mode="messages" when the prediction worker is
        # on, so every call streams. ChatOpenAI turns stream_usage off for
        # non-official base URLs, which reports 0 tokens to LangSmith; vLLM
        # supports stream_options.include_usage, so turn it back on.
        model_config["stream_usage"] = True
        model_config["extra_body"] = _extra_body(config)
        seed = Configuration.from_runnable_config(config).seed
        if seed is not None:
            model_config["seed"] = seed

        # Fixed-trace study: route model traffic through the record/replay
        # transport. Returns None (and changes nothing) unless SWE_TRACE_MODE
        # (or ODR_TRACE_MODE) is set. See agent/common/trace_store.py.
        trace_client = trace_store.get_http_client()
        if trace_client is not None:
            model_config["http_async_client"] = trace_client
    return model_config


def chat_model(config: RunnableConfig | None, max_tokens: int):
    """The model to call from inside a node, labelled for this node's request."""
    return configurable_model.with_config(get_model_config(config, max_tokens))
