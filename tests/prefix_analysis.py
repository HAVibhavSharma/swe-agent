"""Longest-repeating-prefix analysis, client side.

Lifted from ``open_deep_research/tests/run_evaluate_prefix.py`` and factored
into its own module (ODR keeps it inline; there is nothing workload-specific in
it, and one copy is easier to keep honest than two).

Every LLM call the agent issues is a "turn", and every turn answers two
questions.

1. How much of this prompt has this node already sent before? That is the
   longest common prefix between this prompt and the best-matching *earlier
   prompt from the same node* in the same job. Scoped to one node on purpose:
   a `conduct_research` prompt and a `creating_diffs_for_task` prompt start
   with different system text, so they diverge at character zero and comparing
   them measures nothing.

2. Where did control come from? The node that ran immediately before this one
   inside the same graph instance -- the `from-node`. The node issuing the
   prompt is the `to-node`.

So `tools:conduct_research` with a 40KB prefix reads as: research was re-entered
from its tool node, and 40KB of what it re-sent it had already sent itself.
`__start__:come_up_with_research_next_step` with 0 is a genuinely cold entry.

The from-node cannot be read off `langgraph_triggers`: LangGraph names trigger
channels after their destination (`branch:to:<node>`), so a trigger says where
control went, not where it came from. It is tracked here off `on_chain_start`,
which fires for every node -- including the two ToolNodes, which issue no
prompts of their own and would otherwise be invisible.

Outputs, all keyed by the prefix's sha256:
  prefix_edge_counts.json  hash -> {"from:to": count}
  prefix_texts.json        hash -> the prefix text itself
  prefix_totals.json       hash -> {count, direct_count, prefix_tokens,
                           prefix_chars}; `direct_count` sums the edge dict,
                           `count` rolls in every prefix that contains this one
  prefix_containment.json  hash -> {parent, delta_chars, delta_tokens, ...},
                           the non-overlapping view (prefixes nest, so summing
                           their sizes double-counts)
plus prefix_turns.jsonl, one line per turn, so a crashed run still leaves a
reconstructable record.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.pregel._vllm_agent import AGENT_UNIT_METADATA_KEY

PREFIX_EDGE_COUNTS_FILENAME = "prefix_edge_counts.json"
PREFIX_TEXTS_FILENAME = "prefix_texts.json"
PREFIX_TOTALS_FILENAME = "prefix_totals.json"
PREFIX_CONTAINMENT_FILENAME = "prefix_containment.json"
PREFIX_TURNS_FILENAME = "prefix_turns.jsonl"

# The from-node for a node that is the first to run in its graph instance.
PREFIX_FROM_START = "__start__"
PREFIX_NODE_UNKNOWN = "__unknown__"

# LangGraph's own namespace separators, from
# langgraph/_internal/_constants.py. A task's checkpoint namespace is
# `{parent_ns}|{node}:{task_id}`, so stripping the last `|` segment yields the
# graph instance the node ran inside -- the scope control flow is sequential
# in, and therefore the scope a predecessor is meaningful in.
LANGGRAPH_NS_SEP = "|"
LANGGRAPH_NS_END = ":"
LANGGRAPH_PATH_SEP = "/"

# Tool call ids are random per job and appear in both the AIMessage that
# requested the tool and the ToolMessage that answered it. Keeping them makes
# the serialized prompt match what the server sees; dropping them makes
# prefixes comparable across jobs. Matching is per job, so keep them.
INCLUDE_TOOL_CALL_IDS = True


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _serialize_message(message: Any) -> str:
    """Render one message as text, tool calls included.

    ``get_buffer_string`` is not usable here: an AIMessage that only requests
    tools has empty ``content``, so it would serialize to nothing -- and the
    tool call payloads are exactly the part of the research prompt that grows
    turn over turn.
    """
    role = getattr(message, "type", None) or type(message).__name__
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = json.dumps(content, sort_keys=True, default=str)

    parts = [f"<|{role}|>", content]

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        normalized = []
        for tool_call in tool_calls:
            entry = dict(tool_call) if isinstance(tool_call, dict) else {"repr": str(tool_call)}
            if not INCLUDE_TOOL_CALL_IDS:
                entry.pop("id", None)
            normalized.append(entry)
        parts.append("<|tool_calls|>" + json.dumps(normalized, sort_keys=True, default=str))

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id and INCLUDE_TOOL_CALL_IDS:
        parts.append(f"<|tool_call_id|>{tool_call_id}")

    name = getattr(message, "name", None)
    if name:
        parts.append(f"<|name|>{name}")

    return "\n".join(parts)


def _serialize_prompt(messages: Any) -> str:
    return "\n".join(_serialize_message(message) for message in messages)


def _longest_common_prefix_len(left: str, right: str) -> int:
    """Length of the longest common prefix of two strings.

    Binary search over slice equality: each comparison runs at C speed, so this
    stays fast on the 100KB+ prompts a research loop accumulates, where a
    character-at-a-time Python loop would not.
    """
    limit = min(len(left), len(right))
    if limit == 0 or left[0] != right[0]:
        return 0
    if left[:limit] == right[:limit]:
        return limit

    low, high = 1, limit - 1
    while low < high:
        mid = (low + high + 1) // 2
        if left[:mid] == right[:mid]:
            low = mid
        else:
            high = mid - 1
    return low


def _graph_instance_key(metadata: dict[str, Any]) -> str:
    """Name the graph instance a node ran inside.

    Control flow is sequential within one instance, which is what makes "the
    node that ran before this one" well defined. The checkpoint namespace is
    ``{parent_ns}|{node}:{task_id}``, so dropping the last segment leaves the
    enclosing instance and puts sibling nodes -- a research node and its tool
    node -- in the same bucket.
    """
    namespace = metadata.get("langgraph_checkpoint_ns") or ""
    parent = namespace.rsplit(LANGGRAPH_NS_SEP, 1)[0] if LANGGRAPH_NS_SEP in namespace else ""
    unit = metadata.get(AGENT_UNIT_METADATA_KEY)
    return f"{parent}#unit{unit}" if unit is not None else parent


def _graph_path(metadata: dict[str, Any]) -> str:
    """Where the node sits in the graph, as ``parent/child/grandchild``.

    Built from the checkpoint namespace, whose every segment is
    ``{node}:{task_id}``; dropping the task ids leaves the chain of node names.
    A bare name is ambiguous once subgraphs are involved -- the path is what
    says a research node ran under `swe_developer` and not under
    `swe_architect`.
    """
    namespace = metadata.get("langgraph_checkpoint_ns") or ""
    if not namespace:
        return metadata.get("langgraph_node") or PREFIX_NODE_UNKNOWN
    segments = [
        segment.rsplit(LANGGRAPH_NS_END, 1)[0]
        for segment in namespace.split(LANGGRAPH_NS_SEP)
        if segment
    ]
    return LANGGRAPH_PATH_SEP.join(segment for segment in segments if segment)


def _node_label(metadata: dict[str, Any]) -> str:
    """The graph path, numbered when the path names more than one live node.

    This agent has no parallel fan-out, so the unit is always absent and the
    label is the path. Kept so the output schema matches ODR's and the same
    analysis scripts read both.
    """
    path = _graph_path(metadata)
    unit = metadata.get(AGENT_UNIT_METADATA_KEY)
    return f"{path}#{unit}" if unit is not None else path


def _hash_prefix(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Characters are what the prefix search works in, but tokens are what a KV
# cache is measured in, so the totals file reports both. Which tokenizer is
# used is recorded rather than assumed: a token count from the wrong vocabulary
# is worse than no token count, so when nothing loads this reports null.
_TOKENIZER_CACHE: tuple[Any, str] | None = None


def _load_tokenizer(model_name: str) -> tuple[Any, str]:
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is not None:
        return _TOKENIZER_CACHE

    name = os.getenv("PREFIX_TOKENIZER", "").strip() or model_name
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        # Prefixes routinely exceed the model's context window and the only
        # consequence here is a warning per call. Silence it.
        tokenizer.model_max_length = int(1e9)
        _TOKENIZER_CACHE = (
            lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"]),
            f"transformers:{name}",
        )
        return _TOKENIZER_CACHE
    except Exception:
        pass

    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(name)
            label = f"tiktoken:{name}"
        except Exception:
            encoding = tiktoken.get_encoding("o200k_base")
            label = "tiktoken:o200k_base"
        _TOKENIZER_CACHE = (
            lambda text: len(encoding.encode(text, disallowed_special=())),
            label,
        )
        return _TOKENIZER_CACHE
    except Exception:
        pass

    _TOKENIZER_CACHE = (None, "unavailable")
    return _TOKENIZER_CACHE


def _containment_payload(
    texts: dict[str, str],
    token_counts: dict[str, int | None],
) -> dict[str, Any]:
    """Decompose nested prefixes into non-overlapping segments.

    Recorded prefixes nest: a loop's turn-3 prefix literally begins with its
    turn-2 prefix. Summing their lengths counts the same bytes many times. This
    walks the containment tree and reports, for each prefix, only what it adds
    on top of the longest prefix it contains -- so the deltas partition the
    bytes and adding them up is meaningful.

    ``delta_tokens`` is a difference of token counts, not the token count of
    the delta text: tokens do not split cleanly at an arbitrary character
    boundary, and the difference is the figure that matters anyway -- what the
    prefix costs to prefill given its parent is already cached.
    """
    digests = sorted((d for d, t in texts.items() if t), key=lambda d: texts[d])

    payload: dict[str, Any] = {}
    stack: list[str] = []
    for digest in digests:
        text = texts[digest]
        while stack and not text.startswith(texts[stack[-1]]):
            stack.pop()
        parent = stack[-1] if stack else None
        parent_chars = len(texts[parent]) if parent else 0
        tokens = token_counts.get(digest)
        parent_tokens = token_counts.get(parent) if parent else 0
        payload[digest] = {
            "parent": parent,
            "prefix_chars": len(text),
            "prefix_tokens": tokens,
            "delta_chars": len(text) - parent_chars,
            "delta_tokens": (
                None if tokens is None or parent_tokens is None else tokens - parent_tokens
            ),
        }
        stack.append(digest)
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)
    temporary.replace(path)


class PrefixAnalyzer:
    """Accumulate longest-repeating-prefix statistics across a whole run."""

    def __init__(self, *, run_id: str, analysis_dir: Path, model_name: str) -> None:
        self._lock = Lock()
        self._run_id = run_id
        self._analysis_dir = analysis_dir
        self._model_name = model_name
        # (job_id, node_path) -> [(turn_index, node_label, serialized_prompt)]
        # One pool per node: that is the only history whose prefixes a node can
        # actually re-send. Dropped when the job ends.
        self._node_pools: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
        # (job_id, graph instance) -> {"last": label, "previous": label}
        self._lineage: dict[tuple[str, str], dict[str, Any]] = {}
        self._edges: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self._totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Global rather than per job: a hash *is* the text, so one copy is
        # always right and per-job copies would only duplicate.
        self._texts: dict[str, str] = {}
        self._token_counts: dict[str, int | None] = {}
        self._turn_count = 0

    @property
    def tokenizer_label(self) -> str:
        return _load_tokenizer(self._model_name)[1]

    def job_dir(self, job_id: str) -> Path:
        return self._analysis_dir / f"job_{job_id}"

    def _count_tokens(self, text: str) -> int | None:
        count_fn, _ = _load_tokenizer(self._model_name)
        if count_fn is None:
            return None
        try:
            return count_fn(text)
        except Exception:
            return None

    def _tokens_for(self, digest: str, text: str) -> int | None:
        """Token count for a prefix, computed at most once per hash.

        Deliberately not called under the lock: tokenising is the slowest thing
        here, and two threads racing on a new digest just do the work twice and
        agree on the answer.
        """
        with self._lock:
            if digest in self._token_counts:
                return self._token_counts[digest]
        tokens = self._count_tokens(text)
        with self._lock:
            self._token_counts[digest] = tokens
        return tokens

    def note_node_start(self, *, job_id: str, instance: str, label: str) -> None:
        """Record that ``label`` began running inside ``instance``.

        Fires many times per node (the node's runnable, then the model binding
        inside it, all with the same namespace). Only a change of label
        advances the lineage, which makes the repeats harmless.
        """
        with self._lock:
            state = self._lineage.setdefault(
                (job_id, instance), {"last": None, "previous": None, "calls": 0}
            )
            if state["last"] != label:
                state["previous"] = state["last"]
                state["last"] = label
                state["calls"] = 0

    def observe(
        self,
        *,
        job_id: str,
        instance: str,
        label: str,
        path: str,
        node: str,
        prompt_text: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Stamped before the lock, so the time is when the prompt was issued
        # rather than when this turn got its turn at the lock.
        observed_at_utc = datetime.now(timezone.utc).isoformat()
        observed_at_ns = time.perf_counter_ns()

        with self._lock:
            state = self._lineage.get((job_id, instance))
            # Only trust the lineage if it agrees this node is the one
            # currently running; anything else means the chain callback did not
            # fire as expected, and a wrong edge is worse than __start__.
            if state and state["last"] == label and state["previous"]:
                from_label = state["previous"]
            else:
                from_label = PREFIX_FROM_START

            if state is not None and state["last"] == label:
                state["calls"] += 1
                call_index = state["calls"]
            else:
                call_index = 1

            pool = self._node_pools[(job_id, path)]

            # Scan newest first and take a strict improvement, so ties resolve
            # to the most recent prompt -- the one a real cache would still
            # have resident.
            best_length = 0
            best_turn: int | None = None
            best_label: str | None = None
            for prior_turn, prior_label, prior_text in reversed(pool):
                length = _longest_common_prefix_len(prompt_text, prior_text)
                if length > best_length:
                    best_length = length
                    best_turn = prior_turn
                    best_label = prior_label

            prefix_text = prompt_text[:best_length]
            digest = _hash_prefix(prefix_text)
            edge = f"{from_label}:{label}"

            self._edges[job_id][digest][edge] += 1
            self._totals[job_id][digest] += 1
            self._texts.setdefault(digest, prefix_text)
            self._turn_count += 1
            turn_index = self._turn_count

            pool.append((turn_index, label, prompt_text))
            pool_size = len(pool)

        prefix_tokens = self._tokens_for(digest, prefix_text)

        record = {
            "run_id": self._run_id,
            "job_id": job_id,
            "turn_index": turn_index,
            # Wall clock for lining rows up against anything outside this
            # process. The perf counter is monotonic and shares its origin with
            # node_metrics/*.jsonl, so it is the one to subtract when timing
            # turns against each other.
            "observed_at": observed_at_utc,
            "observed_at_ns": observed_at_ns,
            "from_node": from_label,
            "to_node": label,
            "to_node_call_index": call_index,
            "edge": edge,
            "to_node_name": node,
            "to_node_path": path,
            "prefix_sha256": digest,
            "prefix_chars": best_length,
            "prefix_tokens": prefix_tokens,
            "prompt_chars": len(prompt_text),
            "prefix_fraction": (best_length / len(prompt_text)) if prompt_text else 0.0,
            "prefix_source_turn": best_turn,
            "prefix_source_node": best_label,
            "node_prompts_seen": pool_size,
            "graph_instance": instance,
            **(extra or {}),
        }
        _append_jsonl(self.job_dir(job_id) / PREFIX_TURNS_FILENAME, record)
        return record

    def close_job(self, job_id: str) -> None:
        """Release the job's prompt pools. Its counts are kept for the merge."""
        with self._lock:
            for key in [key for key in self._node_pools if key[0] == job_id]:
                del self._node_pools[key]
            for key in [key for key in self._lineage if key[0] == job_id]:
                del self._lineage[key]

    def _write_stats(
        self,
        directory: Path,
        edges: dict[str, dict[str, int]],
        counts: dict[str, int],
    ) -> dict[str, str]:
        with self._lock:
            texts = {digest: self._texts.get(digest, "") for digest in counts}
            token_counts = {digest: self._token_counts.get(digest) for digest in counts}

        # A zero-token prefix is not a prefix -- it is the record of a turn that
        # reused nothing, and it collects every cold entry in the run under one
        # hash. Dropped from the hash-keyed files; every such turn is still a
        # row in prefix_turns.jsonl with prefix_chars 0.
        ordered = [
            item
            for item in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if not (token_counts[item[0]] == 0 or not texts[item[0]])
        ]
        if not ordered:
            return {}
        edge_payload = {
            digest: dict(sorted(edges.get(digest, {}).items())) for digest, _ in ordered
        }
        text_payload = {digest: texts[digest] for digest, _ in ordered}
        containment = _containment_payload(text_payload, token_counts)

        # A prefix is present in every prompt that contains it, so a parent's
        # own count understates how often it was re-sent. Roll each entry's
        # count up into its parent. Longest first, so a prefix has already
        # collected everything from its descendants before handing the total
        # up -- a child is strictly longer than its parent, which makes that
        # ordering exact.
        cumulative = {digest: count for digest, count in ordered}
        for digest in sorted(containment, key=lambda item: -containment[item]["prefix_chars"]):
            parent = containment[digest]["parent"]
            if parent:
                cumulative[parent] += cumulative[digest]

        totals_payload = {
            digest: {
                "count": cumulative[digest],
                "direct_count": count,
                "prefix_tokens": token_counts[digest],
                "prefix_chars": len(texts[digest]),
            }
            for digest, count in sorted(
                ordered, key=lambda item: (-cumulative[item[0]], item[0])
            )
        }

        _write_json(directory / PREFIX_EDGE_COUNTS_FILENAME, edge_payload)
        _write_json(directory / PREFIX_TEXTS_FILENAME, text_payload)
        _write_json(directory / PREFIX_TOTALS_FILENAME, totals_payload)
        _write_json(directory / PREFIX_CONTAINMENT_FILENAME, containment)
        return {
            "prefix_edge_counts_json": str(directory / PREFIX_EDGE_COUNTS_FILENAME),
            "prefix_texts_json": str(directory / PREFIX_TEXTS_FILENAME),
            "prefix_totals_json": str(directory / PREFIX_TOTALS_FILENAME),
            "prefix_containment_json": str(directory / PREFIX_CONTAINMENT_FILENAME),
        }

    def dump_job(self, job_id: str) -> dict[str, str]:
        """Write one job's files into its own directory."""
        with self._lock:
            edges = {digest: dict(by_edge) for digest, by_edge in self._edges[job_id].items()}
            counts = dict(self._totals[job_id])
        if not counts:
            return {}
        paths = self._write_stats(self.job_dir(job_id), edges, counts)
        paths["prefix_turns_jsonl"] = str(self.job_dir(job_id) / PREFIX_TURNS_FILENAME)
        return paths

    def dump_run(self) -> dict[str, str]:
        """Write the merge of every job into the run directory."""
        merged_edges: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        merged_counts: dict[str, int] = defaultdict(int)
        with self._lock:
            for by_digest in self._edges.values():
                for digest, by_edge in by_digest.items():
                    for edge, count in by_edge.items():
                        merged_edges[digest][edge] += count
            for by_digest_counts in self._totals.values():
                for digest, count in by_digest_counts.items():
                    merged_counts[digest] += count
        if not merged_counts:
            return {}
        return self._write_stats(
            self._analysis_dir,
            {digest: dict(by_edge) for digest, by_edge in merged_edges.items()},
            dict(merged_counts),
        )


class PrefixCaptureHandler(BaseCallbackHandler):
    """Feed every prompt LangGraph sends to a model into the analyzer.

    Attached once at the top of a graph run: the callback manager propagates it
    down through both subgraphs, and ``metadata["langgraph_node"]`` names the
    node that issued each prompt.
    """

    # Run in the callback thread rather than being handed to an executor, so
    # the pool sees prompts in issue order.
    run_inline = True

    def __init__(self, analyzer: PrefixAnalyzer, *, job_id: str) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._job_id = job_id

    def _record(self, prompt_text: str, metadata: dict[str, Any] | None) -> None:
        metadata = metadata or {}
        self._analyzer.observe(
            job_id=self._job_id,
            instance=_graph_instance_key(metadata),
            label=_node_label(metadata),
            path=_graph_path(metadata),
            node=metadata.get("langgraph_node") or PREFIX_NODE_UNKNOWN,
            prompt_text=prompt_text,
            extra={
                "langgraph_step": metadata.get("langgraph_step"),
                "langgraph_triggers": _normalize_json(metadata.get("langgraph_triggers")),
                "langgraph_checkpoint_ns": metadata.get("langgraph_checkpoint_ns"),
                "research_unit": metadata.get(AGENT_UNIT_METADATA_KEY),
            },
        )

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        # Fires for every node, prompting or not. This is the only place the
        # two ToolNodes are ever seen, and they are the predecessors that
        # matter for the react loops.
        metadata = kwargs.get("metadata") or {}
        if not metadata.get("langgraph_node"):
            return
        self._analyzer.note_node_start(
            job_id=self._job_id,
            instance=_graph_instance_key(metadata),
            label=_node_label(metadata),
        )

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        for batch in messages or []:
            self._record(_serialize_prompt(batch), kwargs.get("metadata"))

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        for prompt in prompts or []:
            self._record(prompt, kwargs.get("metadata"))
