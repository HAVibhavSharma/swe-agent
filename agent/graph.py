from agent.architect.graph import swe_architect
from agent.common.entities import ImplementationPlan
from agent.developer.graph import swe_developer
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages, StateGraph, START, END
from typing import Annotated, Optional

class AgentState(BaseModel):
    implementation_research_scratchpad: Annotated[list[AnyMessage], add_messages]
    implementation_plan: Optional[ImplementationPlan] = Field(None, description="The implementation plan to be executed")


def create_workflow_graph():
    """Create and return the workflow graph with conditional routing"""
    # Initialize graph
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("swe_architect", swe_architect)
    graph_builder.add_node("swe_developer", swe_developer)
    # Add edges for the workflow
    graph_builder.add_edge(START, "swe_architect")
    graph_builder.add_edge("swe_architect", "swe_developer")
    graph_builder.add_edge("swe_developer", END)

    return graph_builder

# Exported for the benchmark harnesses, which compile it themselves with a
# checkpointer -- the same shape as ODR's `deep_researcher_builder`.
swe_agent_builder = create_workflow_graph()

# `_is_root=True` is a fork-only kwarg (langgraph-dev, branch
# Prediction-SWEbench). It is what makes the compiler walk this graph *and its
# subgraphs* to build `prompt_composition` and `transition_prediction`, the
# metadata the prefetch predictor runs on. The two subgraphs are already
# compiled at import time above, which is the order the analyser needs -- see
# plan/02-required-changes.md.
#
# On stock langgraph this raises TypeError; that is intentional. The workload
# has no meaning without the fork.
swe_agent = swe_agent_builder.compile(_is_root=True).with_config(
    {"tags": ["agent-v1"], "recursion_limit": 200}
)
