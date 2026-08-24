import json
from typing import List, TypedDict, Optional

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END, START
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph

from agent.architect.state import SoftwareArchitectState
from agent.common.configuration import Configuration
from agent.common.model import (
    EXTRACT_PLAN_MAX_TOKENS,
    PLAN_MAX_TOKENS,
    RESEARCH_MAX_TOKENS,
    configurable_model,
    get_model_config,
)
from agent.tools.search import search_tools
from agent.tools.codemap import codemap_tools
from agent.tools.write import get_files_structure
from helpers.prompts import markdown_to_prompt_template
from agent.common.entities import ImplementationPlan


class ResearchStep(BaseModel):
    reasoning: str = Field(description="The reasoning behind the research step, why research is needed how it going to help the implmentation of the task")
    hypothesis: str = Field(description="The hypothesis that need to be researched")


class ResearchEvaluation(BaseModel):
    reasoning: str = Field(description="The reason why the research step is valid or not 1-3 sentences")
    is_valid: bool = Field(description="Whether the research step is valid")

# prompt
plan_next_step_prompt = markdown_to_prompt_template("agent/architect/prompts/plan_next_step_prompt.md")
check_research_prompt = markdown_to_prompt_template("agent/architect/prompts/check_research_already_explored.md")
conduct_research_prompt = markdown_to_prompt_template("agent/architect/prompts/conduct_research_plan_prompt.md")
extract_implementation_prompt = markdown_to_prompt_template("agent/architect/prompts/extract_implementation_plan.md")

# The model is *not* bound at import time any more. Every call builds it from
# the node's RunnableConfig, because that config is what carries
# metadata["langgraph_node"] into extra_body -- see agent/common/model.py.

research_tools = search_tools + codemap_tools

tool_node = ToolNode(codemap_tools+search_tools, messages_key="implementation_research_scratchpad")

class ComeUpWithResearchNextStepOutput(TypedDict):
    research_next_step: str
    implementation_research_scratchpad: List[AnyMessage]

def come_up_with_research_next_step(state: SoftwareArchitectState, config: RunnableConfig) -> ComeUpWithResearchNextStepOutput:
    """Generate the next research step based on the current state"""
    runnable = plan_next_step_prompt | configurable_model.with_structured_output(
        ResearchStep
    ).with_config(get_model_config(config, PLAN_MAX_TOKENS))
    response = runnable.invoke({
        "implementation_research_scratchpad": state.implementation_research_scratchpad,
        "codebase_structure": get_files_structure.invoke({
            "directory": "./workspace_repo"
        }),
    })
    return {"research_next_step": response.hypothesis,
            "implementation_research_scratchpad": [
                AIMessage(content=f"My next thing i need to check is {response.hypothesis}"
                          f"This is why I think it is useful: {response.reasoning}")]}

class CheckResearchStepOutput(TypedDict):
    is_valid_research_step: bool
    implementation_research_scratchpad: List[AnyMessage]

def check_research_step(state: SoftwareArchitectState, config: RunnableConfig)-> CheckResearchStepOutput:
    """Check if the proposed research step has already been explored"""
    runnable = check_research_prompt | configurable_model.with_structured_output(
        ResearchEvaluation
    ).with_config(get_model_config(config, PLAN_MAX_TOKENS))
    response = runnable.invoke({
        "implementation_research_scratchpad": state.implementation_research_scratchpad
    })
    if not response.is_valid:
        return {
            "is_valid_research_step": False,
            "implementation_research_scratchpad": [HumanMessage(content="The research path is not valid, here is why: " + response.reasoning)]
        }
    else:
        return {
            "is_valid_research_step": True,
            "implementation_research_scratchpad": [HumanMessage(content=f"The research path is valid, start conducting the research")]
        }

def conduct_research(state: SoftwareArchitectState, config: RunnableConfig):
    """Conduct research based on the proposed hypothesis"""
    runnable = conduct_research_prompt | configurable_model.bind_tools(
        research_tools
    ).with_config(get_model_config(config, RESEARCH_MAX_TOKENS))
    response = runnable.invoke({
        "implementation_research_scratchpad": state.implementation_research_scratchpad,
        "codebase_structure": get_files_structure.invoke({"directory": "./workspace_repo"})
    })
    # Both counters advance here rather than in the router: a router may be
    # called more than once for one visit, and a count that is not exactly
    # "turns taken" makes the compiled limit rule predict the wrong exit.
    return {
        "implementation_research_scratchpad": [response],
        "research_iterations": state.research_iterations + 1,
        "tool_call_iterations": state.tool_call_iterations + 1,
    }

def convert_tools_messages_to_ai_and_human(implementation_research_scratchpad: List[AnyMessage]):
    messages = []
    for message in implementation_research_scratchpad:
        if message.type == "ai":
            if message.tool_calls:
                tool_name = message.tool_calls[0]["name"]
                tool_args = json.dumps(message.tool_calls[0]["args"])
                messages.append(AIMessage(content=f"I want to call the tool {tool_name} with the following arguments: {tool_args}"))
            else:
                messages.append(message)
        elif message.type == "tool":
            messages.append(HumanMessage(content=f"When executing Tool {message.name} \n The result was {message.content} was called"))
        else:
            messages.append(message)
    return messages

def extract_implementation_plan(state: SoftwareArchitectState, config: RunnableConfig):
    """Extract implementation plan from research findings"""
    runnable = (
        extract_implementation_prompt
        | configurable_model.with_config(get_model_config(config, EXTRACT_PLAN_MAX_TOKENS))
        | JsonOutputParser(pydantic_object=ImplementationPlan)
    )
    response = runnable.invoke({
        "research_findings": convert_tools_messages_to_ai_and_human(state.implementation_research_scratchpad),
        "codebase_structure": get_files_structure.invoke({"directory": "./workspace_repo"}),
        "output_format": JsonOutputParser(pydantic_object=ImplementationPlan).get_format_instructions()
    })
    response = ImplementationPlan(**response)
    return {"implementation_plan": response}

def should_call_tool(state: SoftwareArchitectState, config: RunnableConfig):
    """Router function to determine if tools should be called.

    Written in the shape the fork's compile-time analyser reads
    (`langgraph/graph/state.py::_extract_tool_call_branch_targets`): one `if`
    on `<message>.tool_calls` whose branches return string literals that are
    keys of this branch's path map. That is what produces the
    `any_tool_calls` / `no_tool_calls` prediction rules for this node.
    """
    configurable = Configuration.from_runnable_config(config)
    most_recent_message = state.implementation_research_scratchpad[-1]

    # Bound the react loop. Upstream had none: the loop ended only when the
    # model stopped emitting tool calls, or at the root recursion_limit of 200.
    if state.tool_call_iterations >= configurable.max_react_tool_calls:
        return "implement_plan"

    if not most_recent_message.tool_calls:
        return "implement_plan"

    return "should_call_tool"

def should_conduct_research(state: SoftwareArchitectState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    # Bound the outer research loop the same way ODR bounds its supervisor:
    # `research_iterations > configurable.max_researcher_iterations`. Without
    # it, `plan_is_not_valid` can bounce back to planning indefinitely.
    if state.research_iterations > configurable.max_researcher_iterations:
        return "plan_is_valid"
    if state.is_valid_research_step:
        return "plan_is_valid"
    else:
        return "plan_is_not_valid"

class SoftwareArchitectInput(TypedDict):
    implementation_research_scratchpad: List[AnyMessage]

class SoftwareArchitectOutput(TypedDict):
    implementation_plan: Optional[ImplementationPlan]

workflow = StateGraph(SoftwareArchitectState,
                      input=SoftwareArchitectInput,
                      output=SoftwareArchitectOutput)

# Define all workflow nodes
workflow.add_node("come_up_with_research_next_step", come_up_with_research_next_step)
workflow.add_node("check_research_step", check_research_step)
workflow.add_node("conduct_research", conduct_research)
workflow.add_node("extract_implementation_plan", extract_implementation_plan)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "come_up_with_research_next_step")
workflow.add_edge("come_up_with_research_next_step", "check_research_step")
workflow.add_conditional_edges(
    "check_research_step",
    should_conduct_research,
    {
        "plan_is_valid": "conduct_research",
        "plan_is_not_valid": "come_up_with_research_next_step"
    }
)
workflow.add_conditional_edges(
    "conduct_research",
    should_call_tool,
    {
        "should_call_tool": "tools",
        "implement_plan": "extract_implementation_plan",
    }
)
workflow.add_edge("tools", "conduct_research")
workflow.add_edge("extract_implementation_plan", END)

swe_architect = workflow.compile().with_config({"tags": ["research-agent-v3"]})
