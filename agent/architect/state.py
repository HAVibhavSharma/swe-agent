from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from agent.common.entities import ImplementationPlan


class SoftwareArchitectState(BaseModel):
    research_next_step: Optional[str] = Field(None, description="The next research step to be conducted")
    implementation_plan: Optional[ImplementationPlan] = Field(None, description="The implementation plan to be executed")
    implementation_research_scratchpad: Annotated[list[AnyMessage], add_messages] = Field([], description="The scratchpad for implementation research")
    is_valid_research_step: Optional[bool] = Field(None, description="Whether the research step is valid")
    # Loop counters. Both are read by the routers against `Configuration`
    # limits, which is the shape the fork's compile-time analyser turns into a
    # `state_over_config_limit` prediction rule -- see
    # agent/common/configuration.py for why the names are ODR's.
    research_iterations: int = Field(0, description="Completed `conduct_research` turns")
    tool_call_iterations: int = Field(0, description="Consecutive tool round-trips in the research react loop")
