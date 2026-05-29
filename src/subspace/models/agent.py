from enum import StrEnum

from pydantic import BaseModel, Field


class AgentRuntime(StrEnum):
    """Runtime ownership model for an agent backend."""

    INLINE = "inline"
    PROVISIONED = "provisioned"


class AgentCapabilities(BaseModel):
    """Typed feature flags describing an effective backend/middleware chain."""

    streaming: bool = True
    text_input: bool = True
    image_input: bool = False
    function_tools: bool = False
    server_tools: bool = False
    conversation_history: bool = False
    delegation: bool = False
    multiple_delegations: bool = False
    provisioned_runtime: bool = False
    custom_events: bool = True

    def require(self, requirement: "CapabilityRequirement") -> list[str]:
        """Return the missing fields required by a router or middleware."""
        missing: list[str] = []
        if requirement.streaming and not self.streaming:
            missing.append("streaming")
        if requirement.text_input and not self.text_input:
            missing.append("text_input")
        if requirement.image_input and not self.image_input:
            missing.append("image_input")
        if requirement.function_tools and not self.function_tools:
            missing.append("function_tools")
        if requirement.server_tools and not self.server_tools:
            missing.append("server_tools")
        if requirement.conversation_history and not self.conversation_history:
            missing.append("conversation_history")
        if requirement.delegation and not self.delegation:
            missing.append("delegation")
        if requirement.multiple_delegations and not self.multiple_delegations:
            missing.append("multiple_delegations")
        if requirement.provisioned_runtime and not self.provisioned_runtime:
            missing.append("provisioned_runtime")
        if requirement.custom_events and not self.custom_events:
            missing.append("custom_events")
        return missing


class CapabilityRequirement(BaseModel):
    """Typed feature requirements for router/backend compatibility checks."""

    streaming: bool = False
    text_input: bool = False
    image_input: bool = False
    function_tools: bool = False
    server_tools: bool = False
    conversation_history: bool = False
    delegation: bool = False
    multiple_delegations: bool = False
    provisioned_runtime: bool = False
    custom_events: bool = False

    def enabled_fields(self) -> list[str]:
        """Return capability field names that are required."""
        fields: list[str] = []
        if self.streaming:
            fields.append("streaming")
        if self.text_input:
            fields.append("text_input")
        if self.image_input:
            fields.append("image_input")
        if self.function_tools:
            fields.append("function_tools")
        if self.server_tools:
            fields.append("server_tools")
        if self.conversation_history:
            fields.append("conversation_history")
        if self.delegation:
            fields.append("delegation")
        if self.multiple_delegations:
            fields.append("multiple_delegations")
        if self.provisioned_runtime:
            fields.append("provisioned_runtime")
        if self.custom_events:
            fields.append("custom_events")
        return fields


class Skill(BaseModel):
    name: str
    description: str


class AgentCard(BaseModel):
    name: str
    description: str = ""
    skills: list[Skill] = Field(default_factory=list)
    runtime: AgentRuntime = AgentRuntime.INLINE
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
