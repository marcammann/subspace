from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from subspace.models.request import CreateResponseRequest
from subspace.models.response import ResponseResource

request_state: ContextVar[dict[str, Any]] = ContextVar("request_state")


class RequestContext(BaseModel):
    response_id: str
    response: ResponseResource
    app: Any = None
    deps: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    _raw_request: CreateResponseRequest = PrivateAttr()
    _instruction_layers: list[str] = PrivateAttr(default_factory=list)
    _resolved_request: CreateResponseRequest | None = PrivateAttr(default=None)
    _state: dict[str, Any] = PrivateAttr(default_factory=dict)
    _request_called: set[int] = PrivateAttr(default_factory=set)
    _locals: dict[int, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, *, request: CreateResponseRequest, **data: Any) -> None:
        super().__init__(**data)
        self._raw_request = request

    @property
    def request(self) -> CreateResponseRequest:
        if not self._instruction_layers:
            return self._raw_request
        if self._resolved_request is not None:
            return self._resolved_request
        self._resolved_request = self._raw_request.model_copy(
            update={"instructions": self._resolve_instructions()}
        )
        return self._resolved_request

    @request.setter
    def request(self, value: CreateResponseRequest) -> None:
        self._raw_request = value
        self._resolved_request = None

    def add_instructions(self, template: str) -> None:
        self._instruction_layers.append(template)
        self._resolved_request = None

    def _resolve_instructions(self) -> str | None:
        result = self._raw_request.instructions or ""
        for template in reversed(self._instruction_layers):
            if "{instructions}" in template:
                result = template.replace("{instructions}", result)
            else:
                result = f"{result}\n\n{template}".strip() if result else template
        return result or None

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def local(self, middleware: Any) -> Any:
        key = id(middleware)
        if key not in self._locals:
            self._locals[key] = type(middleware).local_factory()
        return self._locals[key]

    def prepared(self, middleware: Any) -> bool:
        return id(middleware) in self._request_called

    def mark_prepared(self, middleware: Any) -> None:
        self._request_called.add(id(middleware))
