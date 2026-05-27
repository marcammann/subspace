from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from subspace.models.request import CreateResponseRequest
    from subspace.models.response import ResponseResource

request_state: ContextVar[dict[str, Any]] = ContextVar("request_state")


@dataclass
class RequestContext:
    request: "CreateResponseRequest"
    response_id: str
    response: "ResponseResource"
    app: Any = None
    deps: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _request_called: set[int] = field(default_factory=set, init=False, repr=False)
    _locals: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        pass

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
