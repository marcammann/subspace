from typing import Protocol

from subspace.models.items import AnyItem, OutputItem


class Storage(Protocol):
    async def save_response(
        self,
        response_id: str,
        input_items: list[AnyItem],
        output_items: list[OutputItem],
    ) -> None: ...

    async def load_history(
        self,
        response_id: str,
    ) -> list[AnyItem | OutputItem]: ...
